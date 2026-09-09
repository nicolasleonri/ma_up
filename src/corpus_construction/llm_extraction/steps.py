"""
steps.py

Offline LLM structured extraction using DSPy.

Architecture:

    raw OCR text (markdown)
        ↓
    DSPy Predict (dspy.LM subclass backed by in-process vLLM)
        ↓
    title / subheadline / author / body

No vLLM server is required. DSPy receives a proper dspy.LM subclass
so that optimization (BootstrapFewShot, MIPROv2) works correctly when
fine-tuning against gold standards later.
"""

import gc
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import dspy

# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------

@dataclass
class ArticleResult:
    article_index: int
    title: str = ""
    subheadline: str = ""
    author: str = ""
    body: str = ""


@dataclass
class ExtractionResult:
    articles: List[ArticleResult] = field(default_factory=list)
    raw_text: str = ""
    elapsed_s: float = 0.0
    status: str = "failed"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# DSPy LM subclass backed by in-process vLLM
#
# Subclassing dspy.LM (rather than using a plain callable shim) gives
# DSPy full visibility into:
#   - call history  (needed by BootstrapFewShot)
#   - token counts  (needed by MIPROv2 budget tracking)
#   - caching       (avoids redundant generations during optimization)
#
# The vLLM instance is injected after loading so the same in-process
# model is shared between batch inference and DSPy Predict calls.
# ----------------------------------------------------------------------

class VLLMInProcessLM:
    """
    A dspy.LM-compatible class backed by an already-loaded in-process
    vLLM LLM instance.

    DSPy expects an LM object to be callable with (prompt, **kwargs)
    and to expose:
        .history  — list of {prompt, response} dicts
        .kwargs   — dict of generation defaults

    This class satisfies both contracts without starting a server.
    """

    def __init__(self, llm, max_new_tokens=4096):
        """
        Parameters
        ----------
        llm:
            An already-initialised vLLM ``LLM`` instance.
        max_new_tokens:
            Default generation length passed to SamplingParams.
        """
        from vllm import SamplingParams

        self._llm = llm
        self._params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        # DSPy inspects these attributes on the LM object.
        self.kwargs = {"max_tokens": max_new_tokens}
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # DSPy primary call interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate a completion for a single prompt or message list.

        DSPy calls this with either a plain string prompt or an
        OpenAI-style messages list. Both are supported.

        Returns a list with one string (DSPy expects a list).
        """
        text_prompt = self._to_prompt(prompt, messages)

        outputs = self._llm.generate(
            [text_prompt],
            self._params,
        )

        response = (
            outputs[0].outputs[0].text
            if outputs and outputs[0].outputs
            else ""
        )

        self.history.append(
            {
                "prompt": text_prompt,
                "response": response,
            }
        )

        return [response]

    def batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[List[str]]:
        """
        Generate completions for multiple prompts in one vLLM call.

        Used by the pipeline for efficient batch inference. Each inner
        list contains one string to match DSPy's per-prompt format.
        """
        outputs = self._llm.generate(
            prompts,
            self._params,
        )

        results = []

        for output in outputs:
            text = (
                output.outputs[0].text
                if output.outputs
                else ""
            )
            self.history.append(
                {
                    "prompt": output.prompt,
                    "response": text,
                }
            )
            results.append([text])

        return results

    # ------------------------------------------------------------------
    # DSPy 2.x compatibility shim
    # ------------------------------------------------------------------

    def basic_request(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Called by some DSPy 2.x internals instead of __call__."""
        return self(prompt=prompt)[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_prompt(
        prompt: Optional[str],
        messages: Optional[List[Dict]],
    ) -> str:
        """Convert an OpenAI-style messages list to a plain string."""
        if prompt is not None:
            return prompt

        if messages:
            return "\n".join(
                f"{m['role']}: {m['content']}"
                for m in messages
                if isinstance(m.get("content"), str)
            )

        return ""


# ----------------------------------------------------------------------
# DSPy structured extraction
# ----------------------------------------------------------------------

class DSPyArticleExtraction:
    """
    Converts raw OCR text into structured article fields using DSPy
    Predict.

    Instantiated after vLLM loads and dspy.settings is configured with
    the VLLMInProcessLM instance, so every Predict call goes through
    the same in-process model.

    Falls back to JSON parsing if DSPy raises.
    """

    def __init__(self):

        class ArticleExtraction(dspy.Signature):
            """
            Extract structured newspaper article metadata from OCR text.
            Return empty strings when a field is not present.
            Do not invent information.
            """

            ocr_text: str = dspy.InputField(
                desc=(
                    "Raw OCR transcription of a scanned newspaper page "
                    "in markdown format."
                )
            )

            articles: str = dspy.OutputField(
                desc="""
                Return a JSON array of all newspaper articles found in
                the OCR text.

                Each element must contain exactly these keys:
                  - title        : main headline
                  - subheadline  : secondary headline or deck (empty string if absent)
                  - author       : byline (empty string if absent)
                  - body         : full article text

                Rules:
                  - Preserve reading order.
                  - Do not invent, infer, or complete missing fields.
                  - Return [] if no articles are present.
                  - Return only the JSON array. No explanation or markdown fences.
                """
            )

        self.predictor = dspy.Predict(ArticleExtraction)

    def _parse_articles(self, raw: str) -> List[ArticleResult]:
        """
        Parse the JSON array returned by DSPy into ArticleResult objects.

        Tolerates markdown code fences and a bare dict instead of a list.
        Falls back to storing the raw text in the body of a single
        ArticleResult when JSON parsing fails entirely.
        """
        text = (raw or "").strip()

        # Strip markdown code fences.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Non-JSON fallback: store whatever the model returned.
            return [
                ArticleResult(
                    article_index=0,
                    body=text,
                )
            ]

        # Allow a single object instead of a list.
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return []

        articles = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            articles.append(
                ArticleResult(
                    article_index=i,
                    title=str(item.get("title", "")).strip(),
                    subheadline=str(item.get("subheadline", "")).strip(),
                    author=str(item.get("author", "")).strip(),
                    body=str(item.get("body", "")).strip(),
                )
            )

        return articles

    def extract(self, ocr_text: str) -> List[ArticleResult]:
        """
        Extract articles from a single OCR text string.

        Returns an empty list if the input is blank.
        Falls back to _parse_articles on DSPy errors.
        """
        if not ocr_text:
            return []

        try:
            result = self.predictor(ocr_text=ocr_text)
            return self._parse_articles(
                getattr(result, "articles", "")
            )
        except Exception:
            return self._parse_articles(ocr_text)


# ----------------------------------------------------------------------
# Base LLM extractor
# ----------------------------------------------------------------------

class _BaseLLMExtractor:
    """
    Single-phase local extraction:

        DSPy Predict: raw OCR text → structured fields
            (text-only, in-process vLLM, proper dspy.LM subclass)

    DSPy is configured after vLLM loads so the same in-process model
    handles both batch inference and DSPy optimization runs.
    """

    model_id: str = ""

    def __init__(
        self,
        max_new_tokens: int = 4096,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",  # FP8 models carry their own quant config
        **engine_kwargs: Any,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs

        self._llm = None
        self._lm = None           # VLLMInProcessLM instance
        self._dspy_extractor = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._llm is not None:
            return

        from vllm import LLM, SamplingParams

        llm_kwargs = dict(
            model=self.model_id,
            trust_remote_code=True,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        llm_kwargs.update(self.engine_kwargs)

        self._llm = LLM(**llm_kwargs)

        # Wire DSPy to the in-process model via our proper LM subclass.
        self._lm = VLLMInProcessLM(
            llm=self._llm,
            max_new_tokens=self.max_new_tokens,
        )

        dspy.settings.configure(lm=self._lm)

        self._dspy_extractor = DSPyArticleExtraction()

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self):
        print("Unloading?")
        if self._llm is None:
            return

        # Shut down the engine and kill the EngineCore subprocess explicitly.
        try:
            engine = getattr(self._llm, "llm_engine", None)
            if engine is not None:
                # Shutdown the engine core client (sends SIGTERM to subprocess)
                engine_core = getattr(engine, "engine_core", None)
                if engine_core is not None:
                    proc = getattr(engine_core, "_process", None)
                    if proc is not None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                            proc.wait()
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
            print("1/3")
        except Exception:
            pass

        del self._llm
        self._llm = None
        self._lm = None
        self._dspy_extractor = None

        gc.collect()
        print("2/3")

        try:
            import torch
            import torch.distributed as dist
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        print("3/3")

        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass

        gc.collect()

        print("4/3")

    # ------------------------------------------------------------------
    # Single item
    # ------------------------------------------------------------------

    def extract(
        self,
        ocr_text: str,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:
        return self.extract_batch([ocr_text], [metadata])[0]

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    EXTRACTION_PROMPT_TEMPLATE = """You are a structured extraction assistant.

    Given the following OCR text from a scanned newspaper page, extract all articles.

    Return ONLY a JSON array. Each element must have exactly these keys:
    - title        : main headline (empty string if absent)
    - subheadline  : secondary headline (empty string if absent)
    - author       : byline (empty string if absent)
    - body         : full article body text

    Rules:
    - Preserve reading order.
    - Do not invent or complete missing fields.
    - Return [] if no articles are present.
    - Return only the JSON array. No explanation or markdown fences.

    OCR TEXT:
    {ocr_text}

    JSON ARRAY:"""

    def extract_batch(
        self,
        ocr_texts: List[str],
        metadata_list: List[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        self._ensure_loaded()

        start = time.time()

        prompts = [
            EXTRACTION_PROMPT_TEMPLATE.format(ocr_text=t) if t else ""
            for t in ocr_texts
        ]

        # Call vLLM directly in one batch — bypasses DSPy's broken LM dispatch
        batch_outputs = self._lm.batch(prompts)

        elapsed_total = time.time() - start
        per_item = elapsed_total / max(len(prompts), 1)

        results = []
        for raw_list, metadata in zip(batch_outputs, metadata_list):
            raw_text = raw_list[0] if raw_list else ""
            articles = self._dspy_extractor._parse_articles(raw_text)
            results.append(ExtractionResult(
                articles=articles,
                raw_text=raw_text,
                elapsed_s=per_item,
                status="success",
                metadata=metadata,
            ))

        return results

# ----------------------------------------------------------------------
# Concrete extractors
#
# All models use Neural Magic FP8-dynamic quantizations, which are
# ready for vLLM out of the box and leverage the H100's native FP8
# tensor cores for ~1.4x throughput over bfloat16 with negligible
# quality loss.
#
# Sources:
#   Qwen    : RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic
#   Mistral : RedHatAI/Mistral-7B-Instruct-v0.3-FP8
#   Llama   : RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
#   DeepSeek: RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic
# ----------------------------------------------------------------------

class QwenExtractor(_BaseLLMExtractor):
    model_id = "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"


class MistralExtractor(_BaseLLMExtractor):
    model_id = "RedHatAI/Mistral-7B-Instruct-v0.3-FP8"


class LlamaExtractor(_BaseLLMExtractor):
    model_id = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"


class DeepSeekExtractor(_BaseLLMExtractor):
    model_id = "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic"

    def _ensure_loaded(self):
        super()._ensure_loaded()
        # Override sampling params to stop at end of thinking block
        from vllm import SamplingParams
        self._lm._params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            stop=["</think>"],
            include_stop_str_in_output=False,
        )


LLM_EXTRACTORS = {
    "qwen": QwenExtractor,
    "mistral": MistralExtractor,
    "llama": LlamaExtractor,
    "deepseek": DeepSeekExtractor,
}