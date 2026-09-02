"""OCR-text -> local LLM -> DSPy structured article extraction."""
from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ArticleResult:
    article_index: int
    title: str = ""
    subheadline: str = ""
    author: str = ""
    body: str = ""


class VLLMDSPyLM:
    """Small DSPy LM adapter backed by a local vLLM engine.

    DSPy modules call ``forward`` with either a prompt or chat messages. The
    adapter delegates generation to the already-loaded vLLM engine, so the
    structured extraction stage stays local and does not require a separate
    OpenAI-compatible server.
    """

    def __init__(self, engine: Any, sampling_params: Any, model_id: str):
        import dspy

        self._dspy = dspy
        self.engine = engine
        self.sampling_params = sampling_params
        super().__init__()
        self.model = model_id

    def __getattr__(self, name: str):
        # DSPy's BaseLM methods are installed dynamically below by
        # ``build_dspy_lm``. This class intentionally remains a thin adapter.
        raise AttributeError(name)


def build_dspy_lm(engine: Any, sampling_params: Any, model_id: str):
    """Create a DSPy BaseLM around an existing vLLM engine."""
    import dspy

    class LocalVLLMLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model=model_id)
            self.engine = engine
            self.sampling_params = sampling_params

        def forward(self, prompt=None, messages=None, **kwargs):
            if messages is not None:
                # vLLM's chat interface accepts OpenAI-style messages.
                outputs = self.engine.chat(
                    messages,
                    self.sampling_params,
                    use_tqdm=False,
                )
            else:
                outputs = self.engine.generate(
                    [prompt or ""], self.sampling_params, use_tqdm=False
                )
            texts = []
            for output in outputs:
                if getattr(output, "outputs", None):
                    texts.append(output.outputs[0].text)
                else:
                    texts.append("")
            return texts

    return LocalVLLMLM()


import dspy


class ArticleExtraction(dspy.Signature):
    """Extract faithful newspaper article fields from OCR text."""

    ocr_text: str = dspy.InputField(
        desc="OCR transcription from a complete newspaper page or an article crop."
    )
    articles: str = dspy.OutputField(
        desc=(
            "JSON array of article objects. Each object must contain exactly "
            "title, subheadline, author, and body. Preserve wording from the OCR."
        )
    )


def parse_articles(text: str) -> list[ArticleResult]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [ArticleResult(0, body=text)] if text else []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    result = []
    for i, article in enumerate(data):
        if isinstance(article, dict):
            result.append(
                ArticleResult(
                    i,
                    str(article.get("title", "")).strip(),
                    str(article.get("subheadline", "")).strip(),
                    str(article.get("author", "")).strip(),
                    str(article.get("body", "")).strip(),
                )
            )
    return result


class DSPyArticleExtractor:
    """DSPy program used after raw OCR transcription."""

    def __init__(self, engine: Any, sampling_params: Any, model_id: str):
        import dspy

        lm = build_dspy_lm(engine, sampling_params, model_id)
        # JSONAdapter makes the output parser explicit while retaining DSPy's
        # signature/module abstraction.
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
        self.predict = dspy.Predict(ArticleExtraction)

    def extract(self, text: str) -> list[ArticleResult]:
        prediction = self.predict(ocr_text=text)
        return parse_articles(getattr(prediction, "articles", ""))


RAW_TRANSCRIPTION_PROMPT = """You are a faithful OCR transcription system for digitized Peruvian newspapers.
Transcribe every visible textual element in the supplied newspaper image input.
Preserve wording, spelling, punctuation, line/section order, and uncertainty.
Do not correct, translate, summarize, or invent text. Do not infer missing text.
Use === SECTION BREAK === between clearly separated textual regions.
Return plain transcription text only.
"""


class LocalLLM:
    """Local vLLM transcription followed by DSPy article extraction."""

    def __init__(
        self,
        model_id,
        max_new_tokens=4096,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.kw = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )
        if max_model_len is not None:
            self.kw["max_model_len"] = max_model_len
        self.llm = None
        self.params = None
        self.dspy_extractor = None

    def _ensure(self):
        if self.llm is not None:
            return
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=self.model_id, trust_remote_code=True, **self.kw)
        # Phase 1: faithful OCR-text generation. Phase 2 uses the same model
        # through DSPy's BaseLM adapter for structured extraction.
        self.params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
        self.dspy_extractor = DSPyArticleExtractor(
            self.llm, self.params, self.model_id
        )

    def extract_batch(self, texts):
        self._ensure()
        prompts = [RAW_TRANSCRIPTION_PROMPT + "\nOCR INPUT:\n" + t for t in texts]
        outputs = self.llm.generate(prompts, self.params, use_tqdm=False)
        raw_texts = [
            o.outputs[0].text if getattr(o, "outputs", None) else "" for o in outputs
        ]
        # DSPy is deliberately kept as the structured extraction layer. The
        # raw transcription is preserved by the pipeline for diagnostics.
        return [self.dspy_extractor.extract(text) for text in raw_texts], raw_texts

    def unload(self):
        if self.llm is None:
            return
        try:
            engine = getattr(self.llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
        except Exception:
            pass
        self.llm = None
        self.params = None
        self.dspy_extractor = None
        gc.collect()
        try:
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        gc.collect()
