"""
vlm_steps.py

Two-phase offline VLM extraction:

    Phase 1 — Vision (image → raw text)
        Local vLLM in-process via llm.chat().
        No server required.

    Phase 2 — Structure (raw text → fields)
        DSPy Predict using the same vLLM model as a
        text-only in-process adapter.
        Configured lazily after the model loads.
        Falls back to JSON parsing if DSPy fails.

Output fields:
    title / subheadline / author / body
"""

import gc
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# Result container
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
# Vision prompt — image → raw text
# ----------------------------------------------------------------------

VISION_PROMPT = """
You are an expert OCR system specialized in digitizing Peruvian newspapers from the Grupo El Comercio.
The input is a digital newspaper page. Pages may contain multiple independent articles arranged in several columns, as well as editorials, opinion pieces, advertisements, public notices, captions, tables, legal announcements, classifieds, and other printed material.
Your task is to produce a complete and faithful transcription of every visible piece of text on the page.

GENERAL RULES
- Transcribe ALL readable text exactly as printed.
- Preserve the original spelling, capitalization, punctuation, accents, abbreviations, and formatting whenever possible.
- Do NOT modernize the language.
- Do NOT correct spelling or grammar.
- Do NOT translate the text.
- Do NOT summarize or explain anything.
- Never invent, infer, or complete missing words.

READING ORDER
- Follow the natural reading order of the newspaper.
- Read from top to bottom and left to right, respecting the column structure.
- When multiple articles are present, transcribe them sequentially in reading order.
- Preserve paragraph breaks whenever possible.
- Preserve line breaks within headlines when they are visually significant.

TEXT TO INCLUDE
Include every readable textual element, including:
- headlines
- subheadlines
- bylines
- article body
- captions
- editorial notes
- opinion columns
- public notices
- tables
- classifieds
- page continuation notes
- section headers
- dates
- page numbers when readable

IGNORE
Do NOT describe:
- photographs
- illustrations
- logos
- decorative borders
- ornaments
- page layout
- font styles
- advertisements

Only transcribe visible text.

UNCERTAIN TEXT
If a word cannot be read with confidence:

- transcribe the readable portion;
- replace unreadable characters with [UNCLEAR];
- never guess the missing text.

STRUCTURE
Whenever a clearly distinct text block or article ends and another begins,
insert exactly this separator on its own line:

=== SECTION BREAK ===

OUTPUT

Return ONLY the transcription.

Do not include explanations, comments, Markdown, or JSON.
"""


# ----------------------------------------------------------------------
# DSPy structured extraction — raw text → fields
# ----------------------------------------------------------------------

class DSPyArticleExtraction:
    """
    Converts raw VLM transcription into structured article fields
    using DSPy Predict.

    Instantiated after vLLM loads so DSPy's LM is already configured
    when the first Predict call is made.

    Falls back to JSON parsing if DSPy raises.
    """

    def __init__(self):
        import dspy

        class ArticleExtraction(dspy.Signature):
            """
            Extract structured newspaper article metadata from OCR text.
            Return empty strings when a field is not present.
            Do not invent information.
            """
            ocr_text = dspy.InputField(
                desc="Raw VLM transcription of a scanned newspaper article."
            )
            articles = dspy.OutputField(
                desc="""
                Return a JSON array of all newspaper articles found in the OCR text.

                Each element must contain:
                - title
                - subheadline
                - author
                - body

                Preserve reading order.
                Return [] if no articles exist.
                """
            )

        self.predictor = dspy.Predict(ArticleExtraction)

    def _parse_dspy_articles(self, raw_text: str) -> List[ArticleResult]:
        """
        Parse a JSON array of articles into ArticleResult objects.

        Expected format:

        [
            {
                "title": "...",
                "subheadline": "...",
                "author": "...",
                "body": "..."
            },
            ...
        ]

        Returns an empty list if parsing fails.
        """

        text = (raw_text or "").strip()

        # Remove Markdown code fences if present.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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

        for i, article in enumerate(data):

            if not isinstance(article, dict):
                continue

            articles.append(
                ArticleResult(
                    article_index=i,
                    title=str(article.get("title", "")).strip(),
                    subheadline=str(article.get("subheadline", "")).strip(),
                    author=str(article.get("author", "")).strip(),
                    body=str(article.get("body", "")).strip(),
                )
            )

        return articles

    def extract(self, raw_text: str):
        if not raw_text:
            return "", "", "", ""

        try:
            result = self.predictor(ocr_text=raw_text)
            return self._parse_dspy_articles(
                getattr(result, "articles", "")
            )
        except Exception:
            return self._parse_dspy_articles(raw_text)


# ----------------------------------------------------------------------
# JSON fallback parser
# ----------------------------------------------------------------------

def _parse_fields(raw_text: str):
    """Parse model output into (title, subheadline, author, body).

    Tries JSON first, falls back to treating the first line as title
    and the rest as body.
    """
    text = (raw_text or "").strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        return (
            str(obj.get("title", "")).strip(),
            str(obj.get("subheadline", "")).strip(),
            str(obj.get("author", "")).strip(),
            str(obj.get("body", "")).strip(),
        )
    except (json.JSONDecodeError, AttributeError):
        lines = text.splitlines()
        if not lines:
            return "", "", "", ""
        return lines[0].strip(), "", "", "\n".join(lines[1:]).strip()


# ----------------------------------------------------------------------
# Image conversion
# ----------------------------------------------------------------------

def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(image[:, :, ::-1])


# ----------------------------------------------------------------------
# Base VLM extractor
# ----------------------------------------------------------------------

class _BaseVLLMExtractor:
    """
    Two-phase local extraction:

        Phase 1  llm.chat()  image → raw text   (vision, in-process vLLM)
        Phase 2  DSPy        raw text → fields  (text-only, same model)

    DSPy is configured after vLLM loads so it can reuse the same
    in-process model without starting a server.
    """

    model_id: str = ""

    def __init__(
        self,
        server_url: Optional[str] = None,
        use_local: bool = True,
        max_new_tokens: int = 2048,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        **engine_kwargs: Any,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs
        self._llm = None
        self._sampling_params = None
        self._dspy_extractor = None  # initialized after vLLM loads

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
            limit_mm_per_prompt={"image": 1},
        )
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        llm_kwargs.update(self.engine_kwargs)

        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )

        # Configure DSPy to use the same in-process vLLM model
        # for text-only structured extraction.
        self._configure_dspy()
        self._dspy_extractor = DSPyArticleExtraction()

    def _configure_dspy(self):
        """Wire DSPy to the already-loaded vLLM model.

        Creates a minimal text-only adapter so DSPy can call the model
        without a separate HTTP server.
        """
        import dspy

        vllm_instance = self._llm

        class _VLLMTextAdapter:
            """Minimal DSPy LM interface backed by an in-process vLLM model."""

            def __init__(self):
                from vllm import SamplingParams
                self._params = SamplingParams(temperature=0.0, max_tokens=1024)
                self.kwargs = {}
                self.history = []

            def __call__(self, prompt=None, messages=None, **kwargs):
                if messages:
                    prompt = "\n".join(
                        f"{m['role']}: {m['content']}"
                        for m in messages
                        if isinstance(m.get("content"), str)
                    )
                outputs = vllm_instance.generate([prompt or ""], self._params)
                text = (
                    outputs[0].outputs[0].text
                    if outputs and outputs[0].outputs
                    else ""
                )
                self.history.append({"prompt": prompt, "response": text})
                return [text]

            # DSPy 2.x calls basic_request for some backends
            def basic_request(self, prompt, **kwargs):
                return self(prompt=prompt)

        dspy.settings.configure(lm=_VLLMTextAdapter())

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self):
        if self._llm is None:
            return

        try:
            engine = getattr(self._llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
        except Exception:
            pass

        del self._llm  # explicit delete before gc
        self._llm = None
        self._sampling_params = None
        self._dspy_extractor = None

        gc.collect()

        try:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # release IPC memory handles
        except Exception:
            pass

        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass

        gc.collect()  # second pass after destroying parallel state

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray, metadata: Dict[str, Any]) -> ExtractionResult:
        return self.extract_batch([image], [metadata])[0]

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        images: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        self._ensure_loaded()

        # ------------------------------------------------------------------
        # Phase 1: Vision  image → raw text via llm.chat()
        # ------------------------------------------------------------------
        conversations = []
        for image in images:
            pil_image = _to_pil(image)
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": pil_image},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }])

        start = time.time()
        try:
            outputs = self._llm.chat(conversations, self._sampling_params)
            raw_texts = [
                output.outputs[0].text if output.outputs else ""
                for output in outputs
            ]
        except Exception as exc:
            elapsed = time.time() - start
            per_item = elapsed / max(len(metadata_list), 1)
            return [
                ExtractionResult(
                    elapsed_s=per_item,
                    status="failed",
                    error=str(exc),
                    metadata=metadata,
                )
                for metadata in metadata_list
            ]

        elapsed_vision = time.time() - start

        # ------------------------------------------------------------------
        # Phase 2: DSPy  raw text → structured fields
        # ------------------------------------------------------------------
        results = []
        for raw_text, metadata in zip(raw_texts, metadata_list):
            per_item = elapsed_vision / max(len(raw_texts), 1)
            try:
                articles = self._dspy_extractor.extract(raw_text)
                results.append(
                    ExtractionResult(
                        articles=articles,
                        raw_text=raw_text,
                        elapsed_s=per_item,
                        status="success",
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                results.append(
                    ExtractionResult(
                        articles=[],
                        raw_text=raw_text,
                        elapsed_s=per_item,
                        status="failed",
                        error=str(exc),
                        metadata=metadata,
                    )
                )

        return results


# ----------------------------------------------------------------------
# Concrete extractors
# ----------------------------------------------------------------------

class OlmOCRExtractor(_BaseVLLMExtractor):
    model_id = "allenai/olmOCR-2-7B-1025-FP8"


class RolmOCRExtractor(_BaseVLLMExtractor):
    model_id = "AccsoAndreBuesgen/RolmOCR-bnb-4bit"


class NanonetsOCRExtractor(_BaseVLLMExtractor):
    model_id = "sayed0am/Nanonets-OCR2-3B-FP8-Dynamic"


VLM_EXTRACTORS = {
    "olmocr": OlmOCRExtractor,
    "rolmocr": RolmOCRExtractor,
    "nanonets": NanonetsOCRExtractor,
}