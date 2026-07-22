"""
vlm_steps.py

Offline VLM extraction using in-process vLLM.

Architecture:

    image
      ↓
    local VLM
      ↓
    generated text
      ↓
    DSPy structured extraction
      ↓
    title
    subheadline
    author
    body

No vLLM server is used.

DSPy is currently used without optimization/fine-tuning.
Later, a DSPy optimizer can be added using the gold-standard dataset.
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
class ExtractionResult:
    title: str = ""
    subheadline: str = ""
    author: str = ""
    body: str = ""
    raw_text: str = ""
    elapsed_s: float = 0.0
    status: str = "failed"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        out = dict(self.metadata)

        out.update(
            {
                "title": self.title,
                "subheadline": self.subheadline,
                "author": self.author,
                "body": self.body,
                "raw_text": self.raw_text,
                "elapsed_s": self.elapsed_s,
                "status": self.status,
                "error": self.error,
            }
        )

        return out


# ----------------------------------------------------------------------
# DSPy structured extraction
# ----------------------------------------------------------------------

class DSPyArticleExtraction:

    """
    DSPy module responsible for converting raw OCR/VLM text into
    structured article fields.

    This is intentionally NOT optimized yet.

    Later we can optimize this module against the gold-standard
    dataset using DSPy optimizers.
    """

    def __init__(self):

        import dspy

        self.dspy = dspy

        class ArticleExtraction(
            dspy.Signature
        ):
            """
            Extract structured newspaper article metadata
            from OCR text.

            Return empty strings when a field is not present.
            Do not invent information.
            """

            ocr_text = dspy.InputField(
                desc=(
                    "Raw OCR or VLM transcription "
                    "of a scanned newspaper article."
                )
            )

            title = dspy.OutputField(
                desc=(
                    "The article headline. "
                    "Return an empty string if "
                    "no headline is visible."
                )
            )

            subheadline = dspy.OutputField(
                desc=(
                    "The article subheadline, "
                    "standfirst, or deck. "
                    "Return an empty string if absent."
                )
            )

            author = dspy.OutputField(
                desc=(
                    "The article author or byline. "
                    "Return an empty string if absent."
                )
            )

            body = dspy.OutputField(
                desc=(
                    "The full article body text. "
                    "Preserve paragraph breaks."
                )
            )

        self.predictor = dspy.Predict(
            ArticleExtraction
        )

    def extract(
        self,
        raw_text: str,
    ):

        if not raw_text:
            return (
                "",
                "",
                "",
                "",
            )

        try:

            result = self.predictor(
                ocr_text=raw_text
            )

            return (
                self._clean(
                    getattr(
                        result,
                        "title",
                        "",
                    )
                ),
                self._clean(
                    getattr(
                        result,
                        "subheadline",
                        "",
                    )
                ),
                self._clean(
                    getattr(
                        result,
                        "author",
                        "",
                    )
                ),
                self._clean(
                    getattr(
                        result,
                        "body",
                        "",
                    )
                ),
            )

        except Exception:

            # If DSPy parsing fails, preserve the
            # raw VLM output as body text.
            return (
                "",
                "",
                "",
                raw_text.strip(),
            )

    @staticmethod
    def _clean(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return str(
            value
        ).strip()


# ----------------------------------------------------------------------
# Image conversion
# ----------------------------------------------------------------------

def _to_pil(
    image: np.ndarray,
) -> Image.Image:

    if image.ndim == 2:

        return Image.fromarray(
            image
        )

    return Image.fromarray(
        image[:, :, ::-1]
    )


# ----------------------------------------------------------------------
# Base VLM extractor
# ----------------------------------------------------------------------

class _BaseVLLMExtractor:

    """
    Base class for local in-process VLM inference.

    Important:
    - No vLLM server.
    - Models are loaded locally.
    - Uses LLM.generate().
    - DSPy performs structured extraction after generation.
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

        self.max_new_tokens = (
            max_new_tokens
        )

        self.max_model_len = (
            max_model_len
        )

        self.gpu_memory_utilization = (
            gpu_memory_utilization
        )

        self.tensor_parallel_size = (
            tensor_parallel_size
        )

        self.dtype = dtype

        self.engine_kwargs = (
            engine_kwargs
        )

        self._llm = None
        self._sampling_params = None

        # DSPy structured extraction.
        self._dspy_extractor = (
            DSPyArticleExtraction()
        )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    def _ensure_loaded(self):

        if self._llm is not None:
            return

        from vllm import (
            LLM,
            SamplingParams,
        )

        llm_kwargs = {

            "model": self.model_id,

            "trust_remote_code": True,

            "dtype": self.dtype,

            "tensor_parallel_size":
                self.tensor_parallel_size,

            "gpu_memory_utilization":
                self.gpu_memory_utilization,

            "limit_mm_per_prompt": {
                "image": 1
            },
        }

        if self.max_model_len is not None:

            llm_kwargs[
                "max_model_len"
            ] = self.max_model_len

        llm_kwargs.update(
            self.engine_kwargs
        )

        self._llm = LLM(
            **llm_kwargs
        )

        self._sampling_params = (
            SamplingParams(
                temperature=0.0,
                max_tokens=(
                    self.max_new_tokens
                ),
            )
        )

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self):

        if self._llm is None:
            return

        try:

            engine = getattr(
                self._llm,
                "llm_engine",
                None,
            )

            if (
                engine is not None
                and hasattr(
                    engine,
                    "shutdown",
                )
            ):

                engine.shutdown()

        except Exception:
            pass

        self._llm = None
        self._sampling_params = None

        gc.collect()

        try:

            import torch

            torch.cuda.empty_cache()

            torch.cuda.synchronize()

        except Exception:
            pass

        try:

            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
            )

            destroy_model_parallel()

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:

        return self.extract_batch(
            [image],
            [metadata],
        )[0]

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        images: List[np.ndarray],
        metadata_list: List[
            Dict[str, Any]
        ],
    ) -> List[ExtractionResult]:

        self._ensure_loaded()

        prompts = []

        for image in images:

            pil_image = _to_pil(
                image
            )

            prompts.append(
                {
                    "prompt": (
                        "<|user|>\n"
                        "You are transcribing a scanned "
                        "newspaper article.\n\n"
                        "Read the provided newspaper image "
                        "and transcribe the article as accurately "
                        "as possible.\n\n"
                        "Include the headline, subheadline, "
                        "author/byline, and complete article body "
                        "when visible.\n\n"
                        "Do not summarize.\n"
                        "Do not invent missing information.\n"
                        "Preserve the original article text.\n"
                        "<|assistant|>\n"
                    ),
                    "multi_modal_data": {
                        "image": pil_image
                    },
                }
            )

        start = time.time()

        try:

            outputs = self._llm.generate(
                prompts,
                self._sampling_params,
            )

            elapsed_total = (
                time.time()
                - start
            )

            per_item = (
                elapsed_total
                / max(
                    len(outputs),
                    1,
                )
            )

            results = []

            for (
                output,
                metadata,
            ) in zip(
                outputs,
                metadata_list,
            ):

                raw_text = ""

                if output.outputs:

                    raw_text = (
                        output
                        .outputs[0]
                        .text
                    )

                # --------------------------------------------------
                # DSPy structured extraction
                # --------------------------------------------------

                (
                    title,
                    subheadline,
                    author,
                    body,
                ) = (
                    self._dspy_extractor.extract(
                        raw_text
                    )
                )

                results.append(
                    ExtractionResult(

                        title=title,

                        subheadline=(
                            subheadline
                        ),

                        author=author,

                        body=body,

                        raw_text=raw_text,

                        elapsed_s=(
                            per_item
                        ),

                        status="success",

                        error=None,

                        metadata=metadata,
                    )
                )

            return results

        except Exception as exc:

            elapsed_total = (
                time.time()
                - start
            )

            per_item = (
                elapsed_total
                / max(
                    len(metadata_list),
                    1,
                )
            )

            return [

                ExtractionResult(

                    title="",

                    subheadline="",

                    author="",

                    body="",

                    raw_text="",

                    elapsed_s=(
                        per_item
                    ),

                    status="failed",

                    error=str(
                        exc
                    ),

                    metadata=metadata,
                )

                for metadata
                in metadata_list
            ]


# ----------------------------------------------------------------------
# Concrete extractors
# ----------------------------------------------------------------------

class OlmOCRExtractor(
    _BaseVLLMExtractor
):

    model_id = (
        "allenai/olmOCR-2-7B-1025-FP8"
    )


class RolmOCRExtractor(
    _BaseVLLMExtractor
):

    model_id = (
        "AccsoAndreBuesgen/RolmOCR-bnb-4bit"
    )


class NanonetsOCRExtractor(
    _BaseVLLMExtractor
):

    model_id = (
        "sayed0am/Nanonets-OCR2-3B-FP8-Dynamic"
    )


VLM_EXTRACTORS = {

    "olmocr":
        OlmOCRExtractor,

    "rolmocr":
        RolmOCRExtractor,

    "nanonets":
        NanonetsOCRExtractor,
}