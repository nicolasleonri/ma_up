"""
vlm_steps.py — VLM extractor classes for offline batch text extraction.

All extractors run fully offline using vLLM's in-process LLM API.

Each extraction attempts to return four structured fields:

    - title
    - subheadline
    - author
    - body

The original model response is also preserved in raw_text so that
failed parsing or unexpected model behavior can be inspected later.

Models are loaded lazily and reused for all batches handled by one
extractor instance.
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
# Shared result container
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
    metadata: Dict[str, Any] = field(default_factory=dict)

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
# Shared extraction prompt
# ----------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "You are transcribing a scanned newspaper article. "
    "Read the image and return ONLY a JSON object with four keys: "
    '"title" (the article headline, or empty string if none is visible), '
    '"subheadline" (the article subheadline or standfirst, or empty string '
    "if none is visible), "
    '"author" (the author or byline, or empty string if none is visible), '
    '"body" (the full article body text, preserving paragraph breaks as \\n). '
    "Do not include any commentary outside the JSON object."
)


def _parse_extraction(raw_text: str):
    """Best-effort parse of model output."""
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
        return "", "", "", text


def _to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert a BGR OpenCV image to an RGB PIL image.
    """

    if image.ndim == 2:
        return Image.fromarray(
            image
        )

    return Image.fromarray(
        image[:, :, ::-1]
    )


# ----------------------------------------------------------------------
# Base vLLM extractor
# ----------------------------------------------------------------------

class _BaseVLLMExtractor:
    """
    Base class for offline vLLM inference.

    Models are loaded lazily on first extraction and reused for
    subsequent batches.
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

    def _ensure_loaded(self):
        """
        Lazily load the vLLM model.
        """

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
            "tensor_parallel_size": (
                self.tensor_parallel_size
            ),
            "gpu_memory_utilization": (
                self.gpu_memory_utilization
            ),
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

    def unload(self):
        """
        Free GPU memory before loading another VLM.
        """

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

    def extract(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:

        return self.extract_batch(
            [image],
            [metadata],
        )[0]

    def extract_batch(
        self,
        images: List[np.ndarray],
        metadata_list: List[
            Dict[str, Any]
        ],
    ) -> List[ExtractionResult]:
        """
        Run one offline vLLM batch.

        The elapsed_s value is the average wall-clock batch
        time per image. The original raw model response is
        always retained.
        """

        self._ensure_loaded()

        conversations = []

        for image in images:
            pil_image = _to_pil(
                image
            )

            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_pil",
                                "image_pil": (
                                    pil_image
                                ),
                            },
                            {
                                "type": "text",
                                "text": (
                                    EXTRACTION_PROMPT
                                ),
                            },
                        ],
                    }
                ]
            )

        start = time.time()

        try:
            outputs = self._llm.chat(
                conversations,
                self._sampling_params,
            )

            elapsed_total = (
                time.time() - start
            )

            per_item = (
                elapsed_total
                / max(
                    len(conversations),
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
                raw_text = (
                    output.outputs[0].text
                    if output.outputs
                    else ""
                )

                (
                    title,
                    subheadline,
                    author,
                    body,
                ) = _parse_extraction(
                    raw_text
                )

                results.append(
                    ExtractionResult(
                    title=title,
                    subheadline=subheadline,
                    author=author,
                    body=body,
                    raw_text=raw_text,
                    elapsed_s=per_item,
                    status="ok",
                    error=None,
                    metadata=metadata,
                    )
                )

            return results

        except Exception as exc:
            elapsed_total = (
                time.time() - start
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
                    elapsed_s=per_item,
                    status="failed",
                    error=str(exc),
                    metadata=metadata,
                )
                for metadata in metadata_list
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
    "olmocr": OlmOCRExtractor,
    "rolmocr": RolmOCRExtractor,
    "nanonets": NanonetsOCRExtractor,
}