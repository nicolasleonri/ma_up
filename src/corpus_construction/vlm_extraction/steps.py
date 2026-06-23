"""
vlm_steps.py — VLM extractor classes for offline batch text extraction.

All three extractors run fully offline via vLLM's in-process `LLM` API
(`llm.chat(...)`). No HTTP server, no open ports, no `--server-url` needed —
this is meant for HPC compute nodes (e.g. `srun --pty bash` on curta) where
you can't reach a separately-running vLLM server process.

Each extractor lazily loads its model on first use and reuses it for every
subsequent call within the same process. Use `extract_batch()` instead of
calling `extract()` in a loop whenever you can — vLLM continuously batches
everything passed to a single `generate`/`chat` call, which is where the
real throughput win comes from on a GPU node.
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
                "body": self.body,
                "raw_text": self.raw_text,
                "elapsed_s": self.elapsed_s,
                "status": self.status,
                "error": self.error,
            }
        )
        return out


# Shared prompt — ask for structured JSON so title/body can be split
# reliably regardless of which VLM produced the output.
EXTRACTION_PROMPT = (
    "You are transcribing a scanned newspaper article crop. "
    "Read the image and return ONLY a JSON object with two keys: "
    '"title" (the article headline, or empty string if none is visible) and '
    '"body" (the full body text, preserving paragraph breaks as \\n). '
    "Do not include any commentary outside the JSON object."
)


def _parse_title_body(raw_text: str):
    """Best-effort parse of model output into (title, body)."""
    text = (raw_text or "").strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        return str(obj.get("title", "")).strip(), str(obj.get("body", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        lines = text.splitlines()
        if not lines:
            return "", ""
        return lines[0].strip(), "\n".join(lines[1:]).strip()


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV array (cv2.imread output) to RGB PIL.Image."""
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(image[:, :, ::-1])


# ----------------------------------------------------------------------
# Base class — wraps a vLLM offline LLM instance
# ----------------------------------------------------------------------

class _BaseVLLMExtractor:
    """
    Base class for fully offline VLM inference via vLLM's `LLM.chat()`
    batch API. Weights load once, on first `.extract()`/`.extract_batch()`
    call, directly onto the local GPU.

    `server_url` and `use_local` kwargs are still accepted (and ignored)
    so this is a drop-in replacement for the previous server-based
    constructor signatures used in pipeline.py / vlm_extraction.py — you
    don't need to change those call sites.
    """

    model_id: str = ""  # set by subclass

    def __init__(
        self,
        server_url: Optional[str] = None,   # noqa: ARG002 — kept for API compat
        use_local: bool = True,             # noqa: ARG002 — kept for API compat
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

    # -- lazy load -------------------------------------------------
    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # heavy + optional dep, import lazily

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

    # -- teardown ----------------------------------------------------
    def unload(self):
        """
        Free this model's GPU memory. Call this before loading the next
        VLM in the same process — vLLM does not release device memory on
        its own just because the Python object goes out of scope.
        """
        if self._llm is None:
            return

        try:
            # Newer vLLM versions expose an explicit shutdown hook on the
            # engine; call it if present.
            engine = getattr(self._llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
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
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass

    # -- single-image convenience wrapper ---------------------------
    def extract(self, image: np.ndarray, metadata: Dict[str, Any]) -> ExtractionResult:
        return self.extract_batch([image], [metadata])[0]

    # -- true offline batching --------------------------------------
    def extract_batch(
        self,
        images: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        """
        Run one vLLM offline batch over all given crops at once. vLLM
        continuously batches every conversation passed here internally —
        this is the call to use for throughput, instead of looping
        `extract()` one crop at a time.
        """
        self._ensure_loaded()

        conversations = []
        for image in images:
            pil_image = _to_pil(image)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": pil_image},
                            {"type": "text", "text": EXTRACTION_PROMPT},
                        ],
                    }
                ]
            )

        start = time.time()
        try:
            outputs = self._llm.chat(conversations, self._sampling_params)
            elapsed_total = time.time() - start
            per_item = elapsed_total / max(len(conversations), 1)

            results = []
            for output, metadata in zip(outputs, metadata_list):
                raw_text = output.outputs[0].text if output.outputs else ""
                title, body = _parse_title_body(raw_text)
                results.append(
                    ExtractionResult(
                        title=title,
                        body=body,
                        raw_text=raw_text,
                        elapsed_s=per_item,
                        status="ok",
                        error=None,
                        metadata=metadata,
                    )
                )
            return results

        except Exception as e:
            elapsed_total = time.time() - start
            per_item = elapsed_total / max(len(metadata_list), 1)
            return [
                ExtractionResult(
                    title="",
                    body="",
                    raw_text="",
                    elapsed_s=per_item,
                    status="failed",
                    error=str(e),
                    metadata=metadata,
                )
                for metadata in metadata_list
            ]

# ----------------------------------------------------------------------
# Concrete extractors — only the HF model id differs between them
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