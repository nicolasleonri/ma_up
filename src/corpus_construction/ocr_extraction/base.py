"""Common interface for OCR extractors."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
class OCRExtractor(ABC):
    name: str = ""
    @abstractmethod
    def extract(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Return OCR text for one image."""
    def close(self) -> None:
        """Release backend resources."""
