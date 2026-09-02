"""OpenDataLoader OCR adapter.

OpenDataLoader's current Python API is PDF-oriented. This adapter wraps each
source image in a temporary one-page PDF and sends it through the configured
local OpenDataLoader backend. For scanned pages, the hybrid backend must have
OCR enabled (for example, ``opendataloader-pdf-hybrid --force-ocr``).
"""
from pathlib import Path
from typing import Any
import tempfile
import json
from PIL import Image
class OpenDataLoaderOCRExtractor:
    name = "opendataloader"
    def __init__(self, hybrid="docling-fast", hybrid_mode="full", hybrid_url=None, timeout_ms=0):
        self.hybrid = hybrid
        self.hybrid_mode = hybrid_mode
        self.hybrid_url = hybrid_url
        self.timeout_ms = timeout_ms
    @staticmethod
    def _image_to_pdf(image_path: Path, destination: Path):
        with Image.open(image_path) as image:
            image.convert("RGB").save(destination, format="PDF", resolution=72.0)
    @staticmethod
    def _json_text(path: Path) -> str:
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        def walk(node):
            if isinstance(node, dict):
                if isinstance(node.get("content"), str):
                    parts.append(node["content"])
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(data)
        return "\n".join(p for p in parts if p).strip()
    def extract(self, image_path: Path, metadata: dict[str, Any]) -> str:
        import opendataloader_pdf
        with tempfile.TemporaryDirectory(prefix="ma_up_odl_") as tmp:
            tmpdir = Path(tmp)
            pdf = tmpdir / "page.pdf"
            out = tmpdir / "output"
            out.mkdir()
            self._image_to_pdf(image_path, pdf)
            kwargs = dict(input_path=[str(pdf)], output_dir=str(out), format="text", quiet=True, hybrid=self.hybrid, hybrid_mode=self.hybrid_mode)
            if self.hybrid_url:
                kwargs["hybrid_url"] = self.hybrid_url
            if self.timeout_ms:
                kwargs["hybrid_timeout"] = str(self.timeout_ms)
            opendataloader_pdf.convert(**kwargs)
            txt = list(out.rglob("*.txt"))
            if txt:
                return txt[0].read_text(encoding="utf-8", errors="replace").strip()
            js = list(out.rglob("*.json"))
            return self._json_text(js[0]) if js else ""
