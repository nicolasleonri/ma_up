"""Docling OCR adapter for TIFF/PNG/JPEG newspaper images."""
from pathlib import Path
from typing import Any
class DoclingOCRExtractor:
    name = "docling"
    def __init__(self, languages=None, ocr_backend="auto"):
        self.languages = languages or ["spa"]
        self.ocr_backend = ocr_backend
        self._converter = None
    def _ensure_converter(self):
        if self._converter is not None:
            return
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PipelineOptions, RapidOcrOptions, TesseractOcrOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption
        ocr_options = TesseractOcrOptions(lang=self.languages) if self.ocr_backend == "tesseract" else RapidOcrOptions(lang=self.languages)
        options = PipelineOptions(do_ocr=True, ocr_options=ocr_options)
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)},
        )
    def extract(self, image_path: Path, metadata: dict[str, Any]) -> str:
        self._ensure_converter()
        result = self._converter.convert(str(image_path))
        if result.document is None:
            return ""
        return result.document.export_to_markdown().strip()
    def close(self):
        self._converter = None
