from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    NemotronOcrOptions,
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


class BaseDoclingOCRExtractor:
    """Base class for Docling OCR extractors."""

    ocr_options = None

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
        )

        if self.ocr_options is not None:
            pipeline_options.ocr_options = self.ocr_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def extract(
        self,
        image_path: str,
        metadata: dict | None = None,
    ) -> str:
        result = self.converter.convert(image_path)
        return result.document.export_to_markdown()

    def close(self) -> None:
        """Release extractor resources."""
        self.converter = None


class DoclingOCRExtractor(BaseDoclingOCRExtractor):
    """Docling's default OCR configuration."""

    ocr_options = None


class DoclingEasyOCRExtractor(BaseDoclingOCRExtractor):
    """Docling using EasyOCR."""

    ocr_options = EasyOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )


class DoclingRapidOCRExtractor(BaseDoclingOCRExtractor):
    """Docling using RapidOCR."""

    ocr_options = RapidOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )


class DoclingNemotronOCRExtractor(BaseDoclingOCRExtractor):
    """Docling using NVIDIA Nemotron OCR."""

    ocr_options = NemotronOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )