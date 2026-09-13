import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    NemotronOcrOptions,
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


def _accelerator_options() -> AcceleratorOptions:
    """Use CUDA if available, otherwise use one CPU thread."""
    if torch.cuda.is_available():
        return AcceleratorOptions(
            num_threads=1,
            device=AcceleratorDevice.CUDA,
        )

    return AcceleratorOptions(
        num_threads=1,
        device=AcceleratorDevice.CPU,
    )


class BaseDoclingOCRExtractor:
    """
    Base class for Docling OCR extractors.

    One DocumentConverter/model is initialized per worker process
    and reused for every image assigned to that worker.
    """

    ocr_options = None

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            accelerator_options=_accelerator_options(),
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

    def extract_batch(
        self,
        image_paths: list[str],
    ) -> list[str]:
        """Convert multiple images in one Docling call."""
        results = list(
            self.converter.convert_all(image_paths)
        )

        return [
            result.document.export_to_markdown()
            for result in results
        ]

    def close(self) -> None:
        self.converter = None


class DoclingOCRExtractor(BaseDoclingOCRExtractor):
    ocr_options = None


class DoclingEasyOCRExtractor(BaseDoclingOCRExtractor):
    ocr_options = EasyOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )


class DoclingRapidOCRExtractor(BaseDoclingOCRExtractor):
    ocr_options = RapidOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )


class DoclingNemotronOCRExtractor(BaseDoclingOCRExtractor):
    ocr_options = NemotronOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
    )