"""OCR extractor registry."""

from .docling import (
    DoclingEasyOCRExtractor,
    DoclingOCRExtractor,
    DoclingRapidOCRExtractor,
    DoclingNemotronOCRExtractor,
)

OCR_EXTRACTORS = {
    "docling": DoclingOCRExtractor,
    "docling_easyocr": DoclingEasyOCRExtractor,
    "docling_rapidocr": DoclingRapidOCRExtractor,
    "docling_nemotron-ocr": DoclingNemotronOCRExtractor,
}