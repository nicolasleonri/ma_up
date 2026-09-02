from .docling import DoclingOCRExtractor
from .opendataloader import OpenDataLoaderOCRExtractor
OCR_EXTRACTORS = {"docling": DoclingOCRExtractor, "opendataloader": OpenDataLoaderOCRExtractor}
