"""CLI entry point for OCR extraction."""
import argparse, logging
from src.corpus_construction.ocr_extraction.pipeline import OCRExtractionPipeline
AVAILABLE_OCR_EXTRACTORS=["opendataloader","docling"]
def parse_args():
    p=argparse.ArgumentParser(description="Run OCR extractors over binarized newspaper pages.")
    p.add_argument("--binarization-parquet",required=True); p.add_argument("--binarized-dir",required=True); p.add_argument("--output-parquet",default="data/corpus_construction/ocr_extraction/results.parquet")
    p.add_argument("--extractors",nargs="+",choices=AVAILABLE_OCR_EXTRACTORS,default=AVAILABLE_OCR_EXTRACTORS); p.add_argument("--no-skip-failed",action="store_true")
    p.add_argument("--ocr-lang",nargs="+",default=["spa"]); p.add_argument("--docling-backend",choices=["auto","tesseract"],default="auto")
    p.add_argument("--opendataloader-hybrid",default="docling-fast"); p.add_argument("--opendataloader-hybrid-mode",choices=["auto","full"],default="full"); p.add_argument("--opendataloader-hybrid-url",default=None); p.add_argument("--opendataloader-timeout-ms",type=int,default=0)
    return p.parse_args()
def main():
    a=parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    n=OCRExtractionPipeline(logging.getLogger(__name__),a.extractors,a.binarization_parquet,a.binarized_dir,a.output_parquet,not a.no_skip_failed,{"hybrid":a.opendataloader_hybrid,"hybrid_mode":a.opendataloader_hybrid_mode,"hybrid_url":a.opendataloader_hybrid_url,"timeout_ms":a.opendataloader_timeout_ms},{"languages":a.ocr_lang,"ocr_backend":a.docling_backend}).run()
    logging.info("OCR extraction finished: %d results",n)
if __name__=="__main__": main()
