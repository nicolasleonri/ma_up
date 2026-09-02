"""CLI entry point for the OCR-text -> article LLM stage."""
import argparse, logging
from src.corpus_construction.llm_extraction.pipeline import LLMExtractionPipeline
def parse_args():
    p=argparse.ArgumentParser(description="Run local text-only LLMs over OCR Parquet results.")
    p.add_argument("--ocr-parquet",required=True);p.add_argument("--output-parquet",default="data/corpus_construction/llm_extraction/results.parquet");p.add_argument("--models",nargs="+",required=True)
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--max-new-tokens",type=int,default=4096);p.add_argument("--gpu-memory-utilization",type=float,default=.85);p.add_argument("--tensor-parallel-size",type=int,default=1);p.add_argument("--dtype",default="bfloat16");p.add_argument("--max-model-len",type=int,default=None);p.add_argument("--no-skip-failed",action="store_true");return p.parse_args()
def main():
    a=parse_args();logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    n=LLMExtractionPipeline(logging.getLogger(__name__),a.models,a.batch_size,a.ocr_parquet,a.output_parquet,a.max_new_tokens,a.gpu_memory_utilization,a.tensor_parallel_size,a.dtype,a.max_model_len,not a.no_skip_failed).run();logging.info("LLM extraction finished: %d inputs",n)
if __name__=="__main__":main()
