"""Run local LLMs over successful OCR results."""
from pathlib import Path
import logging, time
import pandas as pd
from .steps import LocalLLM
KEY_COLS=["image_stem","config_id","layout_mode","detector","binarization","binarize_file","ocr_extractor","llm"]
class LLMExtractionPipeline:
    def __init__(self,logger:logging.Logger,models,batch_size,ocr_parquet,output_parquet,max_new_tokens=4096,gpu_memory_utilization=.85,tensor_parallel_size=1,dtype="bfloat16",max_model_len=None,skip_failed=True):
        self.logger=logger; self.models=models; self.batch_size=batch_size; self.ocr_parquet=Path(ocr_parquet); self.output=Path(output_parquet); self.llm_kwargs=dict(max_new_tokens=max_new_tokens,gpu_memory_utilization=gpu_memory_utilization,tensor_parallel_size=tensor_parallel_size,dtype=dtype,max_model_len=max_model_len); self.skip_failed=skip_failed
    def _done(self):
        if not self.output.exists():return set()
        try:df=pd.read_parquet(self.output)
        except Exception:return set()
        if not set(KEY_COLS+["status"]).issubset(df.columns):return set()
        if self.skip_failed:df=df[df.status=="success"]
        return set(zip(*(df[c] for c in KEY_COLS)))
    def _inputs(self):
        df=pd.read_parquet(self.ocr_parquet); req={"image_stem","config_id","layout_mode","detector","binarization","binarize_file","ocr_extractor","text","status"}; missing=req-set(df.columns)
        if missing:raise ValueError(f"OCR Parquet is missing required columns: {sorted(missing)}")
        return df[df.status=="success"].fillna({"detector":""}).to_dict("records")
    def _append(self,rows):
        if not rows:return
        new=pd.DataFrame(rows)
        if self.output.exists():
            try:old=pd.read_parquet(self.output); old=old if set(KEY_COLS).issubset(old.columns) else pd.DataFrame()
            except Exception:old=pd.DataFrame()
            df=pd.concat([old,new],ignore_index=True).drop_duplicates(KEY_COLS+["article_index"],keep="last")
        else:df=new
        df["_detector_sort"]=df.detector.fillna("").astype(str)
        df=df.sort_values(["image_stem","config_id","layout_mode","_detector_sort","binarization","ocr_extractor","llm","article_index"]).drop(columns="_detector_sort")
        self.output.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(self.output,index=False)
    def run(self):
        if not self.ocr_parquet.exists():raise FileNotFoundError(self.ocr_parquet)
        inputs=self._inputs(); done=self._done(); processed=0
        for model in self.models:
            pending=[]
            for item in inputs:
                key=tuple(item[c] for c in KEY_COLS[:-1])+(model,)
                if key not in done:pending.append(item)
            if not pending:continue
            llm=LocalLLM(model,**self.llm_kwargs)
            for start in range(0,len(pending),self.batch_size):
                chunk=pending[start:start+self.batch_size]; texts=[str(x["text"]) for x in chunk]; t=time.time()
                try: article_lists, raw_texts=llm.extract_batch(texts); elapsed=(time.time()-t)/max(len(chunk),1)
                except Exception as exc:
                    self.logger.exception("[%s] LLM batch failed",model)
                    article_lists=[None]*len(chunk); raw_texts=[""]*len(chunk); elapsed=(time.time()-t)/max(len(chunk),1)
                for item,articles,raw_text in zip(chunk,article_lists,raw_texts):
                    base={k:item[k] for k in ["image_stem","config_id","layout_mode","detector","binarization","binarize_file","ocr_extractor"]};base["llm"]=model
                    if articles is None:
                        self._append([dict(base,article_index=-1,title="",subheadline="",author="",body="",ocr_text=item["text"],raw_transcription=raw_text,elapsed_s=elapsed,status="failed",error=str(exc))]);continue
                    rows=[dict(base,article_index=a.article_index,title=a.title,subheadline=a.subheadline,author=a.author,body=a.body,ocr_text=item["text"],raw_transcription=raw_text,elapsed_s=elapsed,status="success",error=None) for a in articles]
                    if not rows:rows=[dict(base,article_index=-1,title="",subheadline="",author="",body="",ocr_text=item["text"],raw_transcription=raw_text,elapsed_s=elapsed,status="success",error=None)]
                    self._append(rows);done.add(tuple(item[c] for c in KEY_COLS[:-1])+(model,));processed+=1
            llm.unload();time.sleep(2)
        return processed
