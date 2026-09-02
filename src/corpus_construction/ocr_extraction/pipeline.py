"""Run OCR backends over successful binarization outputs."""
from pathlib import Path
import logging, time
import pandas as pd
from .registry import OCR_EXTRACTORS
KEY_COLS = ["image_stem","config_id","layout_mode","detector","binarization","binarize_file","ocr_extractor"]
class OCRExtractionPipeline:
    def __init__(self, logger: logging.Logger, extractors, binarization_parquet, binarized_dir, parquet_path, skip_failed=True, opendataloader_kwargs=None, docling_kwargs=None):
        self.logger=logger; self.extractor_names=extractors; self.binarization_parquet=Path(binarization_parquet); self.binarized_dir=Path(binarized_dir); self.parquet_path=Path(parquet_path); self.skip_failed=skip_failed
        self.kwargs={"opendataloader":opendataloader_kwargs or {},"docling":docling_kwargs or {}}
    def _load_done(self):
        if not self.parquet_path.exists(): return set()
        try: df=pd.read_parquet(self.parquet_path)
        except Exception as exc: self.logger.warning("Could not read existing OCR Parquet: %s",exc); return set()
        if not set(KEY_COLS+["status"]).issubset(df.columns): return set()
        if self.skip_failed: df=df[df.status=="success"]
        return set(zip(*(df[c] for c in KEY_COLS)))
    def _discover_inputs(self):
        df=pd.read_parquet(self.binarization_parquet); required={"image_stem","config_id","detector","binarization","binarize_file","status"}; missing=required-set(df.columns)
        if missing: raise ValueError(f"Binarization Parquet is missing required columns: {sorted(missing)}")
        df=df[df.status=="success"]
        out=[]
        for _,r in df.iterrows():
            rel=str(r.binarize_file); path=self.binarized_dir/rel
            if not path.exists(): self.logger.warning("Binarized file missing: %s",path); continue
            det=None if pd.isna(r.detector) else str(r.detector)
            out.append(dict(image_path=path,image_stem=str(r.image_stem),config_id=int(r.config_id),layout_mode="none" if det is None else "layout",detector=det,binarization=str(r.binarization),binarize_file=rel))
        return out
    def _append(self,row):
        new=pd.DataFrame([row])
        if self.parquet_path.exists():
            try: old=pd.read_parquet(self.parquet_path); old=old if set(KEY_COLS).issubset(old.columns) else pd.DataFrame()
            except Exception: old=pd.DataFrame()
            df=pd.concat([old,new],ignore_index=True).drop_duplicates(KEY_COLS,keep="last")
        else: df=new
        df["_detector_sort"]=df.detector.fillna("").astype(str)
        df=df.sort_values(["image_stem","config_id","layout_mode","_detector_sort","binarization","ocr_extractor"]).drop(columns="_detector_sort")
        self.parquet_path.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(self.parquet_path,index=False)
    def run(self):
        if not self.binarization_parquet.exists(): raise FileNotFoundError(self.binarization_parquet)
        inputs=self._discover_inputs(); done=self._load_done(); processed=0
        for name in self.extractor_names:
            if name not in OCR_EXTRACTORS: raise ValueError(f"Unknown OCR extractor {name!r}; choose from {list(OCR_EXTRACTORS)}")
            extractor=OCR_EXTRACTORS[name](**self.kwargs[name])
            for item in inputs:
                key=tuple(item[c] for c in KEY_COLS[:-1])+(name,)
                if key in done: continue
                start=time.time(); row=dict(item,ocr_extractor=name)
                try: row.update(text=extractor.extract(item["image_path"],row),elapsed_s=time.time()-start,status="success",error=None)
                except Exception as exc: row.update(text="",elapsed_s=time.time()-start,status="failed",error=str(exc)); self.logger.exception("[%s] OCR failed for %s",name,item["image_path"])
                self._append(row); processed+=1
                if row["status"]=="success": done.add(key)
            extractor.close()
        return processed
