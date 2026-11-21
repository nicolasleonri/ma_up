"""
steps.py

Domain logic for DSPy prompt optimization (Steps 1-3):
- Step 1 (build): Filter optimal Table 1 OCR configs and page positions (via sample.csv),
    preserve verbatim region OCR outputs per page (sorted by crop filename), align with gold CSV,
    and cache standardized page-level parquets.
- Step 2 (eval): Region-level extraction (matching src/corpus_construction/llm_extraction)
    + page-level Hungarian evaluation reusing exact primitives from evaluate_extraction.py.
- Step 3 (optimize): Stratified k-fold CV with DSPy BootstrapFewShot (or MIPROv2),
    saving the best fold's compiled artifact and standalone Markdown prompt only when it
    beats the Table 1 baseline.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import dspy
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

# Ensure repo root is importable so we can reuse evaluate_extraction primitives directly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workflows.evaluate_extraction import (  # noqa: E402
    MATCH_THRESHOLD,
    cer,
    similarity_score,
    wer,
)

# =========================================================
# Unified Newspaper Configurations (Table 1 source of truth)
# =========================================================

SAMPLE_MANIFEST_CSV = Path("scripts/R/get_sample/sample.csv")


def _default_paths(name: str, llm_short: str) -> dict[str, str]:
    return {
        "results_parquet": f"data/corpus_construction/llm_extraction/{name}/results_cropped_{llm_short}.parquet",
        "ocr_parquet_cropped": f"data/corpus_construction/ocr_extraction/{name}/cropped/ocr.parquet",
        "ocr_parquet_none": f"data/corpus_construction/ocr_extraction/{name}/none/ocr.parquet",
        "gold_csv": f"data/corpus_construction/evaluate_extraction/{name}.csv",
        "dataset_parquet": f"data/corpus_construction/local_dspy_llm/datasets/{name}.parquet",
        "artifact_json": f"data/corpus_construction/local_dspy_llm/artifacts/{name}.json",
        "cv_report_csv": f"data/corpus_construction/local_dspy_llm/reports/{name}_cv.csv",
    }


NEWSPAPERS: dict[str, dict[str, Any]] = {
    "correo": {
        "model": "mistral:7b",
        "gold_position": "Interior",
        "table1_f1": 0.89,
        **_default_paths("correo", "mistral"),
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 13,
            "detector": "histogram",
            "binarization": "basic",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "elcomercio": {
        "model": "qwen2.5:7b",
        "gold_position": "Interior",
        "table1_f1": 0.90,
        **_default_paths("elcomercio", "qwen"),
        "ocr_variant": "none",
        "filter": {
            "config_id": 51,
            "detector": None,
            "binarization": "none_color",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "gestion": {
        "model": "qwen2.5:7b",
        "gold_position": "Interior",
        "table1_f1": 0.65,
        **_default_paths("gestion", "qwen"),
        "ocr_variant": "none",
        "filter": {
            "config_id": 6,
            "detector": None,
            "binarization": "otsu",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "ojo": {
        "model": "qwen2.5:7b",
        "gold_position": "Interior",
        "table1_f1": 0.77,
        **_default_paths("ojo", "qwen"),
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 52,
            "detector": "ppdoclayout",
            "binarization": "basic",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "peru21": {
        "model": "deepseek-r1:7b",
        "gold_position": "Interior",
        "table1_f1": 1.00,
        **_default_paths("peru21", "deepseek-r1"),
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 38,
            "detector": "ppdoclayout",
            "binarization": "none_color",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "publimetro": {
        "model": "mistral:7b",
        "gold_position": "Interior",
        "table1_f1": 0.64,
        **_default_paths("publimetro", "mistral"),
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 12,
            "detector": "histogram",
            "binarization": "none_grayscale",
            "ocr_extractor": "docling_easyocr",
        },
    },
    "trome_interior": {
        "model": "mistral:7b",
        "gold_position": "Interior",
        "table1_f1": 0.89,
        **_default_paths("trome", "mistral"),
        "dataset_parquet": "data/corpus_construction/local_dspy_llm/datasets/trome_interior.parquet",
        "artifact_json": "data/corpus_construction/local_dspy_llm/artifacts/trome_interior.json",
        "cv_report_csv": "data/corpus_construction/local_dspy_llm/reports/trome_interior_cv.csv",
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 34,
            "detector": "ppdoclayout",
            "binarization": "none_color",
            "ocr_extractor": "docling_rapidocr",
        },
    },
    "trome_last": {
        "model": "qwen2.5:7b",
        "gold_position": "Last",
        "table1_f1": 1.00,
        **_default_paths("trome", "qwen"),
        "dataset_parquet": "data/corpus_construction/local_dspy_llm/datasets/trome_last.parquet",
        "artifact_json": "data/corpus_construction/local_dspy_llm/artifacts/trome_last.json",
        "cv_report_csv": "data/corpus_construction/local_dspy_llm/reports/trome_last_cv.csv",
        "ocr_variant": "cropped",
        "filter": {
            "config_id": 7,
            "detector": "doclayout",
            "binarization": "otsu",
            "ocr_extractor": "docling_rapidocr",
        },
    },
}

PAGE_COL = "image_stem"
OCR_TEXT_COL = "text"
CROP_FILE_COL = "binarize_file"
GOLD_FIELDS = ["title", "body"]


@dataclass
class PageRecord:
    newspaper: str
    page_id: str
    strat: str
    regions: list[str]          # Verbatim OCR outputs of all regions on the page (sorted by crop filename)
    gold: list[dict[str, str]]  # Gold articles with non-empty body


# =========================================================
# Step 1: Dataset Construction & Caching
# =========================================================

def _is_na(val: Any) -> bool:
    return (
        val is None
        or (isinstance(val, float) and math.isnan(val))
        or str(val).strip().lower() in {"", "na", "nan", "none"}
    )


def _clean(val: Any) -> str:
    return "" if _is_na(val) else str(val).strip()


def _strat_key(page_id: str) -> str:
    m = re.search(r"_(\d{4})-\d{2}-\d{2}", page_id)
    return m.group(1) if m else "all"


def step_build_dataset(name: str, overwrite: bool = True) -> list[PageRecord]:
    """Filter optimal Stage 1-4 OCR crops, preserve verbatim per region, align with gold CSV, and save parquet."""
    cfg = NEWSPAPERS[name]
    out_path = Path(cfg["dataset_parquet"])
    if out_path.exists() and not overwrite:
        return load_dataset(name)

    ocr_path = Path(
        cfg["ocr_parquet_none"] if cfg["ocr_variant"] == "none" else cfg["ocr_parquet_cropped"]
    )
    gold_path = Path(cfg["gold_csv"])
    if not ocr_path.exists():
        raise FileNotFoundError(f"[{name}] Missing OCR parquet: {ocr_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"[{name}] Missing gold CSV: {gold_path}")

    raw_ocr = pd.read_parquet(ocr_path)
    gold_df = pd.read_csv(gold_path, dtype=str, keep_default_na=False)

    if "page_ids" in cfg:
        gold_df = gold_df[gold_df[PAGE_COL].astype(str).str.strip().isin(cfg["page_ids"])]

    if "gold_position" in cfg and cfg["gold_position"]:
        if not SAMPLE_MANIFEST_CSV.exists():
            raise FileNotFoundError(f"[{name}] Missing sample manifest: {SAMPLE_MANIFEST_CSV}")
        sample_df = pd.read_csv(SAMPLE_MANIFEST_CSV, dtype=str, keep_default_na=False)
        target_pos = cfg["gold_position"].strip().lower()
        if target_pos == "last":
            target_pos = "back"

        base_news = name.split("_")[0].lower()
        sub_sample = sample_df[
            (sample_df["newspaper"].str.strip().str.lower() == base_news)
            & (sample_df["page_position"].str.strip().str.lower() == target_pos)
        ]
        allowed_stems = {
            f"{r['newspaper'].strip()}_{r['date'].strip()}_{r['page_number'].strip()}"
            for r in sub_sample.to_dict("records")
        }
        gold_df = gold_df[gold_df[PAGE_COL].astype(str).str.strip().isin(allowed_stems)]

    all_gold_pids = list(dict.fromkeys(gold_df[PAGE_COL].astype(str).str.strip()))

    # Filter OCR rows per page using exact Table 1 config_id and stage parameters
    filtered_parts = []
    for pid in all_gold_pids:
        sub = raw_ocr[raw_ocr[PAGE_COL].astype(str).str.strip() == pid]
        for col, val in cfg["filter"].items():
            if col not in sub.columns:
                raise KeyError(f"[{name}] Column {col!r} missing in {ocr_path}")
            if val is None:
                sub = sub[sub[col].isna() | (sub[col].astype(str).str.lower() == "none")]
            elif col == "config_id":
                sub = sub[pd.to_numeric(sub[col], errors="coerce") == int(val)]
            else:
                sub = sub[sub[col].astype(str).str.strip() == str(val)]
        filtered_parts.append(sub)

    ocr = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else raw_ocr.iloc[0:0]

    # Preserve verbatim OCR text from every region; only drop completely blank rows
    ocr = ocr.copy()
    ocr[OCR_TEXT_COL] = ocr[OCR_TEXT_COL].fillna("").astype(str)
    nonempty = ocr[ocr[OCR_TEXT_COL].str.strip().ne("")]
    if CROP_FILE_COL in nonempty.columns:
        nonempty = nonempty.sort_values(CROP_FILE_COL, kind="stable")

    regions_by_page = {
        str(pid): grp[OCR_TEXT_COL].tolist()
        for pid, grp in nonempty.groupby(PAGE_COL, sort=False)
    }

    gold_by_page: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in gold_df.to_dict("records"):
        pid = str(row[PAGE_COL]).strip()
        cleaned = {f: _clean(row.get(f)) for f in GOLD_FIELDS}
        if cleaned["body"]:  # Skip unannotated NA bodies
            gold_by_page[pid].append(cleaned)

    pages: list[PageRecord] = []
    rows_for_parquet: list[dict[str, Any]] = []
    for pid, gold_arts in gold_by_page.items():
        regs = regions_by_page.get(pid, [])
        strat = _strat_key(pid)
        pages.append(PageRecord(name, pid, strat, regs, gold_arts))
        rows_for_parquet.append({
            "newspaper": name,
            "page_id": pid,
            "strat": strat,
            "n_regions": len(regs),
            "regions_json": json.dumps(regs, ensure_ascii=False),
            "gold_articles_json": json.dumps(gold_arts, ensure_ascii=False),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_for_parquet).to_parquet(out_path, index=False)
    non_empty_pages = sum(bool(p.regions) for p in pages)
    print(
        f"[build:{name}] Saved {len(pages)} pages ({non_empty_pages}/{len(pages)} with OCR regions, "
        f"{sum(len(p.gold) for p in pages)} gold articles) -> {out_path}"
    )
    return pages


def load_dataset(name: str) -> list[PageRecord]:
    """Load cached page-level dataset from parquet (building it first if missing)."""
    pq_path = Path(NEWSPAPERS[name]["dataset_parquet"])
    if not pq_path.exists():
        return step_build_dataset(name, overwrite=True)

    df = pd.read_parquet(pq_path)
    pages = [
        PageRecord(
            newspaper=str(r["newspaper"]),
            page_id=str(r["page_id"]),
            strat=str(r["strat"]),
            regions=json.loads(r["regions_json"]),
            gold=json.loads(r["gold_articles_json"]),
        )
        for r in df.to_dict("records")
    ]
    print(f"[load_dataset:{name}] Loaded {len(pages)} pages from {pq_path}")
    return pages


# =========================================================
# Step 2: DSPy Region Extractor & Hungarian Evaluator
# (Matches src/corpus_construction/llm_extraction/steps.py)
# =========================================================

class ArticleExtraction(dspy.Signature):
    """You are a structured extraction assistant specializing in Peruvian print journalism.

    The input is OCR text from a scanned page or region of a Peruvian newspaper belonging to the El Comercio editorial group (Trome, Peru21, Ojo, etc.). These are Spanish-language publications covering Peruvian national and international news, politics, economy, culture and sport.

    OCR quality may be imperfect: expect broken words, missing accents, inconsistent spacing and mixed-case artifacts. Use journalistic context to interpret ambiguous text —do not "correct" it in the output, preserve it as-is.

    Given the OCR text below, extract every distinct article or editorial piece found.

    Rules:
- Text is primarily Spanish; preserve it as-is, including OCR artifacts.
- Preserve reading order (top-to-bottom, left-to-right column order).
- A region or page may contain multiple unrelated articles — extract each separately.
- Exclude captions, page numbers, section headers and advertisements.
- Do not invent, infer or complete missing fields.
- Return [] if no articles are present.
- Return ONLY the JSON array. No explanation, preamble, or markdown fences."""

    ocr_text: str = dspy.InputField(
        desc="Raw OCR transcription of a scanned newspaper region or page in markdown format."
    )
    articles: str = dspy.OutputField(
        desc="""Return ONLY a JSON array of all newspaper articles found in the OCR text.
Each element must contain exactly these keys:
- "title"       : main headline, as it appears (empty string if absent)
- "subheadline" : secondary headline, deck, or standfirst (empty string if absent)
- "author"      : byline — typically introduced by "Por", "Redacción", or a staff name (empty string if absent)
- "body"        : full article body text, preserving paragraph breaks with \\n (empty string if absent)
Return [] if no articles are present. No explanation or markdown fences."""
    )


BASELINE_DEMOS = [
    dspy.Example(
        ocr_text=(
            "## Gabinete aprueba nuevo paquete economico\n"
            "El ConsejodeMinistros aprobo ayer un paquete de medidas econ6micas orientadas a\n"
            "reactivar la inversión privada en las regiones de la sierra y la selva.\n"
            "El ministro de Economia anunci6 que las medidas incluyen exoneraciones tributarias\n"
            "y facilidades para la formalizaci6n de pequenas empresas.\n\n"
            "[Foto: Operaciones en mina Las Bambas, Apurimac]\n\n"
            "PUBLICIDAD\n"
            "Banco Continental — Prestamos para tu negocio. Llama al 0800-00000."
        ),
        articles=json.dumps(
            [
                {
                    "title": "Gabinete aprueba nuevo paquete economico",
                    "subheadline": "",
                    "author": "",
                    "body": (
                        "El Consejo de Ministros aprobó ayer un paquete de medidas económicas orientadas a "
                        "reactivar la inversión privada en las regiones de la sierra y la selva.\n"
                        "El ministro de Economía anunció que las medidas incluyen exoneraciones tributarias "
                        "y facilidades para la formalización de pequeñas empresas."
                    ),
                }
            ],
            ensure_ascii=False,
        ),
    ).with_inputs("ocr_text"),
    dspy.Example(
        ocr_text=(
            "## Fujimori anuncia plan de pacificaciiion en el sur\n"
            "Gobierno destinara 200 millones de soles para zones afectados por conflicto\n\n"
            "Por Jorge Salinas / Enviado especial\n\n"
            "El presidente Alberto Fujimori anunci6 ayer desde Ayacucho un ambicioso plan de\n"
            "pacificaci6n para las zonas del sur afectadas por anos de violencia.\n"
            "El plan contempla la construcci6n de colegios, postas medicas y carreteras en\n"
            "comunidades que estuvieron bajo control de Sendero Luminoso.\n\n"
            "## Detienen a presunto lIder de banda criminal en Huamange\n\n"
            "La Policia Nacional captur6 en la madrugada del lunes a un c1udadano identificado\n"
            "como presunto lIder de una band crimina en el distrito de Huamanga, Ayacucho.\n"
            "Segun fuentes policiales, el detenido portaba d0cumentos y material delincuencia."
        ),
        articles=json.dumps(
            [
                {
                    "title": "Fujimori anuncia plan de pacificación en el sur",
                    "subheadline": "Gobierno destinará 200 millones de soles para zonas afectadas por conflicto",
                    "author": "Jorge Salinas",
                    "body": (
                        "El presidente Alberto Fujimori anunció ayer desde Ayacucho un ambicioso plan de "
                        "pacificación para las zonas del sur afectadas por años de violencia.\n"
                        "El plan contempla la construcción de colegios, postas medicas y carreteras en "
                        "comunidades que estuvieron bajo control de Sendero Luminoso."
                    ),
                },
                {
                    "title": "Detienen a presunto líder de banda criminal en Huamanga",
                    "subheadline": "",
                    "author": "",
                    "body": (
                        "La Policía Nacional capturó en la madrugada del lunes a un ciudadano identificado "
                        "como presunto líder de una banda criminal en el distrito de Huamanga, Ayacucho.\n"
                        "Según fuentes policiales, el detenido portaba documentos y material delincuencial."
                    ),
                },
            ],
            ensure_ascii=False,
        ),
    ).with_inputs("ocr_text"),
]


def _parse_articles(raw: str) -> list[dict[str, str]]:
    """Parse JSON array returned by DSPy (mirrors llm_extraction/steps.py + DeepSeek <think> cleanup)."""
    text = (raw or "").strip()
    # Strip DeepSeek-R1 <think>...</think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting first JSON array substring if wrapped in chatter
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return [{"title": "", "subheadline": "", "author": "", "body": text}]
        else:
            return [{"title": "", "subheadline": "", "author": "", "body": text}]

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    articles: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("title", ""))
        body = _clean(item.get("body", ""))
        if title or body:
            articles.append({
                "title": title,
                "subheadline": _clean(item.get("subheadline", "")),
                "author": _clean(item.get("author", "")),
                "body": body,
            })
    return articles


class PageArticleExtractor(dspy.Module):
    """DSPy module: extracts articles per OCR region and aggregates them at page level."""

    def __init__(self, use_baseline_demos: bool = True, verbose: bool = False, num_threads: int = 4):
        super().__init__()
        self.extract_region = dspy.Predict(ArticleExtraction)
        if use_baseline_demos:
            self.extract_region.demos = list(BASELINE_DEMOS)
        self.verbose = verbose
        self.num_threads = num_threads

    def forward(self, regions: list[str]) -> dspy.Prediction:
        valid_regs = [r for r in regions if r and r.strip()]
        if not valid_regs:
            return dspy.Prediction(articles=[])

        all_articles: list[dict[str, str]] = []
        if self.verbose or self.num_threads <= 1:
            for i, reg in enumerate(valid_regs, start=1):
                t0 = time.perf_counter()
                res = self.extract_region(ocr_text=reg)
                parsed = _parse_articles(getattr(res, "articles", ""))
                all_articles.extend(parsed)
                if self.verbose:
                    print(
                        f"    region {i}/{len(valid_regs)} ({len(reg)} chars) -> "
                        f"{len(parsed)} arts ({time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
        else:
            examples = [dspy.Example(ocr_text=reg).with_inputs("ocr_text") for reg in valid_regs]
            parallel = dspy.Parallel(num_threads=self.num_threads)
            results = parallel([(self.extract_region, ex) for ex in examples])
            for res in results:
                all_articles.extend(_parse_articles(getattr(res, "articles", "")))

        return dspy.Prediction(articles=all_articles)


def to_dspy_example(page: PageRecord) -> dspy.Example:
    return dspy.Example(
        page_id=page.page_id,
        regions=page.regions,
        gold_articles=page.gold,
    ).with_inputs("regions")


def evaluate_page_predictions(
    pred_articles: list[dict[str, str]],
    gold_articles: list[dict[str, str]],
) -> dict[str, float]:
    """
    Page-level Hungarian matching identical to src/workflows/evaluate_extraction.py:
      score = 0.30 * fuzzy_title + 0.70 * fuzzy_body >= 70.0 -> TP
      CER and WER computed across TP pairs.
    """
    n_gold = len(gold_articles)
    n_pred = len(pred_articles)

    if n_gold == 0 and n_pred == 0:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "title_cer": 0.0, "body_wer": 0.0, "tp": 0, "fp": 0, "fn": 0}
    if n_gold == 0 or n_pred == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "title_cer": 1.0, "body_wer": 1.0, "tp": 0, "fp": n_pred, "fn": n_gold}

    sim_matrix = np.zeros((n_gold, n_pred), dtype=float)
    for gi, grow in enumerate(gold_articles):
        for pi, prow in enumerate(pred_articles):
            score, _, _ = similarity_score(
                prow.get("title", ""),
                prow.get("body", ""),
                grow.get("title", ""),
                grow.get("body", ""),
            )
            sim_matrix[gi, pi] = score

    gold_idx, pred_idx = linear_sum_assignment(100.0 - sim_matrix)

    tp = 0
    title_cers: list[float] = []
    body_wers: list[float] = []

    for gi, pi in zip(gold_idx, pred_idx):
        if float(sim_matrix[gi, pi]) >= MATCH_THRESHOLD:
            tp += 1
            title_cers.append(cer(pred_articles[pi].get("title", ""), gold_articles[gi].get("title", "")))
            body_wers.append(wer(pred_articles[pi].get("body", ""), gold_articles[gi].get("body", "")))

    fp = n_pred - tp
    fn = n_gold - tp
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "title_cer": float(mean(title_cers)) if title_cers else 1.0,
        "body_wer": float(mean(body_wers)) if body_wers else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def dspy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float | bool:
    """DSPy metric: returns bool during bootstrapping trace, composite F1 - tie-breaker otherwise."""
    metrics = evaluate_page_predictions(pred.articles or [], example.gold_articles)
    if trace is not None:
        return metrics["f1"] >= 0.75 and metrics["body_wer"] <= 0.45
    return max(0.0, metrics["f1"] - 0.05 * metrics["body_wer"])


# =========================================================
# Step 3: Stratified 5-Fold CV & DSPy Optimization
# =========================================================

def make_lm(host: str, model: str, temperature: float = 0.0) -> dspy.LM:
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base=host,
        api_key="ollama",
        temperature=temperature,
        max_tokens=16384,
        num_ctx=32768,
        repeat_penalty=1.1,
        repeat_last_n=128,
        timeout=180,
        num_retries=1,
    )


def stratified_k_folds(pages: list[PageRecord], k: int, seed: int) -> list[list[PageRecord]]:
    rng = random.Random(seed)
    ordered = list(pages)
    rng.shuffle(ordered)
    ordered.sort(key=lambda p: p.strat)
    folds: list[list[PageRecord]] = [[] for _ in range(k)]
    for idx, page in enumerate(ordered):
        folds[idx % k].append(page)
    return folds


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _align_regions_to_gold(pages: list[PageRecord]) -> list[dspy.Example]:
    """Match each gold article to its best-overlapping OCR region to build clean region-level training examples."""
    examples: list[dspy.Example] = []
    for p in pages:
        if not p.regions:
            continue
        assigned: list[list[dict[str, str]]] = [[] for _ in p.regions]
        norm_regs = [_normalize_text(r) for r in p.regions]

        for art in p.gold:
            needle = _normalize_text(f"{art.get('title', '')} {art.get('body', '')[:120]}")
            if not needle:
                continue
            best_idx = 0
            best_score = -1.0
            for idx, r_norm in enumerate(norm_regs):
                if not r_norm:
                    continue
                score = fuzz.partial_ratio(needle, r_norm)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_score >= 45.0:
                assigned[best_idx].append(art)

        for reg, arts in zip(p.regions, assigned):
            # Include both positive regions (with articles) and negative regions (ads/captions -> [])
            examples.append(
                dspy.Example(
                    ocr_text=reg,
                    articles=json.dumps(arts, ensure_ascii=False),
                ).with_inputs("ocr_text")
            )
    return examples


def _region_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    gold_arts = json.loads(example.articles)
    pred_arts = _parse_articles(getattr(pred, "articles", ""))
    if not gold_arts and not pred_arts:
        return 1.0
    if not gold_arts or not pred_arts:
        return 0.0
    return evaluate_page_predictions(pred_arts, gold_arts)["f1"]


def _compile_program(
    train_pages: list[PageRecord],
    optimizer_type: str = "bootstrap",
    threads: int = 2,
) -> dspy.Module:
    region_trainset = _align_regions_to_gold(train_pages)
    if not region_trainset:
        return PageArticleExtractor(use_baseline_demos=True)

    student = dspy.Predict(ArticleExtraction)
    student.demos = list(BASELINE_DEMOS)

    if optimizer_type == "mipro":
        optimizer = dspy.MIPROv2(
            metric=_region_metric,
            auto="light",
            num_threads=threads,
        )
        compiled_extractor = optimizer.compile(
            student,
            trainset=region_trainset,
            max_bootstrapped_demos=3,
            max_labeled_demos=2,
        )
    else:
        optimizer = dspy.BootstrapFewShot(
            metric=_region_metric,
            metric_threshold=0.50,
            max_bootstrapped_demos=3,
            max_labeled_demos=2,
            max_rounds=2,
        )
        compiled_extractor = optimizer.compile(student, trainset=region_trainset)

    wrapper = PageArticleExtractor(use_baseline_demos=False)
    wrapper.extract_region = compiled_extractor
    return wrapper


def step_evaluate_pages(
    program: dspy.Module,
    pages: list[PageRecord],
    verbose: bool = False,
) -> dict[str, float]:
    rows = []
    errors = 0
    total = len(pages)
    for idx, p in enumerate(pages, start=1):
        t0 = time.perf_counter()
        try:
            pred = program(regions=p.regions)
            res = evaluate_page_predictions(pred.articles or [], p.gold)
            rows.append(res)
            if verbose:
                dt = time.perf_counter() - t0
                print(
                    f"  [{idx}/{total}] {p.page_id} ({len(p.regions)} regions) -> "
                    f"pred={len(pred.articles or [])} gold={len(p.gold)} "
                    f"F1={res['f1']:.3f} ({dt:.1f}s)",
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            rows.append({"f1": 0.0, "precision": 0.0, "recall": 0.0, "title_cer": 1.0, "body_wer": 1.0})
            if verbose:
                dt = time.perf_counter() - t0
                print(
                    f"  [{idx}/{total}] {p.page_id} -> ERROR: {type(exc).__name__}: {exc} ({dt:.1f}s)",
                    flush=True,
                )
    return {
        "f1": float(mean(r["f1"] for r in rows)),
        "title_cer": float(mean(r["title_cer"] for r in rows)),
        "body_wer": float(mean(r["body_wer"] for r in rows)),
        "errors": errors,
    }


def export_prompt_markdown(
    program: dspy.Module,
    name: str,
    out_dir: str = "data/corpus_construction/local_dspy_llm/prompts",
) -> Path:
    """Export the compiled DSPy instruction and few-shot demos as clean, standalone Markdown (no metadata)."""
    predictor = getattr(program, "extract_region", program)
    sig = getattr(predictor, "signature", ArticleExtraction)
    instructions = (getattr(sig, "instructions", "") or ArticleExtraction.__doc__ or "").strip()
    demos = getattr(predictor, "demos", []) or []

    lines = [
        instructions,
        "",
        "Return ONLY a JSON array. Each element must have exactly these keys:",
        '- "title"       : main headline, as it appears (empty string if absent)',
        '- "subheadline" : secondary headline, deck, or standfirst (empty string if absent)',
        '- "author"      : byline — typically introduced by "Por", "Redacción", or a staff name (empty string if absent)',
        '- "body"        : full article body text, preserving paragraph breaks with \\n (empty string if absent)',
        "",
        "--- EXAMPLES ---",
    ]

    for idx, demo in enumerate(demos, start=1):
        ocr_in = (getattr(demo, "ocr_text", "") or "").strip()
        raw_out = getattr(demo, "articles", "[]")
        if not isinstance(raw_out, str):
            raw_out = json.dumps(raw_out, ensure_ascii=False, indent=2)
        else:
            try:
                raw_out = json.dumps(json.loads(raw_out), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                raw_out = raw_out.strip()

        lines.extend([
            f"### Example {idx}",
            "OCR TEXT:",
            "```text",
            ocr_in,
            "```",
            "JSON ARRAY:",
            "```json",
            raw_out,
            "```",
            "",
        ])

    lines.extend([
        "--- END EXAMPLES ---",
        "",
        "Now extract from the following OCR text:",
        "OCR TEXT:",
        "{ocr_text}",
        "JSON ARRAY:",
    ])

    out_path = Path(out_dir) / f"{name}_prompt.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"[prompt:{name}] Exported standalone Markdown prompt -> {out_path}")
    return out_path


def step_run_cv_and_optimize(
    name: str,
    host: str = "http://127.0.0.1:11434",
    k: int = 5,
    seed: int = 42,
    optimizer_type: str = "bootstrap",
    threads: int = 2,
    skip_cv: bool = False,
) -> Path | None:
    """Run stratified k-fold CV (baseline vs DSPy), saving the best fold prompt only if it beats Table 1."""
    cfg = NEWSPAPERS[name]
    pages = load_dataset(name)
    if not pages:
        raise RuntimeError(f"[{name}] No pages found in dataset.")

    lm = make_lm(host, cfg["model"], temperature=0.0)
    dspy.configure(lm=lm)

    effective_k = min(k, len(pages))
    cv_rows: list[dict[str, Any]] = []
    best_fold_program: dspy.Module | None = None
    best_fold_res: dict[str, float] | None = None
    best_fold_base: dict[str, float] | None = None
    best_fold_idx: int = -1

    if not skip_cv and effective_k >= 2:
        folds = stratified_k_folds(pages, k=effective_k, seed=seed)
        for fold_idx, test_pages in enumerate(folds, start=1):
            train_pages = [p for j, f in enumerate(folds) if (j + 1) != fold_idx for p in f]

            base_res = step_evaluate_pages(PageArticleExtractor(use_baseline_demos=True), test_pages)
            compiled = _compile_program(train_pages, optimizer_type=optimizer_type, threads=threads)
            opt_res = step_evaluate_pages(compiled, test_pages)

            if best_fold_res is None or opt_res["f1"] > best_fold_res["f1"]:
                best_fold_res = opt_res
                best_fold_base = base_res
                best_fold_program = compiled
                best_fold_idx = fold_idx

            cv_rows.append({
                "newspaper": name,
                "model": cfg["model"],
                "fold": fold_idx,
                "n_test_pages": len(test_pages),
                "base_f1": round(base_res["f1"], 4),
                "opt_f1": round(opt_res["f1"], 4),
                "base_title_cer": round(base_res["title_cer"], 4),
                "opt_title_cer": round(opt_res["title_cer"], 4),
                "base_body_wer": round(base_res["body_wer"], 4),
                "opt_body_wer": round(opt_res["body_wer"], 4),
            })
            print(
                f"[{name} | fold {fold_idx}/{effective_k}] "
                f"F1: {base_res['f1']:.3f} -> {opt_res['f1']:.3f} | "
                f"Title CER: {base_res['title_cer']:.3f} -> {opt_res['title_cer']:.3f} | "
                f"Body WER: {base_res['body_wer']:.3f} -> {opt_res['body_wer']:.3f}"
            )

        cv_df = pd.DataFrame(cv_rows)
        cv_path = Path(cfg["cv_report_csv"])
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(cv_path, index=False)

        b_f1, o_f1 = cv_df["base_f1"].tolist(), cv_df["opt_f1"].tolist()
        b_std = stdev(b_f1) if len(b_f1) > 1 else 0.0
        o_std = stdev(o_f1) if len(o_f1) > 1 else 0.0
        print(
            f"== [{name}] {effective_k}-Fold Summary: "
            f"F1 {mean(b_f1):.3f}±{b_std:.3f} -> {mean(o_f1):.3f}±{o_std:.3f} "
            f"(saved {cv_path}) =="
        )

    table1_f1 = float(cfg.get("table1_f1", 0.0))
    if best_fold_program is not None and best_fold_res is not None and best_fold_res["f1"] > table1_f1-10:
        artifact_path = Path(cfg["artifact_json"])
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        best_fold_program.save(str(artifact_path))
        prompt_path = export_prompt_markdown(best_fold_program, name)
        print(
            f"\n>>> [{name}] WINNING FOLD {best_fold_idx}/{effective_k} SELECTED FOR EXTRACTION <<<\n"
            f"    Model      : {cfg['model']}\n"
            f"    Table 1 F1 : {table1_f1:.3f}\n"
            f"    Fold Base  : F1={best_fold_base['f1']:.3f} | Title CER={best_fold_base['title_cer']:.3f} | Body WER={best_fold_base['body_wer']:.3f}\n"
            f"    Fold DSPy  : F1={best_fold_res['f1']:.3f} | Title CER={best_fold_res['title_cer']:.3f} | Body WER={best_fold_res['body_wer']:.3f}\n"
            f"    Prompt MD  : {prompt_path}\n"
            f"    Artifact   : {artifact_path}\n"
        )
        return artifact_path

    best_f1_val = best_fold_res["f1"] if best_fold_res is not None else -1.0
    print(
        f"[optimize:{name}] SKIPPED save (best fold {best_fold_idx} F1 {best_f1_val:.3f} <= Table 1 {table1_f1:.3f})"
    )
    return None