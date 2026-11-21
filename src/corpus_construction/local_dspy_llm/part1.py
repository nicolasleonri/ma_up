"""
Config + data loading for OCR crops and gold annotations.

Cropped OCR: the crop texts of a page are concatenated with "\\n" (sorted by crop file name) into one page text.
Output parquet: one row per gold article, columns as in C (+ title/subheadline/author).

Run from the ma_up folder:
    python config_and_loading.py
    python config_and_loading.py correo gestion
"""

import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG (edit by hand)
# --------------------------------------------------------------------------- #
def paths(name: str, llm: str) -> dict:
    """Paths for the results parquet, OCR parquets, gold CSV and output parquet.

    `llm` is the short name used in the results parquet filename, not the
    Ollama model name. The results parquet is retained for later comparison;
    it is not read by this loader.
    """
    return dict(
        parquet=f"data/corpus_construction/llm_extraction/{name}/results_cropped_{llm}.parquet",
        ocr_parquet_cropped=f"data/corpus_construction/ocr_extraction/ocr_extraction/{name}/cropped/ocr.parquet",
        ocr_parquet_none=f"data/corpus_construction/ocr_extraction/ocr_extraction/{name}/none/ocr.parquet",
        gold=f"data/corpus_construction/evaluate_extraction/{name}.csv",
        output=f"data/corpus_construction/local_dspy_llm/{name}.parquet",
    )


NEWSPAPERS = {
    "correo": dict(
        model="qwen2.5:7b",
        **paths("correo", "qwen"),
        ocr_variant="cropped",
        filter={
            "config_id": 13,
            # "layout_mode": "cropped",
            "detector": "histogram",
            "binarization": "basic",
            "ocr_extractor": "docling_rapidocr",
        },
    ),
    "elcomercio": dict(
        model="qwen2.5:7b",
        **paths("elcomercio", "qwen"),
        ocr_variant="none",
        filter={
            "config_id": 51,
            "detector": "none",
            "binarization": "none",
            "ocr_extractor": "docling_rapidocr",
        },
    ),
    "gestion": dict(
        model="qwen2.5:7b",
        **paths("gestion", "qwen"),
        ocr_variant="none",
        filter={
            "config_id": 6,
            "detector": "none",
            "binarization": "otsu",
            "ocr_extractor": "docling_rapidocr",
        },
    ),
    "ojo": dict(
        model="qwen2.5:7b",
        **paths("ojo", "qwen"),
        ocr_variant="cropped",
        filter={"config_id": 52, "detector": "ppdoclayout",
                "binarization": "basic", "ocr_extractor": "docling_rapidocr"},
    ),
    "peru21": dict(
        model="deepseek-r1:7b",
        **paths("peru21", "deepseek-r1"),
        ocr_variant="cropped",
        filter={"config_id": 38, "detector": "ppdoclayout",
                "binarization": "none", "ocr_extractor": "docling_rapidocr"},
    ),
    "publimetro": dict(
        model="mistral:7b",
        **paths("publimetro", "mistral"),
        ocr_variant="cropped",
        filter={"config_id": 12, "detector": "histogram",
                "binarization": "none_grayscale", "ocr_extractor": "docling_easyocr"},
    ),
    "trome1": dict(
        model="qwen2.5:7b",
        **paths("trome", "qwen"),
        ocr_variant="cropped",
        filter={"config_id": 7, "detector": "doclayout",
                "binarization": "otsu", "ocr_extractor": "docling_rapidocr"},
    ),
    "trome2": dict(
        model="mistral:7b",
        **paths("trome", "mistral"),
        ocr_variant="cropped",
        filter={"config_id": 34, "detector": "ppdoclayout",
                "binarization": "none", "ocr_extractor": "docling_rapidocr"},
    ),
}

# Source column names (OCR parquet / gold CSV)
PAGE_COL = "image_stem"
OCR_TEXT_COL = "text"
CROP_FILE_COL = "binarize_file"          # crop order: ..._article_001_..., ..._article_002_...
GOLD_FIELDS = ["title", "subheadline", "author", "body"]

# Column names of the saved parquet: one row per gold article; the OCR text of the whole page is repeated
# on every row of that page.
C = dict(
    newspaper="newspaper",
    page_id="page_id",
    ocr_text="ocr_text",        # OCR text of the page (crops concatenated with "\n")
    gold_body="article_text",   # gold text of one article
    strat="strat",              # publication year parsed from page_id, used to stratify the folds
)
MATCH_FIELDS = ["title", "subheadline"]            # fields used to match predicted and gold articles
META_FIELDS = ["title", "subheadline", "author"]   # gold columns (same names) that the model must also extract
SKIP_IF_NA = ["body"]                              # gold body "NA" = not annotated -> not scored

MATCH_THRESHOLD = 0.5       # min. text similarity for a predicted article to count as matching a gold article
META_THRESHOLD = 0.8        # min. similarity for a metadata field to count as correct
W_F1, W_BODY, W_META = 0.5, 0.3, 0.2   # training metric weights


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    page_id: str
    ocr_text: str          # non-empty crop texts under the chosen config, joined with "\n"
    n_crops: int
    strat: str
    gold: list[dict]       # gold articles on this page; missing values are ""


def is_na(x) -> bool:
    return (
        x is None
        or (isinstance(x, float) and math.isnan(x))
        or str(x).strip().lower() in {"", "na", "nan", "none"}
    )


def clean(x) -> str:
    return "" if is_na(x) else str(x).strip()


def strat_key(page_id: str) -> str:
    """Year-based stratification key, e.g. correo_2015-09-11_1 -> '2015'."""
    match = re.search(r"_(\d{4})-\d{2}-\d{2}", page_id)
    return match.group(1) if match else "all"


def apply_filter(df: pd.DataFrame, filt: dict, name: str) -> pd.DataFrame:
    missing = [col for col in filt if col not in df.columns]
    if missing:
        raise KeyError(
            f"[{name}] filter columns missing from OCR parquet: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    for col, value in filt.items():
        df = df[df[col] == value]
    return df


def load_newspaper(name: str) -> list[Page]:
    if name not in NEWSPAPERS:
        raise KeyError(f"Unknown newspaper {name!r}. Available: {list(NEWSPAPERS)}")

    cfg = NEWSPAPERS[name]
    ocr_path = cfg["ocr_parquet_none" if cfg.get("ocr_variant") == "none"
                    else "ocr_parquet_cropped"]

    if not Path(ocr_path).exists():
        raise FileNotFoundError(f"[{name}] OCR parquet not found: {ocr_path}")
    if not Path(cfg["gold"]).exists():
        raise FileNotFoundError(f"[{name}] gold CSV not found: {cfg['gold']}")

    ocr = apply_filter(pd.read_parquet(ocr_path), cfg["filter"], name)
    if PAGE_COL not in ocr.columns or OCR_TEXT_COL not in ocr.columns:
        raise KeyError(
            f"[{name}] expected OCR columns {PAGE_COL!r} and {OCR_TEXT_COL!r}; "
            f"found {list(ocr.columns)}"
        )

    ocr = ocr.copy()
    ocr[OCR_TEXT_COL] = ocr[OCR_TEXT_COL].fillna("").astype(str).str.strip()
    nonempty = ocr[ocr[OCR_TEXT_COL].ne("")]
    if CROP_FILE_COL in nonempty.columns:          # deterministic crop order (article_001, article_002, ...)
        nonempty = nonempty.sort_values(CROP_FILE_COL, kind="stable")

    crops_by_page = {
        str(pid): group[OCR_TEXT_COL].tolist()
        for pid, group in nonempty.groupby(PAGE_COL, sort=False)
    }
    text_by_page = {pid: "\n".join(crops) for pid, crops in crops_by_page.items()}
    rows_by_page = {str(pid): n for pid, n in ocr.groupby(PAGE_COL, sort=False).size().items()}

    gold_df = pd.read_csv(cfg["gold"], dtype=str, keep_default_na=False)
    if PAGE_COL not in gold_df.columns:
        raise KeyError(
            f"[{name}] gold CSV has no {PAGE_COL!r} column; "
            f"found {list(gold_df.columns)}"
        )
    missing_gold_fields = [field for field in GOLD_FIELDS if field not in gold_df.columns]
    if missing_gold_fields:
        raise KeyError(f"[{name}] gold CSV is missing columns: {missing_gold_fields}")

    gold_by_page = defaultdict(list)
    for row in gold_df.to_dict("records"):
        pid = str(row[PAGE_COL]).strip()
        gold_by_page[pid].append({field: clean(row.get(field)) for field in GOLD_FIELDS})

    n_crops = [len(crops_by_page.get(pid, [])) for pid in gold_by_page]
    crop_lengths = [len(text) for crops in crops_by_page.values() for text in crops]
    page_lengths = [len(text_by_page.get(pid, "")) for pid in gold_by_page]

    print(
        f"[{name}] OCR rows under filter: {len(ocr)} "
        f"({len(ocr) - len(nonempty)} empty) | pages in OCR: {len(rows_by_page)} "
        f"| pages in gold: {len(gold_by_page)}"
    )
    print(
        f"  non-empty crops per gold page: min {min(n_crops, default=0)} / "
        f"median {median(n_crops or [0]):.0f} / max {max(n_crops, default=0)} "
        f"| crop length (chars): median {median(crop_lengths or [0]):.0f} / "
        f"max {max(crop_lengths, default=0)} "
        f"| page text (chars): median {median(page_lengths or [0]):.0f} / max {max(page_lengths, default=0)}"
    )

    no_text = [pid for pid in gold_by_page if not crops_by_page.get(pid)]
    if no_text:
        print(f"  WARNING gold pages with NO usable OCR text (will score 0): {no_text}")
    extra = sorted(set(rows_by_page) - set(gold_by_page))
    if extra:
        print(f"  note: OCR pages without gold are ignored: {extra}")
    if ocr.empty:
        print("  WARNING filter matched no rows; check config_id/detector/binarization/ocr_extractor.")

    # Gold page list is the canonical page set, including gold pages without OCR text.
    return [
        Page(
            page_id=pid,
            ocr_text=text_by_page.get(pid, ""),
            n_crops=len(crops_by_page.get(pid, [])),
            strat=strat_key(pid),
            gold=gold_articles,
        )
        for pid, gold_articles in gold_by_page.items()
    ]


def to_rows(name: str, pages: list[Page]) -> list[dict]:
    """One row per gold article; page-level columns are repeated."""
    rows = []
    for page in pages:
        for article in page.gold:
            rows.append({
                C["newspaper"]: name,
                C["page_id"]: page.page_id,
                C["ocr_text"]: page.ocr_text,
                C["strat"]: page.strat,
                **{field: article[field] for field in META_FIELDS},
                C["gold_body"]: article["body"],
            })
    return rows


def gold_summary(name: str, pages: list[Page]) -> None:
    gold = [article for page in pages for article in page.gold]
    n = len(gold)
    print(f"[{name}] gold articles: {n} on {len(pages)} pages")
    for field in GOLD_FIELDS:
        print(f"  {field:12s} annotated in {sum(bool(article[field]) for article in gold)}/{n} articles")


def load_data(names=None) -> dict[str, list[Page]]:
    """Load selected newspapers and return {newspaper: [Page, ...]}."""
    selected = list(NEWSPAPERS) if names is None else list(names)
    return {name: load_newspaper(name) for name in selected}


# Populated by main() when run as a script. Other modules can call load_data()
# to load the desired newspapers without triggering disk I/O at import time.
DATA: dict[str, list[Page]] = {}


if __name__ == "__main__":
    requested = sys.argv[1:] or list(NEWSPAPERS)
    try:
        DATA.update(load_data(requested))
    except Exception as exc:
        print(f"Loading failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    for newspaper, pages in DATA.items():
        gold_summary(newspaper, pages)
        print(
            {page.page_id: (page.n_crops, len(page.gold)) for page in pages},
            "(crops, gold articles) per page\n",
        )

        save_path = Path(NEWSPAPERS[newspaper]["output"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(to_rows(newspaper, pages)).to_parquet(save_path, index=False)
        print(f"saved {save_path}")