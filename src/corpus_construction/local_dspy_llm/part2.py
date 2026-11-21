#!/usr/bin/env python3
"""
Zero-shot instruction optimization (DSPy GEPA) for the OCR-text -> articles step of the pipeline.

Per newspaper (independently, one model each, set manually in NEWSPAPERS below):
  1. shuffled k-fold over pages
  2. per fold: baseline (hand-written instruction) vs. GEPA-optimized instruction on the held-out fold
     - reported metric: simple article-level F1
     - training metric (GEPA only): 0.5 * F1 + 0.3 * (1 - body error) + 0.2 * metadata accuracy, with text feedback
  3. final optimization on ALL pages of the newspaper -> instructions/<newspaper>.txt

The LLM only returns line ranges + metadata (cheap output); article bodies are assembled in code
from the OCR lines.

Install:   uv pip install pandas pyarrow rapidfuzz scipy numpy
First run: python optimize_format.py --parquet data.parquet --inspect       # prints head + summary, then exits
Run:       python optimize_format.py --parquet data.parquet --newspapers newspaper_a --k 5
           (start one process per newspaper to use the three models in parallel)
"""
import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

import dspy
import numpy as np
import pandas as pd
from pydantic import create_model
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------- #
# CONFIG  
# --------------------------------------------------------------------------- #
NEWSPAPERS = {
    "correo": dict(model="mistral:7b", filter={
        # "config_id": 13,
        # "detector": "histogram",
        # "binarization": "basic",
        # "ocr_extractor": "docling_rapidocr"
        }),
    "elcomercio": dict(model="qwen2.5:7b", filter={"config_id": 51, "detector": "none", "binarization": "none", "ocr_extractor": "docling_rapidocr", }),
    "gestion": dict(model="qwen2.5:7b", filter={"config_id": 6, "detector": "none", "binarization": "otsu", "ocr_extractor": "docling_rapidocr", }),
    "ojo": dict(model="qwen2.5:7b", filter={"config_id": 52, "detector": "ppdoclayout", "binarization": "basic", "ocr_extractor": "docling_rapidocr", }),
    "peru21": dict(model="deepseek-r1:7b", filter={"config_id": 38, "detector": "ppdoclayout", "binarization": "none", "ocr_extractor": "docling_rapidocr", }),
    "publimetro": dict(model="mistral:7b", filter={"config_id": 12, "detector": "histogram", "binarization": "none_grayscale", "ocr_extractor": "docling_easyocr", }),
    "trome1": dict(model="qwen2.5:7b", filter={"config_id": 7, "detector": "doclayout", "binarization": "otsu", "ocr_extractor": "docling_rapidocr", }),
    "trome2": dict(model="mistral:7b", filter={"config_id": 34, "detector": "ppdoclayout", "binarization": "none", "ocr_extractor": "docling_rapidocr", }),
}

# Column names in the parquet. ASSUMED LAYOUT: one row per gold article; the OCR text of the whole page
# is repeated on every row of that page. Adjust after running --inspect.
C = dict(
    newspaper="newspaper",
    page_id="page_id",
    ocr_text="ocr_text",        # OCR text of the page (output of the best pipeline config)
    gold_body="article_text",   # gold text of one article
)

META_FIELDS = ["title", "author"]   # gold columns (same names) that the model must also extract

# --------------------------------------------------------------------------- #
# EVALUATION
# --------------------------------------------------------------------------- #

MATCH_THRESHOLD = 0.5       # min. text similarity for a predicted article to count as matching a gold article
META_THRESHOLD = 0.8        # min. similarity for a metadata field to count as correct
W_F1, W_BODY, W_META = 0.5, 0.3, 0.2   # training metric weights

# --------------------------------------------------------------------------- #
# Data: dict per newspaper -> list of pages
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    page_id: str
    lines: list
    numbered: str
    gold: list          # list of dicts: {"body": ..., <META_FIELDS>...}


def inspect_df(df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("=== HEAD ===")
    print(df.head())
    print("\n=== INFO ===")
    df.info()
    print("\n=== SUMMARY ===")
    print(df.describe(include="all").T)
    print("\n=== UNIQUE VALUES PER COLUMN ===")
    print(df.nunique())


def build_pages(df: pd.DataFrame, filt: dict) -> list:
    for col, val in filt.items():
        df = df[df[col] == val]
    pages = []
    for pid, g in df.groupby(C["page_id"], sort=False):
        lines = [l for l in str(g[C["ocr_text"]].iloc[0]).splitlines() if l.strip()]
        numbered = "\n".join(f"[L{i:03d}] {l}" for i, l in enumerate(lines, 1))
        gold = [{"body": r[C["gold_body"]], **{f: r[f] for f in META_FIELDS}} for r in g.to_dict("records")]
        pages.append(Page(str(pid), lines, numbered, gold))
    return pages


def to_example(p: Page) -> dspy.Example:
    return dspy.Example(page=p.numbered, lines=p.lines, gold_articles=p.gold, page_id=p.page_id).with_inputs("page")


def shuffled_folds(pages: list, k: int, seed: int) -> list:
    """Shuffle pages with a fixed seed, then distribute them evenly across folds."""
    if k < 2:
        raise ValueError("--k must be at least 2")
    rng = random.Random(seed)
    order = list(pages)
    rng.shuffle(order)
    folds = [[] for _ in range(k)]
    for i, page in enumerate(order):
        folds[i % k].append(page)
    return folds


# --------------------------------------------------------------------------- #
# DSPy program: the model returns line ranges + metadata only
# --------------------------------------------------------------------------- #
Article = create_model(
    "Article",
    start_line=(int, ...),
    end_line=(int, ...),
    **{f: (str, "") for f in META_FIELDS},
)


class FormatPage(dspy.Signature):
    """You receive the OCR text of one newspaper page. Every line is numbered like [L012].
    Identify each distinct news article on the page. For each article return the number of the first
    and last line of its body (use only line numbers that exist) and the requested metadata.
    Skip page headers, page numbers, advertisements and other non-article text.
    Never invent line numbers and never rewrite the text."""

    page: str = dspy.InputField(desc="OCR text of one newspaper page, one numbered line per row")
    articles: list[Article] = dspy.OutputField(desc="One entry per article, in reading order")


def program_from_txt(path: str) -> dspy.Module:
    """For the pipeline: build the extraction program from a saved instruction file."""
    return dspy.Predict(FormatPage.with_instructions(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def sim(a, b) -> float:
    return Levenshtein.normalized_similarity(norm(a), norm(b))


def assemble(arts, lines) -> list:
    out = []
    for a in arts:
        s, e = int(a.start_line), int(a.end_line)
        out.append("\n".join(lines[s - 1:e]) if 1 <= s <= e <= len(lines) else "")  # invalid range -> empty
    return out


def match(pred_bodies, gold_bodies) -> list:
    """Hungarian matching on text similarity; returns [(pred_idx, gold_idx)] above the threshold."""
    if not pred_bodies or not gold_bodies:
        return []
    S = np.array([[sim(p, g) for g in gold_bodies] for p in pred_bodies])
    rows, cols = linear_sum_assignment(-S)
    return [(i, j) for i, j in zip(rows, cols) if S[i, j] >= MATCH_THRESHOLD]


def snippet(s, n=60) -> str:
    return norm(s)[:n]


def score_page(ex, pred) -> dict:
    arts = pred.articles
    gold = ex.gold_articles
    bodies = assemble(arts, ex.lines)
    gbodies = [g["body"] for g in gold]
    m = match(bodies, gbodies)
    tp = len(m)
    prec = tp / len(bodies) if bodies else (1.0 if not gold else 0.0)
    rec = tp / len(gold) if gold else (1.0 if not bodies else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if m:
        err = mean(min(1.0, Levenshtein.distance(norm(bodies[i]), norm(gbodies[j])) / max(len(norm(gbodies[j])), 1))
                   for i, j in m)
        meta = mean(sim(getattr(arts[i], f), gold[j][f]) >= META_THRESHOLD for i, j in m for f in META_FIELDS) \
            if META_FIELDS else 1.0
    else:
        err, meta = 1.0, 0.0
    score = W_F1 * f1 + W_BODY * (1 - err) + W_META * meta

    fb = [f"Gold has {len(gold)} articles, you returned {len(bodies)}; {tp} matched (F1={f1:.2f})."]
    hit_g, hit_p = {j for _, j in m}, {i for i, _ in m}
    for j in [j for j in range(len(gold)) if j not in hit_g][:3]:
        fb.append(f"Missed gold article starting with: '{snippet(gbodies[j])}'")
    for i in [i for i in range(len(bodies)) if i not in hit_p][:3]:
        why = " (line range invalid)" if not bodies[i] else ""
        fb.append(f"Returned article lines {arts[i].start_line}-{arts[i].end_line} matches no gold article{why}.")
    if m:
        fb.append(f"Body error on matched articles: {err:.2f}; metadata accuracy: {meta:.2f}.")
    return dict(f1=f1, score=score, feedback=" ".join(fb))


def train_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    r = score_page(gold, pred)
    return dspy.Prediction(score=r["score"], feedback=r["feedback"])


# --------------------------------------------------------------------------- #
# LMs, evaluation, optimization
# --------------------------------------------------------------------------- #
def make_lm(host: str, model: str, temperature=0.0, max_tokens=None) -> dspy.LM:
    mt = max_tokens or (8192 if "r1" in model else 2048)   # r1 writes a long reasoning block first
    return dspy.LM(f"ollama_chat/{model}", api_base=host, api_key="",
                   temperature=temperature, max_tokens=mt, num_retries=2)


def evaluate(program, lm, pages, threads) -> dict:
    h0 = len(lm.history)

    def one(p):
        ex = to_example(p)
        try:
            return score_page(ex, program(page=ex.page))
        except Exception as e:  # parse failures etc. count as 0
            return dict(f1=0.0, score=0.0, error=str(e)[:100])

    with ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(one, pages))
    usage = [(h.get("usage") or {}) for h in lm.history[h0:]]
    return dict(
        f1=mean(r["f1"] for r in rows),
        score=mean(r["score"] for r in rows),
        errors=sum("error" in r for r in rows),
        max_prompt=max((u.get("prompt_tokens", 0) for u in usage), default=0),
        avg_completion=mean([u.get("completion_tokens", 0) for u in usage] or [0]),
    )


def optimize(train_pages, refl_lm, auto, threads, seed):
    val = shuffled_folds(train_pages, 4, seed)[0] if len(train_pages) >= 4 else train_pages
    val_ids = {p.page_id for p in val}
    tr = [p for p in train_pages if p.page_id not in val_ids] or train_pages
    gepa = dspy.GEPA(metric=train_metric, reflection_lm=refl_lm, auto=auto, num_threads=threads)
    return gepa.compile(dspy.Predict(FormatPage),
                        trainset=[to_example(p) for p in tr],
                        valset=[to_example(p) for p in val])


def fmt_ms(xs) -> str:
    return f"{mean(xs):.3f} ± {stdev(xs) if len(xs) > 1 else 0:.3f}"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--inspect", action="store_true", help="print head + summary and exit")
    ap.add_argument("--newspapers", nargs="+", default=list(NEWSPAPERS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--reflection-model", default=None, help="default: the newspaper's own model")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="instructions")
    ap.add_argument("--model", default=None, choices=["mistral:7b", "qwen2.5:7b", "deepseek-r1:7b"])
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    inspect_df(df)
    if args.inspect:
        return

    DATA = {
        n: build_pages(df[df[C["newspaper"]] == n], NEWSPAPERS[n]["filter"])
        for n in args.newspapers
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, pages in DATA.items():
        model = NEWSPAPERS[name]["model"]

        if args.model is not None:
            if model != args.model:
                continue

        if not pages:
            print(f"\n######## {name} | model {model} | no pages matched; skipping")
            continue

        n_lines = [len(p.lines) for p in pages]
        print(f"\n######## {name} | model {model} | {len(pages)} pages | lines/page "
              f"min {min(n_lines)} / median {median(n_lines):.0f} / max {max(n_lines)}")

        task_lm = make_lm(args.host, model)
        refl_lm = make_lm(args.host, args.reflection_model or model, temperature=1.0, max_tokens=8192)
        dspy.configure(lm=task_lm)

        folds = shuffled_folds(pages, args.k, args.seed)
        base_f1, opt_f1 = [], []
        for i, test in enumerate(folds):
            train = [p for j, f in enumerate(folds) if j != i for p in f]
            b = evaluate(dspy.Predict(FormatPage), task_lm, test, args.threads)
            print(b)
            best = optimize(train, refl_lm, args.auto, args.threads, args.seed)
            print(best)
            o = evaluate(best, task_lm, test, args.threads)
            base_f1.append(b["f1"])
            opt_f1.append(o["f1"])
            print(f"fold {i + 1}/{args.k} ({len(test)} pages) | F1 baseline {b['f1']:.3f} -> optimized {o['f1']:.3f} "
                  f"| parse errors {b['errors']} -> {o['errors']} "
                  f"| max prompt tokens {max(b['max_prompt'], o['max_prompt'])} "
                  f"| avg completion tokens {o['avg_completion']:.0f}")

        print(f"[{name} | {model}] F1 baseline {fmt_ms(base_f1)}  ->  optimized {fmt_ms(opt_f1)}  (mean ± SD over folds)")

        final = optimize(pages, refl_lm, args.auto, args.threads, args.seed)   # all pages, no held-out score
        path = out_dir / (re.sub(r"[^\w.-]+", "_", name) + ".txt")
        path.write_text(final.signature.instructions, encoding="utf-8")
        print(f"final instruction written to {path}\n--- instruction ---\n{final.signature.instructions}\n---")


if __name__ == "__main__":
    main()