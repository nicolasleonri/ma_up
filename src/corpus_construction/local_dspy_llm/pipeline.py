#!/usr/bin/env python3
"""
pipeline.py

CLI entry point for DSPy dataset preparation, evaluation, and 5-fold CV optimization.
Leaves Step 4 (full-archive production extraction) out, saving compiled DSPy JSON
artifacts ready for downstream inference.

Examples (run from ma_up root):
    # Run all steps (build -> eval -> optimize) for all newspapers
    python -m src.corpus_construction.local_dspy_llm.pipeline --all

    # Only build/refresh cached datasets
    python -m src.corpus_construction.local_dspy_llm.pipeline --build --newspapers correo elcomercio

    # Evaluate zero-shot baseline only (using evaluate_extraction.py Hungarian logic)
    python -m src.corpus_construction.local_dspy_llm.pipeline --eval --newspapers gestion

    # Run 5-fold CV + compile final DSPy JSON artifact (uses cached dataset if present)
    python -m src.corpus_construction.local_dspy_llm.pipeline --optimize --newspapers peru21
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.corpus_construction.local_dspy_llm.steps import (  # noqa: E402
    NEWSPAPERS,
    PageArticleExtractor,
    load_dataset,
    make_lm,
    step_build_dataset,
    step_evaluate_pages,
    step_run_cv_and_optimize,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Step flags
    ap.add_argument("--all", action="store_true", help="Run --build, --eval, and --optimize")
    ap.add_argument("--build", action="store_true", help="Step 1: Build & cache region-level dataset parquets")
    ap.add_argument("--eval", action="store_true", help="Step 2: Evaluate zero-shot baseline on cached dataset")
    ap.add_argument("--optimize", action="store_true", help="Step 3: Run k-fold CV & save compiled DSPy artifact")

    # Scope & execution options
    ap.add_argument("--newspapers", nargs="+", default=list(NEWSPAPERS), choices=list(NEWSPAPERS))
    ap.add_argument("--model", default=None, help="Filter newspapers by Ollama model (e.g. qwen2.5:7b)")
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama API base URL")
    ap.add_argument("--k", type=int, default=5, help="Number of stratified CV folds (default: 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for stratified folding")
    ap.add_argument(
        "--optimizer",
        default="bootstrap",
        choices=["bootstrap", "random_search"],
        help="DSPy optimizer ('bootstrap' recommended for N=10 pages/newspaper)",
    )
    ap.add_argument("--threads", type=int, default=2, help="Threads for random_search candidate evaluation")
    ap.add_argument("--skip-cv", action="store_true", help="Skip k-fold CV and compile directly on all pages")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (args.all or args.build or args.eval or args.optimize):
        print("No step flag specified. Pass --all, --build, --eval, or --optimize (see --help).", file=sys.stderr)
        raise SystemExit(1)

    run_build = args.all or args.build
    run_eval = args.all or args.eval
    run_optimize = args.all or args.optimize

    selected = [
        name for name in args.newspapers
        if args.model is None or NEWSPAPERS[name]["model"] == args.model
    ]

    for name in selected:
        cfg = NEWSPAPERS[name]
        print(f"\n==================== {name.upper()} ({cfg['model']}) ====================")

        if run_build:
            step_build_dataset(name, overwrite=True)

        if run_eval:
            pages = load_dataset(name)
            dspy.configure(lm=make_lm(args.host, cfg["model"], temperature=0.0))
            metrics = step_evaluate_pages(PageArticleExtractor(), pages, verbose=False)
            print(
                f"[eval:{name}] Baseline -> "
                f"F1: {metrics['f1']:.3f} | "
                f"Title CER: {metrics['title_cer']:.3f} | "
                f"Body WER: {metrics['body_wer']:.3f} | "
                f"Errors: {metrics['errors']}"
            )

        if run_optimize:
            step_run_cv_and_optimize(
                name=name,
                host=args.host,
                k=args.k,
                seed=args.seed,
                optimizer_type=args.optimizer,
                threads=args.threads,
                skip_cv=args.skip_cv,
            )


if __name__ == "__main__":
    main()