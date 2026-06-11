# Masterarbeit UP

## Architecture

```
ma_up/
├── config/
│   ├── pipeline_config.yaml # Determines algorithms to be used
│   ├── slurm_templates.sh # Formats bash-scripts
│   └── gpu_allocation.yaml # Determines combinations of RAM & GPU
│
├── src/ 
│   ├── corpus_construction/
│   │   ├── ocr_pipeline.py        # Main orchestrator + checkpointing
│   │   ├── config_generator.py    # Generates 13,608 possible configs
│   │   ├── preprocessing.py       # Algorithms
│   │   ├── layout_detection.py    # Algorithms 
│   │   ├── vlm_extraction.py      # uses vLLM (with Dspy) - GPU intensive
│   │   ├── llm_postprocessing.py  # vLLM (with Dspy) - GPU intensive
│   │   └── evaluation.py
│   │
│   ├── corpus_annotation/
│   │   ├── topic_filtering.py    # Needs results from semantic_analysis.py
│   │   ├── llm_preannotation.py  # uses vLLM - GPU intensive
│   │   ├── csv_exporter.py       # Before manual validation
│   │   └── csv_importer.py       # After manual validation
│   │
│   ├── model_finetuning/
│   │   ├── domain_adaptation.py  # MLM Training - GPU intensive
│   │   ├── am_finetuning.py      # uses different datasets combinations - GPU intensive
│   │   └── model_evaluation.py   # uses vLLM - GPU intensive
│   │
│   ├── feature_extraction/
│   │   ├── semantic_analysis.py  # BERTopic w/ majority voting - GPU intensive
│   │   ├── sentiment_analysis.py # HF models w/ majority voting
│   │   ├── pos_analysis.py       # Models w/ majority voting (?)
│   │   ├── discourse_analysis.py # 1 Model from HF
│   │   ├── interpretative_frames.py  # Clustering only, manual review
│   │   └── aggregation.py
│   │
│   └── utils/
│       ├── config_tracker.py      # Track completed configs, avoid repeats
│       ├── checkpoint.py          # Resume from last checkpoint
│       ├── data_serialization.py  # Parquet I/O
│       ├── hpc_job_manager.py     # SLURM integration
│       └── logging.py
│
├── data/
│   ├── raw/                       # Downloaded newspaper pages (.jpg)
│   ├── ocr_output/
│   │   ├── configs_metadata.csv   # Saves: config_id, status, CER, F1, etc.
│   │   ├── checkpoints/           # Latest state per newspaper
│   │   │   ├── newspaper_1_checkpoint.pkl
│   │   │   └── ...
│   │   └── best_pipeline.json
│   ├── annotated/
│   │   ├── manual_validation.csv
│   │   └── validated_final.parquet
│   └── final_dataset/
│       └── unified_corpus.parquet
│
├── hpc_scripts/
│   ├── submit_ocr_jobs.sh
│   ├── submit_training.sh
│   └── aggregate_results.sh
│
├── analysis_scripts/              # Standalone R scripts
│   └── dea.R
│   └── hypothesis_testing.R
│   └── graphs.R
│   └── ...
│
├── tests/
│   ├── test_ocr_pipeline.py
│   ├── test_annotation.py
│   └── test_aggregation.py
│
├── main.py                        # Orchestration
└── requirements.txt
```

## Workflow

```
1. CONFIG GENERATION (single node)
   → Generates 13,608 configs
   → Saves to configs_metadata.csv

2. PARALLEL EVALUATION (GPU array job)
   ├─ Job 1: Configs 1-100 (GPU:0)
   ├─ Job 2: Configs 101-200 (GPU:1)
   └─ Job N: Configs 13500-13608 (GPU:N)
   → Outputs: evaluation_results.parquet

3. BEST PIPELINE SELECTION (single node)
   → Aggregates results
   → Saves best_pipeline.json

4. ANNOTATION & VALIDATION (local/interactive)
   → Export: manual_validation.csv
   → You review & edit
   → Import: validated_final.parquet

5. MODEL FINETUNING (GPU job)
   → Uses best config + validated annotations
   → Saves model checkpoints

6. FEATURE EXTRACTION (CPU job, parallelizable)
   → Runs all 5 linguistic layers
   → Outputs: unified_corpus.parquet

7. STATISTICAL ANALYSIS (R on local machine)
   → Read unified_corpus.parquet
   → Run hypothesis tests (H1-H11)
   → Generate tables & visualizations
```
