# ma_up

informality-analysis/
├── config/
│   ├── pipeline_config.yaml
│   ├── slurm_templates.sh
│   └── gpu_allocation.yaml
│
├── src/
│   ├── corpus_construction/
│   │   ├── ocr_pipeline.py        # Main orchestrator + checkpointing
│   │   ├── config_generator.py    # Generates 13,608 configs
│   │   ├── preprocessing.py
│   │   ├── layout_detection.py
│   │   ├── vlm_extraction.py      # GPU
│   │   ├── llm_postprocessing.py  # GPU
│   │   └── evaluation.py
│   │
│   ├── corpus_annotation/
│   │   ├── topic_filtering.py
│   │   ├── llm_preannotation.py
│   │   ├── csv_exporter.py
│   │   └── csv_importer.py
│   │
│   ├── model_finetuning/
│   │   ├── domain_adaptation.py
│   │   ├── am_finetuning.py
│   │   └── model_evaluation.py
│   │
│   ├── feature_extraction/
│   │   ├── semantic_analysis.py
│   │   ├── sentiment_analysis.py
│   │   ├── pos_analysis.py
│   │   ├── discourse_analysis.py
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
│   │   ├── configs_metadata.db    # SQLite: config_id, status, CER, F1, etc.
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
├── analysis_scripts/              # Standalone R scripts (you manage these)
│   └── hypothesis_testing.R
│
├── tests/
│   ├── test_ocr_pipeline.py
│   ├── test_annotation.py
│   └── test_aggregation.py
│
├── main.py                        # Orchestration
└── requirements.txt
