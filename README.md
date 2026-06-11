# Masterarbeit UP

Research pipeline for building, annotating and analyzing a large-scale corpus of Peruvian newspaper articles about the informal economy.

## Project Structure

```
ma_up/
├── config/
│   ├── pipeline_config.yaml    # Determines algorithms to be used
│   ├── slurm_templates.sh      # Formats bash-scripts
│   └── gpu_allocation.yaml     # Presets HPC resources
│
├── src/
│   ├── schemas/                 # Shared data models
│   │   ├── article.py
│   │   ├── paragraph.py
│   │   ├── sentence.py
│   │   ├── annotations.py
│   │   └── provenance.py
│   │
│   ├── corpus_acquisition/
│   │   ├── crawler.py
│   │   ├── downloader.py
│   │   ├── metadata_extractor.py
│   │   └── crawl_registry.py
│   │
│   ├── corpus_construction/
│   │   ├── pipeline.py          # Main orchestrator + checkpointing
│   │   ├── benchmark.py
│   │   ├── config_generator.py  # Generates 13,608 possible configs
│   │   ├── preprocessing.py /   # Algorithms
│   │   │   ├── base.py 
│   │   │   ├── otsu.py
│   │   │   ├── niblack.py 
│   │   │   └── ...
│   │   ├── layout.py /          # Algorithms
│   │   │   ├── base.py 
│   │   │   ├── layoutparser.py
│   │   │   ├── doclayout_yolo.py 
│   │   │   └── ...
│   │   ├── vlm.py /             # w/ Dspy - GPU intensive
│   │   │   ├── base.py 
│   │   │   ├── olmocr.py
│   │   │   ├── rolmocr.py 
│   │   │   └── ...
│   │   ├── llm.py /             # w/ Dspy - GPU intensive
│   │   │   ├── base.py 
│   │   │   ├── llama31.py
│   │   │   ├── mistral.py 
│   │   │   └── ...
│   │   └── evaluation.py
│   │
│   ├── corpus_annotation/
│   │   ├── topic_filtering.py   # Needs results from semantic_analysis.py
│   │   ├── llm_preannotation.py # uses vLLM - GPU intensive
│   │   ├── csv_exporter.py      # Used before manual validation
│   │   └── csv_exporter.py      # Used after manual validation
│   │
│   ├── am_model/
│   │   ├── train_am.py          # Model Training w/ datasets - GPU intensive
│   │   ├── evaluate_am.py
│   │   ├── infer_am.py          # uses best model
│   │   └── domain_adaptation.py # MLM Training - GPU intensive
│   │
│   ├── feature_extraction/
│   │   ├── engine.py 
│   │   └── extractors /
│   │       ├── semantic.py      # BERTopic w/ majority voting - GPU intensive
│   │       ├── sentiment.py     # HF models w/ majority voting
│   │       ├── pos.py           # Models w/ majority voting (?)
│   │       ├── interpretative_frames.py # Clustering only, manual review
│   │       └── discourse.py     # 1 Model from HF
│   │
│   ├── aggregation/
│   │   ├── sentiment.py
│   │   ├── argumentation.py
│   │   ├── pos.py
│   │   ├── sentiment.py
│   │   ├── discourse.py
│   │   └── article_builder.py
│   │
│   ├── providers/
│   │   ├── llm_provider.py
│   │   └── vlm_provider.py
│   │
│   ├── utils/
│   │   ├── config_tracker.py     # Track completed configs, avoid repeats
│   │   ├── checkpoint.py         # Resume from last checkpoint
│   │   ├── data_serialization.py # Parquet I/O
│   │   ├── hpc_job_manager.py    # SLURM integration
│   │   └── logging.py
│   │
│   └── workflows/
│       ├── build_corpus.py
│       ├── annotate_corpus.py
│       ├── train_am.py
│       ├── extract_features.py
│       └── build_final_dataset.py
│
├── data/
│   ├── raw/                       # Downloaded newspaper pages (.jpg)
│   │   ├── images/
│   │   ├── metadata/
│   │   │   └── raw_metadata.parquet
│   │   └── crawl_logs/
│   │
│   ├── ocr_output/
│   │   ├── experiment_results.csv   # Saves: config_id, status, CER, F1, etc.
│   │   ├── checkpoints/           # Latest state per newspaper
│   │   │   ├── newspaper_1_checkpoint.pkl
│   │   │   └── ...
│   │   └── best_pipeline.json
│   │
│   ├── annotated/
│   │   ├── manual_validation.csv
│   │   └── validated_final.parquet
│   │
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
├── main.py                        # Orchestration (only calls workflows/ and config/)
└── requirements.txt
```

## Workflow

0. Corpus Acquisition
   Crawl the digital newspaper archives, download newspaper page images and extract metadata. Store downloaded pages and metadata in the raw data directory.
1. Generate OCR configurations
   Build the full set of preprocessing/layout/VLM/LLM combinations and store metadata in experiment_results.parquet.
2. Run parallel OCR benchmarking
   Execute configurations on HPC nodes, evaluate against the gold standard, and append results to experiment_results.parquet.
3. Select the best pipeline
   Aggregate benchmark results and save the optimal configuration for each newspaper in best_pipeline.json.
4. Build the article-level corpus
   Run the selected OCR pipeline over all newspaper pages and produce structured article objects.
5. Filter and annotate the corpus
   Apply BERTopic filtering, export paragraphs for manual validation and import the validated annotations.
6. Train and evaluate the AM model
   Perform domain adaptation, fine-tuning, evaluation and full-corpus inference.
7. Extract linguistic features
   Run semantic, sentiment, POS, discourse and argumentative extractors, then aggregate all outputs to article level.
8. Build the final analytical dataset
   Merge all article-level variables into unified_corpus.parquet for statistical analysis in R.

