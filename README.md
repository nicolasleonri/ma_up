# Masterarbeit UP

Research pipeline for building, annotating and analyzing a large-scale corpus of Peruvian newspaper articles about the informal economy.

## Project Structure

```
ma_up/
├── config/                      # Includes passwords
│
├── requirements/                      # txt
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
│   │   ├── browser.py
│   │   ├── crawl_registry.py
│   │   ├── crawler.py
│   │   ├── metadata_extractor.py
│   │   └── downloader.py
│   │
│   ├── corpus_construction/
│   │   ├── layout_detection.py /   # Algorithms
│   │   │   ├── pipeline.py 
│   │   │   └── steps.py
│   │   │    
│   │   ├── preprocessing.py /          # Algorithms
│   │   │   ├── pipeline.py 
│   │   │   └── steps.py
│   │   │    
│   │   └──  vlm_extraction.py /             # w/ Dspy - GPU intensive
│   │       ├── pipeline.py 
│   │       └── steps.py    
│   │
│   ├── corpus_annotation/
│   │   └── llm_preannotation.py # Needs results from semantic_analysis.py
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
│   │   ├── discourse.py
│   │   └── article_builder.py
│   │
│   ├── utils/
│   │   └── config_loader.py
│   │
│   └── workflows/
│       ├── acquire_corpus.py
│       ├── evaluate_extraction.py
│       ├── finetune_vlm.py
│       ├── layout_detection.py
│       ├── preprocess_images.py
│       └── vlm_extraction.py
│
├── data/
│   ├── raw/                       # Downloaded newspaper pages (.jpg)
│   │   ├── images/
│   │   ├── metadata/
│   │   │   └── raw_metadata.parquet
│   │   └── crawl_logs/
│   │
│   ├── gold_standard/                       # Downloaded newspaper pages (.jpg)
│   │   ├── newspaper_1.parquet
│   │   ├── newspaper_2.parquet
│   │   └── ...
│   │
│   ├── ocr_output/
│   │   ├── experiment_results.parquet   # Saves: config_id, status, CER, F1, etc.
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
└── analysis_scripts/              # Standalone R scripts
    └── dea.R
    └── hypothesis_testing.R
    └── graphs.R
    └── ...
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

