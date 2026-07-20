# Text Classification Framework: IndoBERT Implementation

## Research Overview
This repository contains the IndoBERT implementation, which serves as the representative Indonesian transformer architecture within the current research framework. The overarching study conducts a comparative analysis between traditional machine learning models (Random Forest and Logistic Regression) and this transformer-based approach. The primary objective is the rigorous evaluation of classification performance and data stability when applied to social media discourse governed by the Indonesian Electronic Information and Transactions (ITE) Law.

## Environment Requirements
The framework requires Python 3.19 for optimal execution and dependency resolution.

To install the requisite packages, execute the following command in the terminal:
```bash
pip install -r requirements.txt
```

## Execution Pipeline

### 1. Dataset Construction
Raw data synthesis is required prior to initiating the training sequence. Execute the following command to download, parse, and aggregate the distinct data sources:
```bash
python build_dataset.py
```
This operational script utilizes a composite key mapping to generate the complete dataset, subsequently persisting it locally as an Excel file for downstream ingestion.

### 2. Model Training
To commence the fine-tuning and training procedure, execute:
```bash
python train.py
```
Due to the extensive computational overhead and prolonged execution time inherent to transformer architectures, this process does not automatically trigger the evaluation module. This decoupling is necessary to preserve absolute visibility and continuous monitoring of the training metrics without interference.

### 3. Model Evaluation
Following the successful completion of the training phase, evaluation must be initiated manually:
```bash
python evaluation.py
```
This script computes the evaluation metrics, constructs classification reports, and generates confusion matrix visual artifacts, storing them within the designated local model directory.

## Experimental Configurations (Boolean Flags)
The experimental pipeline is controlled by specific parameters defined in `configuration.py`. These variables dictate the architectural and data-handling methodologies applied during execution.

*   **USE_HIERARCHICAL**: When configured to True, the pipeline implements a two-stage hierarchical classification schema. It segregates instances into Neutral versus Non-Neutral at Level 1, proceeding to a multiclass evaluation at Level 2. When False, the framework forces a standard flat multiclass classification structure.
*   **USE_UNDERSAMPLING**: When configured to True, a stratified random undersampling strategy is deployed to artificially balance the empirical class distribution within the training set. When False, the model ingests the original, imbalanced data distribution.
*   **USE_PREPROCESSING**: Within the context of the IndoBERT research methodology, this parameter is considered mandatory (effectively True) to enforce a minimalist preprocessing scheme optimized for subword tokenization. However, the boolean toggle remains accessible for researchers intending to experiment with the ingestion of entirely unprocessed raw data (False).
*   **RANDOM_SEED**: Controls the deterministic initialization of the model weights and data partitions. Unlike the traditional algorithmic implementations within this research, automated multi-seed batch execution scripts are intentionally omitted. Running multiple seeds concurrently or sequentially via an automated shell script obscures operational monitoring and introduces hardware monitoring complications during extended utilization. Researchers must modify this seed manually for independent experimental iterations.

## License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.