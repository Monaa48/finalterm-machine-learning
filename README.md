# Final Term - Machine Learning (End-to-End)

## Identity
- Name  : Anom Nur Maulid
- Class : TK4601
- NIM   : 1103223193

## Repository Purpose
This repository contains an end-to-end machine learning pipeline for **fraud transaction detection**.  
The objective is to predict the probability that an online transaction is fraudulent (`isFraud`).

## Project Overview (Fraud Transaction)
Workflow implemented in the notebook:
1. Load training and testing transaction data
2. Separate target label (`isFraud`) and remove identifier (`TransactionID`) from features
3. Data preprocessing:
   - Handle missing values (median for numeric, most-frequent for categorical)
   - Encode categorical variables using One-Hot Encoding
4. Train and compare models:
   - Logistic Regression (baseline)
   - Random Forest (final model)
5. Model evaluation:
   - ROC-AUC
   - PR-AUC (Average Precision) for imbalanced classification
   - F1-based thresholding + classification report + confusion matrix
6. Generate prediction file for test data:
   - `submission_fraud_rf_baseline.csv` (`TransactionID`, `isFraud`)

## Models and Metrics (Validation Results)
The dataset is imbalanced (fraud ratio ~3.5%), therefore PR-AUC is used as a key metric.

| Model | ROC-AUC | PR-AUC |
|------|--------:|------:|
| Logistic Regression (baseline) | ~0.745 | ~0.137 |
| Random Forest (baseline, final) | ~0.941 | ~0.736 |

Additional evaluation (Random Forest, threshold selected by F1):
- Best threshold (F1-based): ~0.1967  
- Fraud class (label 1): Precision ~0.7247, Recall ~0.6637, F1 ~0.6929

## Repository Navigation
- `01_fraud_transaction.ipynb` (or `Fraud_Transaction.ipynb`):  
  Main notebook containing the complete pipeline (preprocessing, training, evaluation, and submission creation).
- `submission_fraud_rf_baseline.csv`:  
  Final prediction output for the test dataset.

## Notes on Dataset
The raw datasets (`train_transaction.csv`, `test_transaction.csv`) are **not uploaded to GitHub** due to size constraints.  
To run the notebook:
1. Place the datasets in your Google Drive (or local folder)
2. Update the `DATA_DIR` path inside the notebook accordingly
3. Run all cells in sequence

## Requirements
Main libraries used:
- pandas
- numpy
- scikit-learn
