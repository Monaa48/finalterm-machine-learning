# Final Term - Machine Learning (End-to-End)

## Identity
- Name  : Anom Nur Maulid
- Class : TK4601
- NIM   : 1103223193

## Repository Purpose
This repository contains end-to-end projects for the Machine Learning final term, covering:
- Fraud transaction detection (classification)
- Regression (predict continuous target from numeric features)
- Image classification using CNN (Fish dataset)

---

# 01 — Fraud Transaction Detection (Classification)

## Objective
Predict the probability that an online transaction is fraudulent (`isFraud`).

## Workflow Summary
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

---

# 02 — Regression (Continuous Target Prediction)

## Objective
Predict a continuous target value (e.g., song release year) from numeric audio features.

## Dataset Structure
- First column: target (`y`)
- Remaining columns: numeric features (`X`)

## Workflow Summary
1. Load dataset and perform sanity checks (shape, target range)
2. Train/validation split
3. Baseline model:
   - Ridge Regression (with scaling)
4. Non-linear model comparison:
   - Gradient Boosting Regressor
   - Random Forest Regressor (best)
5. Basic hyperparameter tuning:
   - Ridge `alpha` via GridSearchCV
6. Deep Learning comparison:
   - MLP (Keras/TensorFlow) trained on the same split
7. Evaluate using:
   - MAE, RMSE, R²
8. Save evaluation summary to CSV

## Models and Metrics (Validation Results)
| Model | MAE | RMSE | R² |
|------|----:|-----:|---:|
| RandomForest (best) | 6.436 | 9.064 | 0.3097 |
| GradientBoosting | 6.561 | 9.305 | 0.2724 |
| Ridge (baseline/tuned) | ~6.778 | ~9.523 | ~0.238 |
| MLP (DL) | 25.172 | 32.955 | -8.125 |

---

# 03 — CNN Fish Image Classification

## Objective
Build an end-to-end image classification pipeline using CNN to classify fish images into **31 classes**.

## Workflow Summary
1. Load dataset from directory structure (`train/val/test`)
2. Data preprocessing:
   - Resize images to 224×224
   - Normalize pixel values
3. Handle class imbalance:
   - Compute and apply `class_weight`
4. CNN from scratch:
   - Custom CNN architecture + augmentation
   - Evaluate with accuracy and classification report
5. Transfer learning:
   - MobileNetV2 (frozen feature extractor)
   - Fine-tuning last layers (best model)
6. Evaluate final model:
   - Validation accuracy + macro/weighted F1
   - Test accuracy

## Key Results
- CNN Scratch: **val_acc ~ 0.7117**
- MobileNetV2 Frozen: **val_acc ~ 0.8920**
- MobileNetV2 Fine-tuned (best): **val_acc ~ 0.9149**, **test_acc ~ 0.9188**

---

## Repository Navigation
- `01_fraud_transaction.ipynb`  
  End-to-end fraud classification pipeline + submission generation.
- `submission_fraud_rf_baseline.csv`  
  Output prediction file for fraud task.
- `02_regression.ipynb`  
  End-to-end regression pipeline (ML + DL comparison).
- `regression_model_results.csv`  
  Summary of regression model metrics (MAE/RMSE/R²).
- `03_cnn_fish_classification.ipynb`  
  End-to-end CNN pipeline (scratch CNN + transfer learning + fine-tuning).
- `cnn_model_comparison.csv`  
  Summary comparison for CNN models (scratch vs transfer learning).

---

## Notes on Dataset
Raw datasets are **not uploaded to GitHub** due to size constraints.  
To run notebooks:
1. Place datasets in your Google Drive (or local folder)
2. Update the `DATA_DIR` / dataset path inside the notebook accordingly
3. Run all cells in sequence

Dataset : https://drive.google.com/drive/folders/1MtlXxKk82PMAOEzAS1-6oJm4wOFb71uK?usp=sharing

---

## Requirements
Main libraries used:
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
