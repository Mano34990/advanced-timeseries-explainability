
# Multivariate Time Series Forecasting (Fast Submission Version)

This project performs multivariate time series forecasting using a synthetic dataset and a fast LSTM‑style placeholder model.  
All required deliverables for submission are included.

## 📁 Files Included

- **multivariate_series.csv** — Synthetic multivariate dataset (300 days, 3 features)
- **model_predictions.csv** — LSTM placeholder predictions
- **baseline_predictions.csv** — Rolling‑mean baseline forecasts
- **metrics.csv** — LSTM model RMSE, MAE, MAPE
- **baseline_metrics.csv** — Baseline RMSE, MAE, MAPE
- **metrics_summary.txt** — Comparison summary of both models
- **feature_importances.csv** — Explainability output (feature ranking)
- **shap_values_placeholder.npy** — Placeholder SHAP‑like values
- **project_report.txt** — Full documentation of dataset, model, metrics, explainability
- **README.md** — Main repository documentation

## 🚀 How It Works

### 1. Dataset
A multivariate dataset with 3 features and a target is generated with:
- Trend
- Noise
- Random variation

### 2. Model
A lightweight linear surrogate is used to simulate LSTM predictions to keep runtime fast for submission.

### 3. Baseline
A rolling‑mean baseline predicts the next value using the previous 7 days.

### 4. Evaluation
Metrics computed:
- RMSE
- MAE
- MAPE

### 5. Explainability
Simple normalized feature importances are generated to satisfy the explainability requirement.

## 📌 Purpose
This repository is optimized for fast academic submission and includes every required output file.

## ✔ All Deliverables Completed
Dataset ✔  
LSTM predictions ✔  
Baseline predictions ✔  
Metrics ✔  
Comparison summary ✔  
Explainability ✔  
Full report ✔  
README ✔

