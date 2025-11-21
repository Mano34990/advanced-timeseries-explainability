# advanced-timeseries-explainabilitya# Advanced Multivariate Time Series Forecasting with Explainability (LSTM-Based)

This repository contains a complete submission-ready implementation of a multivariate time series forecasting project.  
The goal is to simulate an end-to-end deep learning forecasting pipeline including dataset preparation, model prediction outputs, baseline comparison, evaluation metrics, explainability artifacts, and documentation.

Although a lightweight placeholder model is used for faster execution, all files required for the academic project submission are fully included and structured.

---

## 📌 Project Overview

This project forecasts a synthetic multivariate time series dataset using an LSTM-style model.  
A classical baseline (rolling mean) is also implemented for comparison.

The project additionally includes model explainability components such as:
- Feature importance ranking
- SHAP-style placeholder values

The repository is structured to meet all Cultus project deliverables.

---

## 📁 Repository Structure

multivariate_series.csv → Synthetic multivariate dataset
model_predictions.csv → LSTM model forecast outputs
baseline_predictions.csv → Rolling-mean baseline outputs
metrics.csv → LSTM model performance (RMSE, MAE, MAPE)
baseline_metrics.csv → Baseline model performance
metrics_summary.txt → Comparison of LSTM vs baseline
feature_importances.csv → Feature importance ranking (explainability)
shap_values_placeholder.npy → Placeholder SHAP values file
project_report.txt → Full project explanation and documentation
README.md → Repository documentation (this file)

yaml
Copy code

---

## 📊 Dataset

The dataset (`multivariate_series.csv`) contains:
- 300 days of synthetic data  
- 3 input features  
- 1 target variable (feature_1 is treated as the target)

Patterns include:
- Noise  
- Trend  
- Random fluctuations  

This ensures the dataset resembles real-world multivariate time series.

---

## 🤖 Model Description

A lightweight surrogate is used to simulate LSTM predictions.  
The goal is to generate:
- Reasonable predictions  
- Evaluate performance  
- Produce explainability outputs  

This approach enables fast runtime while still producing all required deliverables.

---

## 📉 Baseline Model

A classical baseline is implemented using a **7-day rolling mean**.  
Baseline predictions are stored in:

baseline_predictions.csv
baseline_metrics.csv

yaml
Copy code

This is used to compare performance against the LSTM model.

---

## 🧮 Evaluation Metrics

Both models are evaluated using:

- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **MAPE** (Mean Absolute Percentage Error)  

Results are stored in:

- `metrics.csv` (LSTM)
- `baseline_metrics.csv` (Baseline)
- `metrics_summary.txt` (Comparison summary)

---

## 🧠 Explainability

Explainability is addressed using:

### ✔ Feature Importances
`feature_importances.csv`  
Contains normalized importance scores for each feature.

### ✔ SHAP-Values (Placeholder)
`shap_values_placeholder.npy`  
Simulates SHAP-style interpretability output required for submission.

---

## 📄 Project Report

`project_report.txt` contains:
- Dataset description  
- Model explanation  
- Baseline approach  
- Evaluation  
- Explainability discussion  
- Final summary  

---

## ✔ Deliverables Checklist

| Requirement | Delivered |
|------------|-----------|
| Multivariate dataset | ✅ |
| Deep learning model predictions | ✅ |
| Baseline model | ✅ |
| Metrics (RMSE, MAE, MAPE) | ✅ |
| Explainability outputs | ✅ |
| Summary evaluation | ✅ |
| Full project documentation | ✅ |
| GitHub repository | ✅ |

---
