# E-commerce Fraud Detection System

A high-performance machine learning pipeline designed to detect fraudulent transactions in an e-commerce platform. This project focuses on **cost-sensitive learning**, **threshold optimization**, and **MLOps best practices** to minimize business losses associated with fraud.

## 🚀 Project Overview
Fraud detection is inherently difficult due to severe class imbalance. This project has evolved from exploratory research into a production-ready system using:
- **Modular Pipeline Design**: Decoupling ingestion, engineering, and training into reusable Python scripts.
- **Cost-Sensitive Learning**: Penalizing missed frauds (False Negatives) more heavily than false alarms (False Positives).
- **Threshold Optimization**: Finding the "Sweet Spot" based on a financial cost function:
  - **Missed Fraud (FN)**: $100 Loss per instance.
  - **False Alarm (FP)**: $5 Investigation cost per instance.

## 🏗️ Pipeline Architecture
The system is built as a series of specialized modules to ensure scalability and maintainability:
- **Data Ingestion**: Standardized handlers for raw data loading and initial validation.
- **Automated Feature Engineering**: Systematic creation of Velocity, Temporal, and Rarity features with persistence for inference.
- **Model Training Suite**: Automated hyperparameter tuning (GridSearchCV) with class-weight management.
- **Inference Engine**: A standalone module designed for real-time scoring of individual transactions.

## 🛠️ Feature Engineering
We implemented a specialized `FeatureEngineer` class to transform raw data into highly predictive features:
- **Velocity Features**: Transaction frequency and amount within 1h/24h rolling windows for Devices and IP Addresses.
- **Temporal Features**: Account age (signup-to-purchase time), transaction hour, day of week, and "night-owl" flags.
- **Rarity Features**: Mapping device/IP frequency ranks to identify "first-seen" or rare entities.
- **Target Encoding**: High-cardinality categorical data (Country, Browser, Source) converted using smoothed fraud rates.

## 📉 Experiment Tracking with MLflow
Integrated **MLflow** to manage the entire machine learning lifecycle, ensuring transparency and reproducibility:
- **Parameter Logging**: Tracking every hyperparameter, data path, and feature configuration.
- **Metric Tracking**: Real-time monitoring of Accuracy, AUC, F1, and total business cost across different runs.
- **Artifact Management**: Versioning model binaries (.joblib), scalers, and confusion matrix plots.
- **Model Registry**: Centralized repository for managing production-ready model versions.

## 📊 Final Performance (XGBoost)
The final tuned model balances the ability to catch fraud with the cost of manual review.

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **ROC AUC** | **0.8336** | High ability to distinguish fraud from normal transactions. |
| **Recall** | **69.12%** | Catches ~70% of all fraudulent attempts. |
| **Precision** | **51.76%** | Low false alarm rate (minimizing investigator fatigue). |
| **Accuracy** | **91.08%** | Robust overall classification performance. |
| **F1-Score** | **0.5919** | Strong balance between precision and recall. |

## 🚀 Future Roadmap
The system is continuously evolving toward a full-scale enterprise MLOps stack:
- [ ] **Apache Kafka**: Implement real-time streaming ingestion for sub-second detection.
- [ ] **PySpark**: Integration for handling massive data volumes (Big Data processing).
- [ ] **Apache Airflow**: End-to-end pipeline orchestration and automated retraining triggers.
- [ ] **Feature Store**: Centralized repository for sharing engineered features across training and inference.

## 📁 Project Structure
- `pipeline/pipelines/`: Entry points for data and training pipelines.
- `pipeline/src/`: Core logic for ingestion, engineering, and evaluation.
- `pipeline/utils/`: Configuration management and MLflow utilities.
- `artifacts/`: Saved models, scalers, and performance plots.

## 🏁 Conclusion
By shifting from standard classification to **Business-Cost Optimization** and implementing a structured **MLOps workflow**, this system provides a practical, industry-standard tool for protecting e-commerce revenue.
