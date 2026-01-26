# E-commerce Fraud Detection System

A high-performance machine learning pipeline designed to detect fraudulent transactions in an e-commerce platform. This project focuses on **cost-sensitive learning** and **threshold optimization** to minimize business losses associated with fraud.

## 🚀 Project Overview
Fraud detection is inherently difficult due to severe class imbalance (frauds are rare). This project implements a robust solution using advanced technical strategies:
- **Cost-Sensitive Learning**: penalizing missed frauds (False Negatives) more heavily than false alarms (False Positives).
- **Custom Business Logic**: Optimizing decision thresholds based on a financial cost function:
  - **Missed Fraud (FN)**: $100 Loss
  - **False Alarm (FP)**: $5 Investigation Cost

## 🛠️ Feature Engineering
We implemented a specialized `FeatureEngineer` class to transform raw data into highly predictive features:
- **Velocity Features**: Transaction frequency and amount within 1h/24h rolling windows for Devices and IP Addresses.
- **Temporal Features**: Account age (signup-to-purchase time), transaction hour, day of week, and "night-owl" flags.
- **Rarity Features**: Mapping device/IP frequency ranks to identify "first-seen" or rare entities often linked to bot activity.
- **Cross-Geography**: Mapping IP addresses to countries to detect cross-border anomalies.

## 📊 Model Training & Optimization
We trained multiple models with class imbalance handling:
- **XGBoost**: Utilized `scale_pos_weight` to handle the ~1:10 imbalance ratio.
- **Random Forest**: Utilized `class_weight='balanced_subsample'` for robust ensemble learning.

### Threshold Optimization
Instead of using the default 0.5 threshold, we systematically searched for the "Sweet Spot" that minimizes the total cost to the business.
- **Baseline XGBoost Optimal Threshold**: 0.32
- **Tuned XGBoost Optimal Threshold**: 0.38

## 📈 Final Performance (XGBoost)
The final tuned model achieved exceptional results, balancing the ability to catch fraud with the cost of manual review.

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **ROC AUC** | **0.8336** | High ability to distinguish fraud from normal transactions. |
| **Recall** | **69.12%** | Catches ~70% of all fraudulent attempts. |
| **Precision** | **51.76%** | Over half of all flags are confirmed frauds (low false alarm rate). |
| **Accuracy** | **91.08%** | Robust overall performance. |
| **F1-Score** | **0.5919** | Strong balance between precision and recall. |


## 📁 Project Structure
- `artifacts/`: Contains saved models, preprocessors, and scaled datasets.
- `notebooks/`: Original research and development notebooks.
- `README.md`: Project documentation.

## 🏁 Conclusion
By shifting the focus from standard accuracy to **Business-Cost Optimization**, this system provides a practical tool for e-commerce security, effectively protecting revenue while maintaining operational efficiency.
