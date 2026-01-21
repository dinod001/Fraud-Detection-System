# Fraud Detection System

A high-performance machine learning project for detecting fraudulent transactions using advanced feature engineering and cost-sensitive learning.

## 🚀 Key Features

### 1. Advanced Feature Engineering
This project transforms raw transaction data into predictive signals using several high-impact feature sets:

*   **Temporal Features (Time-based)**:
    *   `account_age_minutes`: Time elapsed between user signup and purchase.
    *   `purchase_hour`: Identify high-risk hours (e.g., late night).
    *   `purchase_day_of_week`: Detect patterns across weekdays vs weekends.
    *   `is_weekend` & `is_night`: Binary flags for suspicious timing.

*   **Velocity Features (Frequency & Volume)**:
    *   `device_id_txn_count_1h` / `24h`: Number of transactions from the same device in short windows.
    *   `device_id_txn_amount_1h` / `24h`: Total spending volume per device.
    *   `ip_address_txn_count_1h` / `24h`: Frequency of transactions from the same IP address.
    *   **Goal**: Detect automated bots and "flash" fraud attempts.

*   **Amount-Based Features**:
    *   `purchase_value_log`: Log transformation to handle skewed transaction values.
    *   `purchase_value_zscore_global`: Amount normalized against the entire dataset.
    *   `is_outlier_amount`: Binary flag for transactions with a Z-score > 3 (extreme outliers).

*   **Behavioral & Rarity Features**:
    *   `device_id_txn_frequency`: Overall historical usage of a device.
    *   `is_device_new`: Flag for devices seen for the very first time.

*   **IP to Country Mapping**:
    *   `country`: Mapping `ip_address` to its origin country using the `IpAddress_to_Country.csv` lookup table.
    *   **Goal**: Capture geographical risk factors (some countries have higher historical fraud rates).

*   **Target Encoding (Smoothed K-Fold)**:
    *   `country_fraud_rate`: Historical fraud probability per country.
    *   `browser_fraud_rate`: Historical fraud probability per browser.
    *   `source_fraud_rate`: Historical fraud probability per marketing source (Ads, SEO, Direct).
    *   **Note**: Uses K-Fold smoothing to prevent data leakage.

### 2. Modeling Strategy
*   **Cost-Sensitive Learning**: Optimizing for extreme class imbalance by assigning higher weight to False Negatives (missed fraud).
*   **Threshold Optimization**: Dynamically calculating the decision threshold that minimizes the total expected business cost (Fraud Loss + False Alarm Cost).

## 📂 Project Structure
*   `notebooks/`: EDA and experimental feature engineering.
*   `notebooks/data/processed/`: Main datasets (`Fraud_Data.csv`).
*   `README.md`: Project overview and documentation.

## 🛠️ Data Cleanup & Dropped Columns
To ensure the model learns generalizable patterns and avoids overfitting to identifiers or raw categories, the following columns are dropped after feature engineering:

1.  **Identifiers**: `user_id`, `device_id`, `ip_address` (Replaced by rarity/velocity stats).
2.  **Timestamps**: `signup_time`, `purchase_time` (Replaced by temporal features).
3.  **Raw Categoricals**: `browser`, `source`, `country` (Replaced by target encoding fraud rates).
4.  **Low Importance**: `sex` (Often dropped to prevent bias and due to low predictive power).
