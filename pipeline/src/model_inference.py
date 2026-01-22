import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys

# Add relevant paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from feature_engineering import FeatureEngineeringHandler

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('feature_engineering').setLevel(logging.WARNING)


class ModelInference:
    """
    Handles end-to-end inference: Preprocessing, Scaling, and Prediction.
    """
    def __init__(self, model_path: str, scaler_path: str, encoding_path: str, 
                 categorical_cols: list, drop_cols: list):
        """
        Loads all required artifacts for inference.
        """
        try:
            logger.info("📂 Loading inference artifacts...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoding_path = encoding_path
            self.categorical_cols = categorical_cols
            self.drop_cols = drop_cols
            logger.info("✅ All artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load inference artifacts: {e}")
            raise e

    def predict_proba(self, df: pd.DataFrame, ip_df: pd.DataFrame) -> np.ndarray:
        """
        Processes raw data and returns fraud probabilities.
        """
        try:
            # Re-use preprocessing logic
            fe_handler = FeatureEngineeringHandler(
                ip_df=ip_df,
                columns=self.categorical_cols,
                drop_columns=self.drop_cols,
                encoding_path=self.encoding_path,
                inference_mode=True
            )
            df_processed = fe_handler.handle(df)
            
            # CRITICAL: Ensure columns match the training set exactly in name and ORDER
            if hasattr(self.scaler, 'feature_names_in_'):
                expected_cols = self.scaler.feature_names_in_
                df_processed = df_processed[expected_cols]
            
            X_scaled = self.scaler.transform(df_processed)
            
            # ========== COMPREHENSIVE FRAUD DETECTION RULES ==========
            # Multi-layered approach to catch patterns the model might miss
            
            fraud_alerts = []
            
            if 'account_age_minutes' in df_processed.columns and 'is_device_new' in df_processed.columns:
                account_age = df_processed['account_age_minutes'].iloc[0]
                is_new_device = df_processed['is_device_new'].iloc[0]
                purchase_value = df_processed['purchase_value'].iloc[0]
                
                # RULE 1: High-Velocity Fraud (Original)
                # Instant purchase from new device with significant value
                if (account_age < 0.033) and (is_new_device == 1) and (purchase_value > 10):
                    fraud_alerts.append(f"VELOCITY: age={account_age:.4f}min, new_device=True, value=${purchase_value:.2f}")
                
                # RULE 2: Medium-Velocity Fraud
                # Slightly slower but still suspicious (3-10 seconds)
                if (0.033 <= account_age < 0.167) and (is_new_device == 1) and (purchase_value > 50):
                    fraud_alerts.append(f"MEDIUM-VELOCITY: age={account_age:.4f}min, new_device=True, high_value=${purchase_value:.2f}")
                
                # RULE 3: High-Value Anomaly
                # Unusually high purchase for account age
                if (account_age < 1.0) and (purchase_value > 200):
                    fraud_alerts.append(f"HIGH-VALUE-ANOMALY: age={account_age:.4f}min, extreme_value=${purchase_value:.2f}")
                
                # RULE 4: Small-Value Rapid Pattern (Carding)
                # Low value but instant - potential credit card testing
                if (account_age < 0.05) and (is_new_device == 1) and (5 < purchase_value <= 10):
                    fraud_alerts.append(f"CARDING-PATTERN: age={account_age:.4f}min, test_value=${purchase_value:.2f}")
                
                # RULE 5: Outlier Detection Override
                # Z-score based outlier combined with new device
                if 'is_outlier_amount' in df_processed.columns:
                    is_outlier = df_processed['is_outlier_amount'].iloc[0]
                    if (is_outlier == 1) and (is_new_device == 1) and (account_age < 5.0):
                        fraud_alerts.append(f"OUTLIER-OVERRIDE: statistical_outlier=True, new_device=True, age={account_age:.4f}min")
            
            # Get model prediction
            model_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            # Override with rules if any triggered
            if fraud_alerts:
                for alert in fraud_alerts:
                    logger.warning(f"🚨 FRAUD ALERT: {alert}")
                # Force high probability to ensure detection
                return np.array([0.95])  # 95% fraud probability
            
            return model_proba
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            raise e
