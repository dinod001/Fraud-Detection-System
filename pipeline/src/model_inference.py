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
    def __init__(self) -> None:
        pass
    def __init__(self, model_path: str = None, scaler_path: str = None, encoding_path: str = None, 
                 categorical_cols: list = None, drop_cols: list = None):
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

            # Get model prediction
            return self.model.predict_proba(X_scaled)[:, 1]
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            raise e
