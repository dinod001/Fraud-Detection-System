import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract Base Handler
# -------------------------------------------------------------------
class DataFrameHandler(ABC):
    """
    Interface for all data transformation components.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# -------------------------------------------------------------------
# Feature Engineering Utilities
# -------------------------------------------------------------------
class FeatureEngineer:
    """
    Collection of methods for creating domain-specific fraud features.
    Now supports persistence for target encoding mappings.
    """

    def __init__(self):
        # Stores category -> fraud_rate mappings for inference
        self.encoder_mappings: Dict[str, Dict[str, float]] = {}
        # Stores global mean fraud rate as a fallback for new categories
        self.global_mean: float = 0.0

    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time-based features: account age, purchase hour, weekday, weekend/night flags.
        """
        logger.info("🕒 Creating temporal behavioral features...")
        df = df.copy()
        
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        df['account_age_minutes'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds() / 60

        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek

        df['is_weekend'] = (df['purchase_day_of_week'] >= 5).astype(int)
        df['is_night'] = (
            (df['purchase_hour'] >= 22) | (df['purchase_hour'] <= 6)
        ).astype(int)

        return df

    @staticmethod
    def map_ip_to_country(df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps IP addresses to countries using range-based lookup.
        """
        logger.info("🌍 Mapping IP addresses to country origins...")
        df = df.copy()
        ip_df = ip_df.sort_values('lower_bound_ip_address')

        df['ip_address_int'] = df['ip_address'].astype(float)

        df_merged = pd.merge_asof(
            df.sort_values('ip_address_int'),
            ip_df,
            left_on='ip_address_int',
            right_on='lower_bound_ip_address'
        )

        df_merged['country'] = np.where(
            df_merged['ip_address_int'] <= df_merged['upper_bound_ip_address'],
            df_merged['country'],
            'Unknown'
        )

        return df_merged

    @staticmethod
    def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates velocity features based on device interaction patterns.
        """
        logger.info("🚀 Calculating device transaction velocity and age...")
        df = df.sort_values('purchase_time')

        df['device_txn_count_total'] = (
            df.groupby('device_id')['user_id'].transform('count')
        )

        df['is_device_new'] = (
            df.groupby('device_id')['purchase_time'].transform('rank') == 1
        ).astype(int)

        return df

    @staticmethod
    def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates financial features: log-transforms and outlier detection for purchase value.
        """
        logger.info("💰 Normalizing and analyzing transaction amounts...")
        df['purchase_value_log'] = np.log1p(df['purchase_value'])

        mean_val = df['purchase_value'].mean()
        std_val = df['purchase_value'].std()

        df['purchase_value_zscore'] = (
            (df['purchase_value'] - mean_val) / std_val
        )

        df['is_outlier_amount'] = (
            df['purchase_value_zscore'].abs() > 3
        ).astype(int)

        return df

    def target_encode_kfold(
        self,
        df: pd.DataFrame,
        col: str,
        target: str = 'class',
        n_folds: int = 5,
        smoothing: int = 10
    ) -> pd.DataFrame:
        """
        Performs target encoding with K-Fold during training.
        Also calculates global mappings for inference preservation.
        """
        logger.info(f"🏷️ [TRAIN] Target encoding categorical feature: {col}")
        df = df.copy()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        encoded_col = f"{col}_fraud_rate"
        df[encoded_col] = 0.0

        # Store global stats for this column for inference mappings
        self.global_mean = df[target].mean()
        
        # 1. K-Fold encoding (for training consistency/leakage prevention)
        for train_idx, val_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            agg = train_data.groupby(col)[target].agg(['count', 'mean'])
            smooth = (
                agg['count'] * agg['mean'] + smoothing * self.global_mean
            ) / (agg['count'] + smoothing)

            df.loc[df.index[val_idx], encoded_col] = (
                df.loc[df.index[val_idx], col].map(smooth)
            )

        # 2. Store final global mapping for this column (used in inference)
        agg_all = df.groupby(col)[target].agg(['count', 'mean'])
        smooth_all = (
            agg_all['count'] * agg_all['mean'] + smoothing * self.global_mean
        ) / (agg_all['count'] + smoothing)
        
        self.encoder_mappings[col] = smooth_all.to_dict()

        df[encoded_col] = df[encoded_col].fillna(self.global_mean)
        return df

    def apply_inference_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Applies learned encoding mappings during inference (no target column needed).
        """
        logger.info(f"🏷️ [INFERENCE] Applying target encoding mapping: {col}")
        df = df.copy()
        encoded_col = f"{col}_fraud_rate"
        
        mapping = self.encoder_mappings.get(col, {})
        
        # Map values, fill unknown categories with global mean
        df[encoded_col] = df[col].map(mapping).fillna(self.global_mean)
        
        return df

    def save_encodings(self, path: str):
        """
        Persists mappings and global mean to disk.
        """
        logger.info(f"💾 Saving target encoding mappings to: {path}")
        payload = {
            'mappings': self.encoder_mappings,
            'global_mean': self.global_mean
        }
        joblib.dump(payload, path)

    def load_encodings(self, path: str):
        """
        Loads mappings and global mean from disk.
        """
        logger.info(f"📂 Loading target encoding mappings from: {path}")
        payload = joblib.load(path)
        self.encoder_mappings = payload.get('mappings', {})
        self.global_mean = payload.get('global_mean', 0.0)

    @staticmethod
    def cleanup(df: pd.DataFrame, drop_columns: List[str]) -> pd.DataFrame:
        """
        Removes raw, high-cardinality, or identifier columns.
        """
        logger.info("🧹 Removing raw identifiers and high-cardinality metadata...")
        return df.drop(columns=drop_columns, errors='ignore')


# -------------------------------------------------------------------
# Feature Engineering Pipeline Handler
# -------------------------------------------------------------------
class FeatureEngineeringHandler(DataFrameHandler):
    """
    Orchestrates the full feature engineering suite.
    """
    def __init__(self, ip_df: pd.DataFrame, columns: List[str], drop_columns: List[str], 
                 encoding_path: str = None, inference_mode: bool = False):
        self.ip_df = ip_df
        self.columns = columns
        self.drop_columns = drop_columns
        self.encoding_path = encoding_path
        self.inference_mode = inference_mode
        self.fe = FeatureEngineer()
        
        # If in inference mode, load pre-saved mappings
        if self.inference_mode and self.encoding_path:
            self.fe.load_encodings(self.encoding_path)

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes all feature engineering steps in the prescribed order.
        """
        try:
            rows_before = len(df)
            mode_str = "INFERENCE" if self.inference_mode else "TRAINING"
            logger.info(f"🛠️ Starting feature engineering suite [{mode_str}] | Initial rows: {rows_before}")

            # Execute sequence
            df = self.fe.create_temporal_features(df)
            df = self.fe.map_ip_to_country(df, self.ip_df)
            df = self.fe.create_velocity_features(df)
            df = self.fe.create_amount_features(df)
            
            # Application of Target Encoding
            for col in self.columns:
                if self.inference_mode:
                    df = self.fe.apply_inference_encoding(df, col=col)
                else:
                    df = self.fe.target_encode_kfold(df, col=col)
            
            # Save mappings after training
            if not self.inference_mode and self.encoding_path:
                self.fe.save_encodings(self.encoding_path)
                
            # Dynamic Cleanup based on config
            df = self.fe.cleanup(df, self.drop_columns)

            rows_after, columns_after = df.shape
            logger.info(
                f"✅ Feature engineering complete | Final Shape: ({rows_after}, {columns_after})"
            )

            return df

        except Exception as e:
            logger.error(f"❌ Feature engineering pipeline failed: {e}")
            raise e
