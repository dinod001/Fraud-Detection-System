import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
pd.set_option('display.max_columns', None)
tqdm.pandas()

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract base handler
# -------------------------------------------------------------------
class DataFrameHandler(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# -------------------------------------------------------------------
# Feature Engineering Utilities
# -------------------------------------------------------------------
class FeatureEngineer:
    """Collection of feature engineering utilities"""

    # ---------------------------------------------------------------
    # Create time-based behavioral features
    # ---------------------------------------------------------------
    # - Measures account age at purchase time
    # - Extracts hour and weekday from transaction timestamp
    # - Flags night-time and weekend transactions
    # ---------------------------------------------------------------
    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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

    # ---------------------------------------------------------------
    # Map IP addresses to country using IP ranges
    # ---------------------------------------------------------------
    # - Converts IP to numeric format
    # - Performs fast range-based lookup using merge_asof
    # - Ensures IP falls within upper/lower bounds
    # ---------------------------------------------------------------
    @staticmethod
    def map_ip_to_country(df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
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

    # ---------------------------------------------------------------
    # Create velocity and device-level behavior features
    # ---------------------------------------------------------------
    # - Counts total transactions per device
    # - Flags first-time usage of a device
    # - Helps identify device reuse across accounts
    # ---------------------------------------------------------------
    @staticmethod
    def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('purchase_time')

        df['device_txn_count_total'] = (
            df.groupby('device_id')['user_id'].transform('count')
        )

        df['is_device_new'] = (
            df.groupby('device_id')['purchase_time'].transform('rank') == 1
        ).astype(int)

        return df

    # ---------------------------------------------------------------
    # Create transaction amount anomaly features
    # ---------------------------------------------------------------
    # - Log-transform purchase value for normalization
    # - Computes global Z-score
    # - Flags extreme transaction amounts
    # ---------------------------------------------------------------
    @staticmethod
    def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # ---------------------------------------------------------------
    # Target encoding with K-Fold leakage prevention
    # ---------------------------------------------------------------
    # - Encodes categorical feature using fraud rate
    # - Uses K-Fold strategy to avoid target leakage
    # - Applies smoothing for rare categories
    # ---------------------------------------------------------------
    @staticmethod
    def target_encode_kfold(
        df: pd.DataFrame,
        col: str,
        target: str = 'class',
        n_folds: int = 5,
        smoothing: int = 10
    ) -> pd.DataFrame:
        df = df.copy()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        encoded_col = f"{col}_fraud_rate"
        df[encoded_col] = 0.0

        global_mean = df[target].mean()

        for train_idx, val_idx in kf.split(df):
            train_data = df.iloc[train_idx]

            agg = train_data.groupby(col)[target].agg(['count', 'mean'])
            smooth = (
                agg['count'] * agg['mean'] + smoothing * global_mean
            ) / (agg['count'] + smoothing)

            df.loc[df.index[val_idx], encoded_col] = (
                df.loc[df.index[val_idx], col].map(smooth)
            )

        df[encoded_col] = df[encoded_col].fillna(global_mean)
        return df

    # ---------------------------------------------------------------
    # Final cleanup before model training
    # ---------------------------------------------------------------
    # - Removes identifiers and high-cardinality columns
    # - Drops raw timestamps and PII-like fields
    # ---------------------------------------------------------------
    @staticmethod
    def cleanup(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [
            'user_id', 'signup_time', 'purchase_time', 'device_id',
            'ip_address', 'ip_address_int',
            'lower_bound_ip_address', 'upper_bound_ip_address',
            'browser', 'source', 'country', 'sex'
        ]
        return df.drop(columns=drop_cols, errors='ignore')


# -------------------------------------------------------------------
# Feature Engineering Pipeline Handler
# -------------------------------------------------------------------
class FeatureEngineeringHandler(DataFrameHandler):
    def __init__(self, ip_df: pd.DataFrame):
        self.ip_df = ip_df
        self.fe = FeatureEngineer()

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            rows_before = len(df)
            logger.info(f"Starting feature engineering | rows={rows_before}")

            df = self.fe.create_temporal_features(df)
            df = self.fe.map_ip_to_country(df, self.ip_df)
            df = self.fe.create_velocity_features(df)
            df = self.fe.create_amount_features(df)
            
            # Target encode key categories
            for col in ['browser', 'source', 'country']:
                df = self.fe.target_encode_kfold(df, col=col)
                
            df = self.fe.cleanup(df)

            # Fix: .shape is a tuple, not a callable
            rows_after, columns_after = df.shape
            logger.info(
                f"Feature engineering completed | rows={rows_after} | columns={columns_after} "
                f"removed={rows_before - rows_after}"
            )

            return df

        except Exception as e:
            logger.exception("Feature engineering failed")
            raise e
