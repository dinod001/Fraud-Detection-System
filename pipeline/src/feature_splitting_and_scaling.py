import logging
import pandas as pd
import joblib
from typing import List, Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Abstract Base Classes
# -------------------------------------------------------------------
class DataFrameHandler(ABC):
    """Abstract base class for all data transformation handlers"""
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass


class FeatureScalingStrategy(ABC):
    """Interface for different scaling methodologies"""
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


# -------------------------------------------------------------------
# Scaling Implementation
# -------------------------------------------------------------------
class StandardScalingStrategy(FeatureScalingStrategy):
    """
    Implements standard Z-score scaling (StandardScaler).
    Handles state management to prevent data leakage between train/test.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Applies scaling. Fits on the first call, transforms on subsequent calls.
        """
        df_scaled = df.copy()
        
        if not self.fitted:
            logger.info(f"Fitting and transforming with Standard scaling: {columns_to_scale}")
            df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
            self.fitted = True
        else:
            logger.info(f"Applying pre-fitted Standard scaling: {columns_to_scale}")
            df_scaled[columns_to_scale] = self.scaler.transform(df_scaled[columns_to_scale])
            
        return df_scaled
    
    def save_scaler(self, path: str):
        """Persists the fitted scaler object to disk"""
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler object saved successfully to: {path}")

    def load_scaler(self, path: str):
        """Loads a persisted scaler object from disk"""
        self.scaler = joblib.load(path)
        self.fitted = True
        logger.info(f"Scaler object loaded successfully from: {path}")
        return self.scaler


# -------------------------------------------------------------------
# Splitting and Scaling Handler
# -------------------------------------------------------------------
class FeatureSplittingAndScalingHandler(DataFrameHandler):
    """
    Orchestrates the splitting of data into Train/Test sets 
    and applies scaling safely to both.
    """
    def __init__(self, target_column: str = 'class', test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_strategy = StandardScalingStrategy()

    def handle(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main execution method for splitting and scaling.
        Returns: (X_train, y_train, X_test, y_test) all as DataFrames.
        """
        try:
            logger.info(f"Starting splitting process | target='{self.target_column}' | test_size={self.test_size}")
            
            # 1. Feature Target Separation
            X = df.drop(columns=[self.target_column])
            y = df[[self.target_column]]
            
            # 2. Stratified Train-Test Split
            # Using stratify=y to maintain fraud class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y
            )
            
            # 3. Scaling Pipeline
            # Identify columns to scale (all features in this case)
            scaling_columns = X_train.columns.tolist()
            
            # Step A: Fit & Transform Training Data
            X_train_scaled = self.scaling_strategy.scale(X_train, scaling_columns)
            
            # Step B: Transform Test Data (Uses parameters from Training Data)
            X_test_scaled = self.scaling_strategy.scale(X_test, scaling_columns)
            
            logger.info(
                f"Splitting & Scaling Success | "
                f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}"
            )
            
            return X_train_scaled, y_train, X_test_scaled, y_test
            
        except Exception as e:
            logger.exception("Data splitting and scaling handler failed")
            raise e