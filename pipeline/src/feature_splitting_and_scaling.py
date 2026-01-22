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
    """Abstract base class for all data transformation components."""
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass


class FeatureScalingStrategy(ABC):
    """Interface for modular scaling methodologies."""
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


# -------------------------------------------------------------------
# Scaling Implementation
# -------------------------------------------------------------------
class StandardScalingStrategy(FeatureScalingStrategy):
    """
    Implements Z-score normalization (StandardScaler).
    Handles state management to prevent leakage from test data into training parameters.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Applies scaling. Fits on the training call, transforms on subsequent calls.
        """
        df_scaled = df.copy()
        
        if not self.fitted:
            logger.info(f"⚖️ Fitting and applying Standard scaling to: {columns_to_scale}")
            df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
            self.fitted = True
        else:
            logger.info(f"⚡ Applying pre-fitted Standard scaling to: {columns_to_scale}")
            df_scaled[columns_to_scale] = self.scaler.transform(df_scaled[columns_to_scale])
            
        return df_scaled
    
    def save_scaler(self, path: str):
        """Persists the fitted scaler object for inference."""
        logger.info(f"💾 Saving fitted scaler to: {path}")
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Loads a persisted scaler from disk."""
        logger.info(f"📂 Loading scaler from: {path}")
        self.scaler = joblib.load(path)
        self.fitted = True
        return self.scaler


# -------------------------------------------------------------------
# Splitting and Scaling Handler
# -------------------------------------------------------------------
class FeatureSplittingAndScalingHandler(DataFrameHandler):
    """
    Orchestrates the division of data into Train/Test subsets 
    and applies leakage-free scaling to specified features.
    """
    def __init__(self, target_column: str = 'class', test_size: float = 0.2, random_state: int = 42, 
                 path: str = 'artifacts/scaler/scaler.joblib', features_to_scale: List[str] = None):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.path = path
        self.features_to_scale = features_to_scale
        self.scaling_strategy = StandardScalingStrategy()

    def handle(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes stratified splitting and configuration-driven scaling.
        Returns: (X_train, y_train, X_test, y_test) as pandas DataFrames.
        """
        try:
            logger.info(f"✂️  Starting stratified split | test_size={self.test_size} | target='{self.target_column}'")
            
            # 1. Feature Target Separation
            X = df.drop(columns=[self.target_column])
            y = df[[self.target_column]]
            
            # 2. Stratified Train-Test Split (Preserves fraud class ratio)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y
            )
            
            # 3. Scaling Pipeline (Driven by Config)
            # If features_to_scale isn't provided, safe-default to all features
            scaling_columns = self.features_to_scale if self.features_to_scale else X_train.columns.tolist()
            
            # Step A: Fit & Transform Training Data
            X_train_scaled = self.scaling_strategy.scale(X_train, scaling_columns)
            
            # Step B: Transform Test Data using learned parameters
            X_test_scaled = self.scaling_strategy.scale(X_test, scaling_columns)

            # 4. Save Scaler for Future Inference
            self.scaling_strategy.save_scaler(path=self.path)
            
            logger.info(
                f"✅ Splitting & Scaling successful | "
                f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}"
            )
            
            return X_train_scaled, y_train, X_test_scaled, y_test
            
        except Exception as e:
            logger.error(f"❌ Data splitting/scaling failure: {e}")
            raise e