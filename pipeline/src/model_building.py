import joblib
import os
from xgboost import XGBClassifier
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# Abstract Base Model Builder
# -------------------------------------------------------------------
class BaseModelBuilder(ABC):
    """
    Interface for defining and building machine learning models.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
    
    @abstractmethod
    def build_model(self):
        """Builds and returns the model object."""
        pass

    def save_model(self, filepath):
        """Saves the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Loads a model from disk."""
        if not os.path.exists(filepath):
            raise ValueError(f"Can't load. File not found at: {filepath}")
        self.model = joblib.load(filepath)


# -------------------------------------------------------------------
# Random Forest Model Implementation
# -------------------------------------------------------------------
class RandomForestModelBuilder(BaseModelBuilder):
    """
    Builder for Scikit-Learn RandomForestClassifier.
    Optimized for fraud detection with balanced class weights.
    """
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 10,
            'n_estimators': 100, 
            'min_samples_split': 2, 
            'min_samples_leaf': 1, 
            'random_state': 42,
            'class_weight': 'balanced_subsample'
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)
    
    def build_model(self):
        self.model = RandomForestClassifier(**self.model_params)
        return self.model


# -------------------------------------------------------------------
# XGBoost Model Implementation
# -------------------------------------------------------------------
class XGboostModelBuilder(BaseModelBuilder):
    """
    Builder for XGBoost Classifier.
    Supports scale_pos_weight for handling data imbalance.
    """
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 8,
            'n_estimators': 100, 
            'random_state': 42,
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'scale_pos_weight': 1.0  # Crucial for imbalanced data
        }
        default_params.update(kwargs)
        super().__init__('XGboost', **default_params)

    def build_model(self):
        self.model = XGBClassifier(**self.model_params)
        return self.model