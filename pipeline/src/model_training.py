import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Model Trainer Implementation
# -------------------------------------------------------------------
class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and persistence.
    """
    def __init__(self, param_grid=None, cv=5, scoring='f1', n_jobs=-1):
        """
        Initializes the trainer with tuning parameters.
        Default scoring is set to 'f1' for fraud detection.
        """
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def train(self, X_train, y_train, model):
        """
        Trains the provided model, optionally performing GridSearchCV if param_grid is set.
        """
        try:
            if self.param_grid:
                logger.info("="*60)
                logger.info("🔍 Starting GridSearchCV for hyperparameter optimization...")
                logger.info(f"📊 CV Folds: {self.cv} | Metric: {self.scoring}")
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=self.param_grid,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

                logger.info(f"✨ Best Parameters: {grid_search.best_params_}")
                logger.info(f"✅ Best CV Score ({self.scoring}): {best_score:.4f}")
                logger.info("="*60)
                
                return best_model, best_score
            else:
                logger.info("🚀 No param_grid provided. Training model directly...")
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                logger.info(f"✅ Training complete | Score: {train_score:.4f}")
                return model, train_score
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise e

    def save_model(self, model, filepath):
        """Saves a model object to a joblib file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model, filepath)
            logger.info(f"💾 Model artifact saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise e

    def load_model(self, filepath):
        """Loads a model artifact from disk."""
        if not os.path.exists(filepath):
            logger.error(f"❌ Model file not found at: {filepath}")
            raise FileNotFoundError(f"Can't load. File not found at: {filepath}")
        
        logger.info(f"📂 Loading model from: {filepath}")
        return joblib.load(filepath)