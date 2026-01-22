import logging
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_building import XGboostModelBuilder, RandomForestModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_column_config, get_business_costs, get_model_settings

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    Orchestrates the model training and evaluation lifecycle.
    """
    def __init__(self):
        # Load configurations
        self.data_paths = get_data_paths()
        self.columns = get_column_config()
        self.costs = get_business_costs()
        self.settings = get_model_settings()

    def run(self):
        """
        Executes the full model pipeline flow.
        """
        try:
            logger.info("="*60)
            logger.info("🚀 Starting the Fraud Detection Model Pipeline...")
            logger.info("="*60)

            # --- STEP 1: Load Processed Data ---
            logger.info("\n" + "-"*60)
            logger.info("📂 [Step 1/4] Loading processed data artifacts...")
            X_train = pd.read_csv(self.data_paths['X_train'])
            y_train = pd.read_csv(self.data_paths['y_train']).values.ravel()
            X_test = pd.read_csv(self.data_paths['X_test'])
            y_test = pd.read_csv(self.data_paths['y_test']).values.ravel()
            
            # Calculate imbalance ratio for XGBoost
            fraud_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
            logger.info(f"📊 Dataset loaded | Fraud Ratio (Imbalance): {fraud_ratio:.2f}")

            # --- STEP 2: Model Building & Tuning ---
            logger.info("\n" + "-"*60)
            logger.info("🛠️ [Step 2/4] Building and tuning XGBoost model...")
            
            # Initialize Builder with imbalance ratio
            builder = XGboostModelBuilder(scale_pos_weight=fraud_ratio)
            xgb_base = builder.build_model()
            
            # Setup Trainer with GridSearchCV params from config
            trainer = ModelTrainer(
                param_grid=self.settings.get('param_grid'),
                cv=self.settings.get('cv_folds', 3), # Fallback to 3 if omitted
                scoring='f1'
            )
            
            # Perform Training/Tuning
            best_xgb, best_score = trainer.train(X_train, y_train, xgb_base)

            # --- STEP 3: Evaluation & Cost Optimization ---
            logger.info("\n" + "-"*60)
            logger.info("⚖️ [Step 3/4] Evaluating performance and optimizing cost...")
            evaluator = ModelEvaluator(
                model=best_xgb,
                model_name="XGBoost_Tuned",
                cost_fn=self.costs.get('false_negative_cost', 100.0),
                cost_fp=self.costs.get('false_positive_cost', 5.0)
            )
            
            # Run evaluation with threshold optimization
            results = evaluator.evaluate(X_test, y_test)
            
            # Plot and save confusion matrix
            plots_dir = os.path.join(self.data_paths.get('artifacts_dir', 'artifacts'), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            evaluator.plot_confusion_matrix(save_path=os.path.join(plots_dir, 'xgb_confusion_matrix.png'))

            # --- STEP 4: Model Preservation ---
            logger.info("\n" + "-"*60)
            logger.info("💾 [Step 4/4] Saving final model artifacts to disk...")
            
            model_save_path = os.path.join(self.data_paths.get('model_artifacts_dir'), 'xgb_final_model.joblib')
            trainer.save_model(best_xgb, model_save_path)

            logger.info("\n" + "="*60)
            logger.info("✨ Model Pipeline execution completed successfully! ✅")
            logger.info(f"📍 Final model saved to: {model_save_path}")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"💥 Model Pipeline crashed during execution: {e}")
            raise e


if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run()
