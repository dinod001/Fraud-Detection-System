import os
import sys
import logging
import pandas as pd
from typing import Tuple

# -------------------------------------------------------------------
# Path Configuration (Ensuring imports work correctly)
# -------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import DataIngestorCSV
from src.handling_missing_and_duplicates import MissingAndDuplicateHandler
from src.feature_engineering import FeatureEngineeringHandler
from src.feature_splitting_and_scaling import FeatureSplittingAndScalingHandler
from utils.config import get_data_paths, get_column_config

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Main Data Pipeline Orchestrator
# -------------------------------------------------------------------
class DataPipeline:
    """
    Orchestrates the end-to-end data processing flow:
    Ingestion -> Cleaning -> Feature Engineering -> Splitting -> Scaling -> Saving
    """
    def __init__(self):
        self.data_paths = get_data_paths()
        self.columns = get_column_config()

    def run(self):
        """
        Executes the full data pipeline suite.
        """
        try:
            logger.info("="*60)
            logger.info("🚀 Starting the Fraud Detection Data Pipeline...")
            logger.info("="*60)

            # --- STEP 1: Data Ingestion ---
            logger.info("\n" + "-"*60)
            logger.info("📡 [Step 1/5] Ingesting raw datasets...")
            ingestor = DataIngestorCSV()
            df_raw = ingestor.ingest(self.data_paths['raw_fraud_data'])
            df_ip = ingestor.ingest(self.data_paths['raw_ip_country'])

            # --- STEP 2: Data Cleaning ---
            logger.info("\n" + "-"*60)
            logger.info("🧹 [Step 2/5] Handling missing values and duplicates...")
            cleaning_handler = MissingAndDuplicateHandler()
            df_cleaned = cleaning_handler.handle(df_raw)

            # --- STEP 3: Feature Engineering ---
            logger.info("\n" + "-"*60)
            logger.info("🛠️ [Step 3/5] Engineering domain-specific features...")
            
            # Ensure encodings directory exists
            encoding_path = self.data_paths.get("target_encoding_artifacts")
            if encoding_path:
                os.makedirs(os.path.dirname(encoding_path), exist_ok=True)

            fe_handler = FeatureEngineeringHandler(
                ip_df=df_ip,
                columns=self.columns.get("categorical_columns", []),
                drop_columns=self.columns.get("drop_columns", []),
                encoding_path=encoding_path,
                inference_mode=False  # Set to True only for production inference
            )
            df_features = fe_handler.handle(df_cleaned)

            # --- STEP 4: Splitting and Scaling ---
            logger.info("\n" + "-"*60)
            logger.info("✂️ [Step 4/5] Splitting data and applying Z-score scaling...")
            scaling_splitter = FeatureSplittingAndScalingHandler(
                target_column=self.columns.get("target", "class"),
                path=self.data_paths.get("scaler_artifacts"),
                features_to_scale=self.columns.get("features_to_scale")
            )
            X_train, y_train, X_test, y_test = scaling_splitter.handle(df_features)

            # --- STEP 5: Artifact Preservation ---
            logger.info("\n" + "-"*60)
            logger.info("💾 [Step 5/5] Saving processed artifacts to disk...")
            
            # Ensure directories exist
            os.makedirs(self.data_paths.get('artifacts_dir', 'artifacts'), exist_ok=True)
            
            X_train.to_csv(self.data_paths['X_train'], index=False)
            y_train.to_csv(self.data_paths['y_train'], index=False)
            X_test.to_csv(self.data_paths['X_test'], index=False)
            y_test.to_csv(self.data_paths['y_test'], index=False)

            logger.info("\n" + "="*60)
            logger.info("✨ Data Pipeline execution completed successfully! ✅")
            logger.info(f"📍 Artifacts available in: {self.data_paths['artifacts_dir']}")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"💥 Pipeline crashed during execution: {e}")
            raise e


# -------------------------------------------------------------------
# Execution Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()