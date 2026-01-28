import os
import sys
import logging
from pathlib import Path
from typing import Dict,Any,Optional

# =========================
# Configure logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Project Environment Setup
# =========================

def setup_project_environment() -> str:
    """
    Setup the project environment by adding key project directories to sys.path
    and setting PYTHONPATH. Returns the absolute path to the project root.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    paths_to_add = [
        str(project_root),
        str(project_root / 'src'),
        str(project_root / 'utils'),
        str(project_root / 'pipelines')
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.environ['PYTHONPATH'] = ':'.join(paths_to_add)
    logger.info(f"Project environment configured. PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    return str(project_root)

# =========================
# Input Data Validation
# =========================
def validate_input_data(data_path: str='data/raw') -> Dict[str,Any]:
    """
    Validate raw input data existence and file size.
    Returns a summary dict indicating success or warnings.
    """
    project_root = setup_project_environment()
    full_path = Path(project_root) / data_path
    logger.info(f"Validating input data at: {full_path}")

    if not full_path.exists():
        logger.warning(f"Input data file not found: {full_path}")
        return {
            'status': 'warning',
            'message': 'Input data file not found',
            'file_path': str(full_path)
        }
    
    file_size = full_path.stat().st_size
    if file_size == 0:
        logger.warning(f"Input data file is empty: {full_path}")
        return {
            'status': 'warning',
            'message': 'Input data file is empty',
            'file_path': str(full_path)
        }
    
    logger.info(f"✅ Input data validation passed: {file_size} bytes")
    return {
        'status': 'success',
        'file_path': str(full_path),
        'file_size_bytes': file_size,
        'message': 'Input data file exists and has content'
    }

# =========================
# Processed Data Validation
# =========================
def validate_processed_data(data_path: str = 'artifacts/data/X_train.csv') -> Dict[str,Any]:
    """
    Validate processed data existence and file size.
    """
    project_root = setup_project_environment()
    full_path = Path(project_root) / data_path

    logger.info(f"Validating processed data at: {full_path}")

    if not full_path.exists():
        logger.warning(f"Processed data file not found: {full_path}")
        return {
            'status': 'warning',
            'message': 'Processed data file not found',
            'file_path': str(full_path)
        }

    file_size = full_path.stat().st_size
    if file_size == 0:
        logger.warning(f"Processed data file is empty: {full_path}")
        return {
            'status': 'warning',
            'message': 'Processed data file is empty',
            'file_path': str(full_path)
        }

    logger.info(f"✅ Processed data validation passed: {file_size} bytes")
    return {
        'status': 'success',
        'file_path': str(full_path),
        'file_size_bytes': file_size,
        'message': 'Processed data file exists and has content'
    }

# =========================
# Model Validation
# =========================
def validate_trained_model(model_path: str = 'artifacts/models') -> Dict[str,Any]:
    """
    Validate that trained model files exist.
    """
    project_root = setup_project_environment()
    model_dir = Path(project_root) / model_path

    logger.info(f"Validating trained model at: {model_dir}")

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return {
            'status': 'warning',
            'message': 'Model directory not found. Run training pipeline first.',
            'model_directory': str(model_dir)
        }

    model_files = list(model_dir.glob('**/*'))
    if not model_files:
        logger.warning(f"No model files found in: {model_dir}")
        return {
            'status': 'warning',
            'message': 'No model files found. Run training pipeline first.',
            'model_directory': str(model_dir)
        }

    logger.info(f"✅ Model validation passed: {len(model_files)} file(s) found")
    return {
        'status': 'success',
        'model_directory': str(model_dir),
        'model_files_count': len(model_files),
        'message': 'Model files found'
    }

# =========================
# Data Pipeline Execution
# =========================
def run_data_pipeline(data_path: str = "data/raw/Fraud_Data.csv") -> Dict[str, Any]:
    """
    Run the data pipeline for preprocessing and return summary info.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from data_pipeline import DataPipeline

        logger.info(f"Starting data pipeline: {data_path}")
        pipeline = DataPipeline()
        result = pipeline.run()

        logger.info("✓ Data pipeline completed successfully")

        X_train = result.get("X_train") if isinstance(result, dict) else None
        X_test  = result.get("X_test") if isinstance(result, dict) else None
        Y_train = result.get("Y_train") if isinstance(result, dict) else None
        Y_test  = result.get("Y_test") if isinstance(result, dict) else None

        return {
            "status": "success",
            "X_train_shape": X_train.shape if X_train is not None else None,
            "X_test_shape": X_test.shape if X_test is not None else None,
            "Y_train_shape": Y_train.shape if Y_train is not None else None,
            "Y_test_shape": Y_test.shape if Y_test is not None else None,
            "message": "Data pipeline completed successfully",
        }

    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        raise

# =========================
# Training Pipeline Execution
# =========================
def run_training_pipeline(
        model_path: str = 'artifacts/models/fraud_detection.joblib'
    ) -> Dict[str, Any]:
    """
    Run model training pipeline and save trained model.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from training_pipeline import ModelPipeline

        pipeline = ModelPipeline()
        pipeline.run()

        logger.info("✓ Training pipeline completed successfully")
        return {
            'status': 'success',
            'model_path': model_path,
            'message': 'Training pipeline completed successfully'
        }
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {str(e)}")
        raise

# =========================
# Inference Pipeline Execution
# =========================
def run_inference_pipeline(sample_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run inference on sample data using trained model.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from inference_pipeline import run_inference_demo

        logger.info("Starting inference pipeline")
        run_inference_demo()
        logger.info("✓ Inference pipeline completed successfully")
        return {
            'status': 'success',
            'message': 'Inference pipeline completed successfully'
        }
    except Exception as e:
        logger.error(f"✗ Inference pipeline failed: {str(e)}")
        raise
