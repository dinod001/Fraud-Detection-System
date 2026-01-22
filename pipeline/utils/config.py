import os
import yaml
import logging
from typing import Dict, Any, List

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Locate the config file relative to this script
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        if not os.path.exists(CONFIG_FILE):
            logger.error(f"Configuration file not found at: {CONFIG_FILE}")
            return {}
            
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error reading configuration: {e}")
        return {}


def get_data_paths() -> Dict[str, str]:
    """Returns the configured data and artifact paths."""
    return load_config().get('data_paths', {})


def get_column_config() -> Dict[str, Any]:
    """Returns feature inclusion/exclusion and target configuration."""
    return load_config().get('columns', {})


def get_business_costs() -> Dict[str, int]:
    """Returns the financial cost weights for FN and FP."""
    return load_config().get('business_cost', {
        'false_negative_cost': 100,
        'false_positive_cost': 5
    })


def get_training_config() -> Dict[str, Any]:
    """Returns general training parameters (CV folds, test size, etc.)."""
    return load_config().get('training', {})


def get_model_settings(model_name: str = 'xgboost') -> Dict[str, Any]:
    """Returns best hyperparameters and thresholds for a specific model."""
    models_config = load_config().get('models', {})
    return models_config.get(model_name, {})


def get_mlflow_config() -> Dict[str, Any]:
    """Returns MLFlow tracking parameters."""
    return load_config().get('mlflow', {})


def get_logging_config() -> Dict[str, Any]:
    """Returns logging level and format settings."""
    return load_config().get('logging', {})


def update_config_value(path: str, value: Any) -> None:
    """
    Updates a specific key in the YAML file using a dot-notated path 
    (e.g., 'models.xgboost.optimal_threshold').
    """
    config = load_config()
    keys = path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
        
    current[keys[-1]] = value
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Updated config key '{path}' to {value}")