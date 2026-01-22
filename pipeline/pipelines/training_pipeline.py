import os
import sys
import logging
import pandas as pd
from data_pipeline import DataPipeline
from typing import Dict, Any, Tuple, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import XGboostModelBuilder, RandomForestModelBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_settings, get_data_paths
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
                    model_params: Optional[Dict[str, Any]] = None,
                    model_path: str = 'artifacts/models/fraud_detection.joblib',
                    ):
    if (not os.path.exists(get_data_paths()['X_train'])) or \
        (not os.path.exists(get_data_paths()['X_test'])) or \
        (not os.path.exists(get_data_paths()['y_train'])) or \
        (not os.path.exists(get_data_paths()['y_test'])):
        
        DataPipeline()
    else:
        print("Loading Data Artifacts from Data Pipeline.")
    
    X_train = pd.read_csv(get_data_paths()['X_train'])
    X_test = pd.read_csv(get_data_paths()['X_test'])
    Y_train = pd.read_csv(get_data_paths()['y_train'])
    Y_test = pd.read_csv(get_data_paths()['y_test'])

    ratio = (len(Y_train['class']) - sum(Y_train['class'])) / sum(Y_train['class'])

    model_builder =XGboostModelBuilder(scale_pos_weight=ratio)
    model = model_builder.build_model()

    trainer = ModelTrainer(param_grid=model_params)
    model,train_score = trainer.train(
                        model=model,
                        X_train = X_train,
                        Y_train = Y_train
                    )
    trainer.save_model(model,model_path)
    
    evaluator = ModelEvaluator(model,"XGboost")
    evaluation_results = evaluator.evaluate(X_test,Y_test)

    evaluation_results_cp = evaluation_results.copy()
    del evaluation_results_cp['cm']

    print(evaluation_results)

if __name__ == '__main__':
    model_config = get_model_settings()
    model_params = model_config.get('params_grid')
    model_path = model_config.get('model_path')
    training_pipeline(model_params=model_params,model_path=model_path)