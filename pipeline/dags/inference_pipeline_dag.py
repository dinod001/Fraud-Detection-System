import sys
from airflow import DAG
from airflow.utils import timezone
from datetime import timedelta
from airflow.operators.python import PythonOperator


import os
import sys

# Dynamically find the Pipeline directory that contains correct 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check 2 levels up (Standard structure: Pipeline/dags/dag.py -> Pipeline)
target_dir = os.path.dirname(current_dir)
if not os.path.exists(os.path.join(target_dir, 'utils')):
    # Check 3 levels up (.airflow structure: Pipeline/.airflow/dags/dag.py -> Pipeline)
    target_dir = os.path.dirname(target_dir)

sys.path.insert(0, target_dir)
from utils.airflow_tasks import validate_trained_model, run_inference_pipeline

default_arguments = {
                    'owner' : 'dinod',
                    'depends_on_past' : False,
                    'start_date': timezone.datetime(2025, 1, 1),
                    'email_on_failure': False,
                    'email_on_retry': False,
                    'retries': 0,
                    }

with DAG(
        dag_id = 'inference_pipeline_dag',
        default_args = default_arguments,
        schedule_interval='*/5 * * * *',
        catchup = False,
        max_active_runs=1,
        description='Inference Pipeline - Every 5 Minutes Scheduled',
        tags=['pandas', 'mlflow', 'batch-processing']
        ) as dag:
    
    #step 1
    validate_trained_model_task = PythonOperator(
        task_id='validate_trained_model',
        python_callable=validate_trained_model,
        execution_timeout=timedelta(minutes=2)
    )

    # step 2
    run_inference_pipeline_task = PythonOperator(
        task_id='run_inference_pipeline',
        python_callable=run_inference_pipeline,
        execution_timeout=timedelta(minutes=15)
    )

    validate_trained_model_task >> run_inference_pipeline_task