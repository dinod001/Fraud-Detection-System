import os
import sys
import pandas as pd
import logging
from typing import Dict, Any

# Configure logging - Clean output mode
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(message)s'  # Simple format without timestamps
)
logger = logging.getLogger(__name__)

# Suppress feature engineering logs completely
logging.getLogger('feature_engineering').setLevel(logging.ERROR)
logging.getLogger('model_inference').setLevel(logging.WARNING)

# Add source paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from model_inference import ModelInference
from config import get_data_paths, get_column_config, get_model_settings

def run_inference_demo():
    """
    Simulates a real-time fraud detection request using the inference pipeline.
    """
    try:
        # 1. Load Configurations
        data_paths = get_data_paths()
        col_config = get_column_config()
        model_config = get_model_settings('xgboost')

        # 2. Test Scenarios
        scenarios = [
            {
                'name': 'HISTORICAL SAFE (User 22058)',
                'data': {
                    'user_id': 22058,
                    'signup_time': '2015-02-24 22:55:49',
                    'purchase_time': '2015-04-18 02:47:11',
                    'purchase_value': 34,
                    'device_id': 'QVPSPJUOCKZAR',
                    'source': 'SEO',
                    'browser': 'Chrome',
                    'sex': 'M',
                    'age': 39,
                    'ip_address': 732758368.0
                }
            },
            {
                'name': 'HISTORICAL FRAUD (User 1359 - 1s Velocity)',
                'data': {
                    'user_id': 1359,
                    'signup_time': '2015-01-01 18:52:44',
                    'purchase_time': '2015-01-01 18:52:45',
                    'purchase_value': 15,
                    'device_id': 'YSSKYOSNTPJWH',
                    'source': 'SEO',
                    'browser': 'Opera',
                    'sex': 'M',
                    'age': 53,
                    'ip_address': 2621473856.0
                }
            },
            {
                'name': 'HIGH-VALUE INSTANT FRAUD ($500 + 1s)',
                'data': {
                    'user_id': 9999,
                    'signup_time': '2015-03-15 14:30:00',
                    'purchase_time': '2015-03-15 14:30:01',
                    'purchase_value': 500,
                    'device_id': 'FRAUD_BOT_001',
                    'source': 'Ads',
                    'browser': 'Chrome',
                    'sex': 'F',
                    'age': 28,
                    'ip_address': 1039507456.0  # Sri Lanka
                }
            },
            {
                'name': 'EDGE CASE: Just Below Threshold (1.9s)',
                'data': {
                    'user_id': 8888,
                    'signup_time': '2015-04-10 10:00:00',
                    'purchase_time': '2015-04-10 10:00:02',  # 2 seconds - should NOT trigger
                    'purchase_value': 50,
                    'device_id': 'EDGE_DEVICE_01',
                    'source': 'SEO',
                    'browser': 'FireFox',
                    'sex': 'M',
                    'age': 35,
                    'ip_address': 3503114240.0
                }
            },
            {
                'name': 'LOW-VALUE INSTANT (Should NOT trigger - $5)',
                'data': {
                    'user_id': 7777,
                    'signup_time': '2015-05-01 08:00:00',
                    'purchase_time': '2015-05-01 08:00:01',  # 1 second
                    'purchase_value': 5,  # Below $10 threshold
                    'device_id': 'SAFE_SMALL_BUY',
                    'source': 'Direct',
                    'browser': 'Safari',
                    'sex': 'F',
                    'age': 42,
                    'ip_address': 123456789.0
                }
            },
            {
                'name': 'NORMAL SAFE TRANSACTION (2 weeks old)',
                'data': {
                    'user_id': 6666,
                    'signup_time': '2015-01-01 10:00:00',
                    'purchase_time': '2015-01-15 14:30:00',
                    'purchase_value': 75,
                    'device_id': 'SAFE_CUSTOMER',
                    'source': 'SEO',
                    'browser': 'Chrome',
                    'sex': 'M',
                    'age': 45,
                    'ip_address': 987654321.0
                }
            }
        ]

        # Load IP-to-Country data for mapping
        ip_df = pd.read_csv(data_paths['raw_ip_country'])

        # 3. Initialize Inference Engine
        inference_engine = ModelInference(
            model_path=model_config.get('model_path', 'artifacts/models/fraud_detection.joblib'),
            scaler_path=data_paths['scaler_artifacts'],
            encoding_path=data_paths['target_encoding_artifacts'],
            categorical_cols=col_config.get('categorical_columns', []),
            drop_cols=col_config.get('drop_columns', [])
        )


        print("\n" + "="*70)
        print("🚀 FRAUD DETECTION SYSTEM - TEST SCENARIOS")
        print("="*70 + "\n")

        for scenario in scenarios:
            name = scenario["name"]
            raw_data = pd.DataFrame([scenario["data"]])
            
            # Execute Prediction
            fraud_prob = inference_engine.predict_proba(raw_data, ip_df)[0]
            
            # Get decision based on optimal threshold
            threshold = model_config.get('optimal_threshold', 0.5)
            is_fraud = fraud_prob >= threshold

            # Output Result
            print(f"📌 {name}")
            print(f"   💰 Amount: ${scenario['data']['purchase_value']} | Age: {scenario['data']['age']}")
            print(f"   📊 Probability: {fraud_prob:.2%} | Threshold: {threshold:.2%}")
            
            if is_fraud:
                print(f"   🚨 RESULT: FRAUD DETECTED! 🛑")
            else:
                print(f"   ✅ RESULT: SAFE TRANSACTION 🟢")
            print()

        print("="*70)
        print("✅ Test complete - All scenarios processed successfully")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"❌ Inference pipeline demo failed: {e}")

if __name__ == "__main__":
    run_inference_demo()