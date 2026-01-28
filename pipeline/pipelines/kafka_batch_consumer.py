"""
Simplified Kafka Consumer with ML Predictions
Processes customer events with real-time ML inference
"""

import json
import logging
import argparse
import os
import sys
import time
from typing import Dict, Any
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from confluent_kafka import KafkaError
from src.model_inference import ModelInference
from utils.config import get_data_paths, get_column_config, get_model_settings
from utils.kafka_utils import NativeKafkaConfig, NativeKafkaProducer, NativeKafkaConsumer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INPUT_TOPIC = "fraud_detection"
OUTPUT_TOPIC = "fraud_detection_scored"

data_paths = get_data_paths()
col_config = get_column_config()
model_config = get_model_settings('xgboost')

class MLKafkaConsumer:
    """Simplified ML Kafka Consumer"""
    
    def __init__(self):
        # Preprocessing class loads models internally on import/usage
        self.inferencer = ModelInference(
            model_path=model_config.get('model_path', 'artifacts/models/fraud_detection.joblib'),
            scaler_path=data_paths['scaler_artifacts'],
            encoding_path=data_paths['target_encoding_artifacts'],
            categorical_cols=col_config.get('categorical_columns', []),
            drop_cols=col_config.get('drop_columns', [])
        )
        self.data_paths = get_data_paths()
        self.df_ip = pd.read_csv(self.data_paths['raw_ip_country'])
        
    def initialize(self):
        """Initialize check"""
        try:
            # Check if artifacts exist as a sanity check
            encoders_dir = os.path.join(project_root, "artifacts", "encode")
            if os.path.exists(encoders_dir):
                logger.info("✅ ML resources found")
                return True
            else:
                logger.warning(f"⚠️ Encoders directory not found at {encoders_dir}")
                return True # Try anyway, let Preprocessing handle errors
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def extract_customer_data(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate customer data"""
        # Handle nested structure
        customer_data = message_data.get('data', message_data)
        
        # Required fields with defaults relative to the dataset
        return {
            'user_id': customer_data.get('user_id', 0),
            'signup_time': customer_data.get('signup_time', datetime.utcnow().isoformat()),
            'purchase_time': customer_data.get('purchase_time', datetime.utcnow().isoformat()),
            'purchase_value': float(customer_data.get('purchase_value', 0.0)),
            'device_id': customer_data.get('device_id', 'unknown'),
            'source': customer_data.get('source', 'Ads'),
            'browser': customer_data.get('browser', 'Chrome'),
            'sex': customer_data.get('sex', 'M'),
            'age': int(customer_data.get('age', 30)),
            'ip_address': float(customer_data.get('ip_address', 0.0))
        }


    
    def process_batch(self, max_messages: int = 1000, timeout: int = 10, 
                     group_id: str = None) -> int:
        """Process batch of messages with ML predictions"""
        
        # Configure consumer
        if group_id is None:
            group_id = f"batch_consumer_{int(time.time())}"
        
        kafka_config = NativeKafkaConfig()
        consumer_config = kafka_config.get_consumer_config(group_id)
        # For batch, we often want earliest
        if 'batch_' in group_id:
            consumer_config['auto.offset.reset'] = 'earliest'
        
        from confluent_kafka import Consumer
        consumer = Consumer(consumer_config)
        consumer.subscribe([INPUT_TOPIC])
        
        # Collect messages
        messages = []
        start_time = time.time()
        
        while len(messages) < max_messages and (time.time() - start_time) < timeout:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    break
                continue
            
            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                messages.append(message_data)
            except json.JSONDecodeError:
                continue
        
        consumer.close()
        
        if not messages:
            logger.warning("⚠️ No messages to process")
            return 0
        
        # Process with ML
        logger.info(f"📥 Processing {len(messages)} messages with ML")
        
        # Setup producer for results
        producer = NativeKafkaProducer()
        processed = 0
        
        print(f"\n📊 FRAUD DETECTIONS")
        print("=" * 70)
        print("Status | Customer   | Prediction | Probability")
        print("-" * 70)
        
        for i, message_data in enumerate(messages):
            user_id = 'N/A' # Default to handle errors
            try:
                # Extract customer data
                customer_data = self.extract_customer_data(message_data)
                user_id = customer_data.get('user_id', 'N/A')
                
                # Make prediction (ModelInference now expects a DataFrame based on your change)
                df_single = pd.DataFrame([customer_data])
                probs = self.inferencer.predict_proba(df_single, self.df_ip)
                
                # Get single prediction (assuming prob > 0.5 is fraud, or use threshold)
                p = float(probs[0])
                status = "Fraud" if p > 0.5 else "Legitimate"
                confidence = f"{p*100:.1f}%"
                
                # Display result
                pred_emoji = "🔴" if status == "Fraud" else "🟢"
                print(f"  {pred_emoji}   | {str(user_id)[:10]:10s} | {status:10s} | {confidence:10s}")

                
                # Send result
                result = {
                    'user_id': user_id,
                    'original_data': customer_data,
                    'prediction': probs,
                    'processed_at': datetime.now().isoformat(),
                    'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
                
                producer.send_message(
                    topic=OUTPUT_TOPIC,
                    key=str(user_id),
                    message=result
                )
                
                processed += 1
                
            except Exception as e:
                print(f"  ❌   | {str(user_id)[:10]:10s} | FAILED     | ERROR")
                logger.error(f"Error processing message {i}: {str(e)}")
        
        producer.flush()
        
        print("-" * 70)
        print(f"✅ Completed: {processed}/{len(messages)} predictions")
        print("=" * 70)
        
        logger.info(f"🎉 Processed {processed} messages successfully")
        return processed
    
    def run_continuous(self, poll_interval: int = 1, show_progress: bool = True):
        """Run true real-time stream processing"""
        logger.info("� Starting real-time ML stream processing")
        logger.info("🛑 Press Ctrl+C to stop")
        
        # Use helper classes from kafka_utils
        stream_consumer = NativeKafkaConsumer(
            group_id='realtime_ml_consumer_v1',
            topics=[INPUT_TOPIC]
        )
        
        producer = NativeKafkaProducer()
        total_processed = 0
        
        print("\n� REAL-TIME FRAUD DETECTION ACTIVE")
        print("="*75)
        print(f"{'TIMESTAMP':12s} | {'USER':10s} | {'RESULT':10s} | {'CONFIDENCE':10s}")
        print("-" * 75)
        
        try:
            while True:
                msg = stream_consumer.consumer.poll(timeout=1.0)
                
                if msg is None: continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF: continue
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
                
                try:
                    # 1. Process Message
                    message_data = json.loads(msg.value().decode('utf-8'))
                    customer_data = self.extract_customer_data(message_data)
                    user_id = customer_data.get('user_id', 'N/A')
                    
                    # 2. Predict
                    df_single = pd.DataFrame([customer_data])
                    probs = self.inferencer.predict_proba(df_single, self.df_ip)
                    score = float(probs[0])
                    
                    # 3. Format
                    status = "Fraud" if score > 0.5 else "Legitimate"
                    confidence = f"{score*100:.1f}%"
                    ts = datetime.now().strftime("%H:%M:%S")
                    
                    # 4. Display
                    emoji = "🔴" if status == "Fraud" else "🟢"
                    print(f"{ts:12s} | {str(user_id)[:10]:10s} | {status:10s} | {confidence:10s} {emoji}")
                    
                    # 5. Emit
                    result = {
                        'user_id': user_id,
                        'fraud_score': score,
                        'original_data': customer_data,
                        'processed_at': datetime.now().isoformat()
                    }
                    producer.send_message(OUTPUT_TOPIC, key=str(user_id), message=result)
                    
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
        except KeyboardInterrupt:
            logger.info(f"🛑 Stream processing stopped (Total: {total_processed})")
        finally:
            consumer.close()
            producer.flush()



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Kafka Consumer with ML Predictions")
    parser.add_argument('--max-messages', type=int, default=1000)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--poll-interval', type=int, default=3)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Kafka ML Consumer")
        
        consumer = MLKafkaConsumer()
        if not consumer.initialize():
            return 1
        
        if args.continuous:
            consumer.run_continuous(args.poll_interval, not args.quiet)
        else:
            processed = consumer.process_batch(args.max_messages, args.timeout)
            return 0 if processed > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Consumer failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())