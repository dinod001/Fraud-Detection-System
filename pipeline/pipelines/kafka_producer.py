import os
import sys
import json
import time
import random
import logging
import argparse
from confluent_kafka import Producer
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.kafka_utils import NativeKafkaProducer, validate_native_setup, create_topic
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEventGenerator:
    def __init__(self,seed:int = 42):
        data_path = os.path.join(project_root, 'data/raw/Fraud_Data.csv')
        self.dataset = pd.read_csv(data_path)
        self.dataset.dropna()

        if 'class' in self.dataset.columns:
            self.features = self.dataset.drop('class',axis=1)
            self.labels = self.dataset['class']
        else:
            self.features = self.dataset.copy()
            self.labels = None
    
    def generate_event(self) -> Dict[str, Any]:
        """Generate single customer event"""
        idx = random.randint(0, len(self.features) - 1)
        row = self.features.iloc[idx]

        event = {}
        for col, value in row.items():
            if pd.isna(value):
                event[col] = None 
            elif isinstance(value, (np.integer, np.int64)):
                event[col] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                event[col] = float(value)
            else:
                event[col] = str(value)
            
        event.update({
                    'event_timestamp': datetime.utcnow().isoformat(),
                    'event_id':f"evt_{idx}_{int(time.time())}",
                    'true_class_label': str(self.labels.iloc[idx]) if self.labels is not None else None 
                    })
        return event
    
    def generate_batch(self, num_events: int) -> List[Dict[str, Any]]:
        """Generate batch of events"""
        return [self.generate_event() for _ in range(num_events)]

class MLKafkaProducer:
    def __init__(self, enable_logging: bool = True):
        validation = validate_native_setup()
        if not validation['setup_valid']:
            raise RuntimeError("Kafka Setup is Invalid ...")

        self.producer = NativeKafkaProducer()
        self.generator = CustomEventGenerator()
        self.enable_logging = enable_logging
    
    def _log_event(self, event: Dict[str, Any], success: bool, count: int):
        """Log event if logging enabled"""
        if not self.enable_logging:
            return
            
        status = "✅" if success else "❌"
        user_id = str(event.get('user_id', 'N/A'))[:8]
        source = str(event.get('source', 'N/A'))
        browser = str(event.get('browser', 'N/A'))
        
        print(f"{status} Event {count:3d}: Customer {user_id} | {source} | {browser}")
    
    def setup_topic(self) -> bool:
        """Setup all required ML topics"""
        from utils.kafka_utils import setup_ml_topics
        return setup_ml_topics()
    
    def produce_batch(self, topic: str = 'fraud_detection', num_events: int = 100) -> int:
        """Produce batch of events"""
        events = self.generator.generate_batch(num_events)
        successful = 0

        for i, event in enumerate(events):
            success = self.producer.send_message(
                                                topic=topic,
                                                message=event,
                                                key=str(event['user_id'])
                                                )

            if success:
                successful += 1

            self._log_event(event, success, i+1)

        if self.enable_logging:
            print(f"Batch completed: {successful}/{num_events} events sent")
        
        return successful
    
    def produce_stream(self, topic: str = 'fraud_detection', 
                      rate: int = 1, duration: int = 300) -> int:
        """Produce streaming events""" # For Micro Batches
        
        start_time = time.time()
        total_events = 0
        successful = 0

        try:
            while time.time() - start_time < duration:
                batch_start = time.time()

                for _ in range(rate):
                    event = self.generator.generate_event()
    
                    success = self.producer.send_message(
                                                topic=topic,
                                                message=event,
                                                key=str(event['user_id'])   
                                                )

                    total_events += 1 
                    if success:
                        successful += 1

                    self._log_event(event, success, total_events)

                sleep_time = max(0, 1 - (time.time() - batch_start))
                if sleep_time > 0: 
                    time.sleep(sleep_time)
    
            if self.enable_logging:
                print(f"Streaming completed: {successful}/{total_events} events sent")
            
            return successful

        except KeyboardInterrupt:
            logger.info("Streaming stopped by CodeSageLK")
            return successful

    def close(self):
        """Close producer"""
        self.producer.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Kafka Producer for ML Pipeline")
    parser.add_argument('--mode', choices=['streaming', 'batch'], default='streaming')
    parser.add_argument('--topic', default='churn_predictions')
    parser.add_argument('--rate', type=int, default=1, help='Events per second')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--num-events', type=int, default=100, help='Number of events')
    parser.add_argument('--setup-topics', action='store_true', help='Setup topics (falls through to produce)')
    parser.add_argument('--only-setup', action='store_true', help='Setup topics and exit')
    parser.add_argument('--list-topics', action='store_true', help='List all topics and exit')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='Disable event logging')
    
    args = parser.parse_args()
    
    if args.validate:
        validation = validate_native_setup()
        if not validation['setup_valid']:
            logger.error("❌ Kafka Setup is Invalid!")
            return 1
        else:
            logger.info("✅ Kafka Installation Check Passed")
            # If ONLY validating, return here
            if not any([args.setup_topics, args.only_setup, args.list_topics]):
                return 0

    # List topics and exit
    if args.list_topics:
        from utils.kafka_utils import list_topics
        topics = list_topics()
        print("\n📝 KAFKA TOPICS:")
        for t in topics: print(f"  - {t}")
        return 0

    # Only instantiate producer if we actually need it
    try:
        producer = MLKafkaProducer(enable_logging=not args.quiet)
    except Exception as e:
        if args.validate or args.only_setup or args.setup_topics:
            logger.warning(f"⚠️ Broker not running: {e}")
            return 0 if not (args.only_setup or args.setup_topics) else 1
        raise

    if args.setup_topics or args.only_setup:
        if producer.setup_topic():
            logger.info("Topic Setup is Completed ...")
        else:
            logger.error("Topic Setup is Failed ...")
            return 1
        if args.only_setup: 
            producer.close()
            return 0

    if args.mode == 'streaming':
        producer.produce_stream(args.topic, args.rate, args.duration)
    else: 
        producer.produce_batch(args.topic, args.num_events)

    producer.close()




if __name__ == "__main__":
    exit(main())

