import pandas as pd
import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# -------------------------------------------------------------------
# Configuration and Warnings
# -------------------------------------------------------------------
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Model Evaluator Implementation
# -------------------------------------------------------------------
class ModelEvaluator:
    """
    Evaluates model performance using standard and cost-sensitive metrics.
    """
    def __init__(self, model, model_name, cost_fn=100.0, cost_fp=5.0):
        self.model = model
        self.model_name = model_name
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.evaluation_results = {}

    def evaluate(self, X_test, y_test, threshold=None):
        """
        Calculates a suite of performance metrics at a specific or optimized threshold.
        """
        try:
            logger.info(f"📊 Running evaluation suite for: {self.model_name}")
            
            # Predict probabilities (REQUIRED for threshold optimization)
            y_probs = self.model.predict_proba(X_test)[:, 1]
            
            # Apply threshold
            if threshold is None:
                logger.info("🎯 No threshold provided. Optimizing based on business cost...")
                threshold, min_cost = self.optimize_threshold(y_test, y_probs)
                logger.info(f"✨ Optimal Threshold found: {threshold:.4f} | Min Cost: ${min_cost:,.2f}")
            else:
                logger.info(f"📍 Using provided threshold: {threshold:.4f}")
            
            y_pred = (y_probs >= threshold).astype(int)

            # Standard Metrics
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_probs)

            # Store result
            self.evaluation_results = {
                'model_name': self.model_name,
                'threshold': threshold,
                'cm': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'business_cost': self._calculate_cost(y_test, y_pred)
            }

            self._log_summary()
            return self.evaluation_results

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise e

    def optimize_threshold(self, y_true, y_probs):
        """
        Finds the threshold that minimizes the business cost function.
        """
        thresholds = np.linspace(0, 1, 101)
        costs = []
        for t in thresholds:
            preds = (y_probs >= t).astype(int)
            costs.append(self._calculate_cost(y_true, preds))
        
        best_idx = np.argmin(costs)
        return thresholds[best_idx], costs[best_idx]

    def _calculate_cost(self, y_true, y_pred):
        """Calculates total business cost: (FN * cost_fn) + (FP * cost_fp)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (fn * self.cost_fn) + (fp * self.cost_fp)

    def _log_summary(self):
        """Logs a clean summary of the results."""
        res = self.evaluation_results
        logger.info("="*60)
        logger.info(f"🏆 PERFORMANCE SUMMARY: {res['model_name']}")
        logger.info(f"🎯 Threshold: {res['threshold']:.3f} | ROC AUC: {res['roc_auc']:.4f}")
        logger.info(f"📈 F1-Score:  {res['f1']:.4f} | Accuracy: {res['accuracy']:.4f}")
        logger.info(f"⚖️  Precision: {res['precision']:.4f} | Recall:   {res['recall']:.4f}")
        logger.info(f"💰 Total Cost: ${res['business_cost']:,.2f}")
        logger.info("="*60)

    def plot_confusion_matrix(self, save_path=None):
        """Generates and optionally saves a confusion matrix plot."""
        cm = self.evaluation_results.get('cm')
        if cm is None:
            logger.warning("⚠️ No evaluation results found to plot.")
            return

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"🖼️ Confusion matrix plot saved to: {save_path}")
        plt.show()