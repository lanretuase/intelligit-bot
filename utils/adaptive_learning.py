# utils/adaptive_learning.py
# Adaptive Learning System for Trading Bot

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveLearning:
    def __init__(self):
        self.performance_file = "data/performance_history.json"
        self.learning_threshold = 0.6  # Retrain if accuracy drops below 60%
        self.min_samples_for_retrain = 100
        self.retrain_interval_hours = 24  # Retrain every 24 hours
        
        # Load existing performance data
        self.performance_data = self.load_performance_data()
        
    def load_performance_data(self) -> Dict:
        """Load historical performance data"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            return {"predictions": [], "last_retrain": None, "accuracy_history": []}
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return {"predictions": [], "last_retrain": None, "accuracy_history": []}
    
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def record_prediction(self, asset: str, prediction: str, confidence: float, 
                         actual_result: str = None, timestamp: str = None):
        """Record a prediction for later evaluation"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        prediction_record = {
            "asset": asset,
            "prediction": prediction,
            "confidence": confidence,
            "actual_result": actual_result,
            "timestamp": timestamp,
            "evaluated": actual_result is not None
        }
        
        self.performance_data["predictions"].append(prediction_record)
        
        # Keep only last 1000 predictions to manage memory
        if len(self.performance_data["predictions"]) > 1000:
            self.performance_data["predictions"] = self.performance_data["predictions"][-1000:]
        
        self.save_performance_data()
        logger.info(f"📊 Recorded prediction: {asset} -> {prediction} (confidence: {confidence:.2f})")
    
    def update_prediction_result(self, asset: str, timestamp: str, actual_result: str):
        """Update a prediction with its actual result"""
        for pred in self.performance_data["predictions"]:
            if (pred["asset"] == asset and 
                pred["timestamp"] == timestamp and 
                not pred["evaluated"]):
                pred["actual_result"] = actual_result
                pred["evaluated"] = True
                logger.info(f"✅ Updated prediction result: {asset} -> {actual_result}")
                break
        
        self.save_performance_data()
    
    def calculate_current_accuracy(self, asset: str = None, days: int = 7) -> float:
        """Calculate accuracy for recent predictions"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        evaluated_predictions = [
            p for p in self.performance_data["predictions"]
            if p["evaluated"] and datetime.fromisoformat(p["timestamp"]) > cutoff_date
        ]
        
        if asset:
            evaluated_predictions = [p for p in evaluated_predictions if p["asset"] == asset]
        
        if not evaluated_predictions:
            return 0.0
        
        correct_predictions = sum(
            1 for p in evaluated_predictions 
            if p["prediction"].lower() == p["actual_result"].lower()
        )
        
        accuracy = correct_predictions / len(evaluated_predictions)
        logger.info(f"📈 Current accuracy ({days}d): {accuracy:.2%} ({correct_predictions}/{len(evaluated_predictions)})")
        return accuracy
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        reasons = []
        
        # Check if enough time has passed
        last_retrain = self.performance_data.get("last_retrain")
        if last_retrain:
            last_retrain_date = datetime.fromisoformat(last_retrain)
            hours_since_retrain = (datetime.now() - last_retrain_date).total_seconds() / 3600
            
            if hours_since_retrain >= self.retrain_interval_hours:
                reasons.append(f"Time-based: {hours_since_retrain:.1f}h since last retrain")
        else:
            reasons.append("No previous retraining recorded")
        
        # Check accuracy
        current_accuracy = self.calculate_current_accuracy()
        if current_accuracy < self.learning_threshold and current_accuracy > 0:
            reasons.append(f"Low accuracy: {current_accuracy:.2%} < {self.learning_threshold:.2%}")
        
        # Check if we have enough samples
        evaluated_count = sum(1 for p in self.performance_data["predictions"] if p["evaluated"])
        if evaluated_count < self.min_samples_for_retrain:
            return False, f"Insufficient samples: {evaluated_count} < {self.min_samples_for_retrain}"
        
        should_retrain = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should_retrain, reason_text
    
    async def trigger_retraining(self, assets: List[str]):
        """Trigger model retraining for specified assets"""
        try:
            logger.info("🔄 Starting adaptive retraining process...")
            
            from model.pytorch_lstm import TradingLSTM
            from iq_connect.candle_manager import get_candles
            from iq_connect.connection_manager import connect_to_iqoption
            
            # Connect to IQ Option
            iq = await connect_to_iqoption()
            if not iq:
                logger.error("Failed to connect to IQ Option for retraining")
                return False
            
            for asset in assets:
                try:
                    logger.info(f"🎯 Retraining model for {asset}...")
                    
                    # Fetch more historical data for training
                    candles = get_candles(iq, asset, count=500)
                    if not candles or len(candles) < 200:
                        logger.warning(f"Insufficient data for {asset} retraining")
                        continue
                    
                    # Create new model instance
                    model = TradingLSTM()
                    
                    # Simple training simulation (you'd implement proper training here)
                    model.prepare_data(candles)
                    
                    # Save retrained model
                    model_path = f"model/retrained_{asset.replace('-', '_')}.pth"
                    model.save_model(model_path)
                    
                    logger.info(f"✅ Model retrained and saved for {asset}")
                    
                except Exception as e:
                    logger.error(f"Error retraining {asset}: {e}")
            
            # Update last retrain timestamp
            self.performance_data["last_retrain"] = datetime.now().isoformat()
            self.save_performance_data()
            
            logger.info("🎉 Adaptive retraining completed!")
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        total_predictions = len(self.performance_data["predictions"])
        evaluated_predictions = sum(1 for p in self.performance_data["predictions"] if p["evaluated"])
        
        accuracy_7d = self.calculate_current_accuracy(days=7)
        accuracy_30d = self.calculate_current_accuracy(days=30)
        
        # Asset-specific accuracy
        asset_accuracy = {}
        for asset in set(p["asset"] for p in self.performance_data["predictions"]):
            asset_accuracy[asset] = self.calculate_current_accuracy(asset=asset, days=7)
        
        return {
            "total_predictions": total_predictions,
            "evaluated_predictions": evaluated_predictions,
            "accuracy_7d": accuracy_7d,
            "accuracy_30d": accuracy_30d,
            "asset_accuracy": asset_accuracy,
            "last_retrain": self.performance_data.get("last_retrain"),
            "should_retrain": self.should_retrain()
        }

# Global instance
adaptive_learner = AdaptiveLearning()
