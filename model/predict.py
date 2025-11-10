# model/predict.py

import numpy as np
import logging

# Optional torch import
try:
    import torch
    from model.pytorch_lstm import TradingLSTM, create_sample_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using fallback prediction method.")

logger = logging.getLogger(__name__)

def load_model(path="model/lstm_trained.pth"):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for model loading")
    
    model = TradingLSTM()
    if model.load_model(path):
        return model
    else:
        raise FileNotFoundError(f"Could not load model from {path}")

def predict(model, candles):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for prediction")
    
    # Use the TradingLSTM predict method
    prediction = model.predict(candles)
    return prediction

def preprocess_candles(candles):
    """
    Prepares candle data for prediction.
    IQ Option candle fields: ['open', 'max', 'min', 'close', 'volume', 'from']
    """
    if len(candles) < 50:
        raise ValueError(f"Insufficient candle data: {len(candles)} < 50")
    
    df = []
    for candle in candles:
        # Handle IQ Option candle format
        df.append([
            float(candle.get('open', 0)),
            float(candle.get('max', candle.get('high', 0))),  # IQ uses 'max' instead of 'high'
            float(candle.get('min', candle.get('low', 0))),   # IQ uses 'min' instead of 'low'
            float(candle.get('close', 0)),
            float(candle.get('volume', 0)),
            float(candle.get('from', candle.get('timestamp', 0)))  # IQ uses 'from' instead of 'timestamp'
        ])

    data = np.array(df, dtype=np.float32)
    
    # Check for invalid data
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Invalid candle data contains NaN or Inf values")

    # Normalize: you can also use min-max or a fitted scaler
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    normalized = (data - mean) / std

    return normalized[-50:]  # last 50 candles

def predict_signal(candles, model_path="model/lstm_trained.pth"):
    """
    Predict trading signal from candle data.
    Returns 'buy', 'sell', or None if prediction fails.
    """
    try:
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available, using fallback prediction")
            return predict_fallback(candles)
        
        import os
        if not os.path.exists(model_path):
            # Try alternative model paths
            alt_paths = [
                "model/checkpoints/lstm_model.pth",
                "models/lstm_trained.pth",
                "models/saved_lstm_model.pkl"
            ]
            
            model_found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    model_found = True
                    break
            
            if not model_found:
                logger.info("No ML model found, using fallback prediction")
                return predict_fallback(candles)
        
        model = load_model(model_path)
        output = predict(model, candles)  # Pass candles directly
        
        if output is None:
            logger.info("Model prediction failed, using fallback")
            return predict_fallback(candles)

        return "buy" if output > 0.5 else "sell"
        
    except Exception as e:
        print(f"ML prediction failed: {e}. Using fallback method.")
        return predict_fallback(candles)

def predict_fallback(candles):
    """
    Fallback prediction using simple technical analysis when ML model fails.
    """
    try:
        if len(candles) < 20:
            return None
            
        # Simple moving average crossover strategy
        closes = [float(c.get('close', 0)) for c in candles[-20:]]
        
        # Calculate short and long moving averages
        short_ma = sum(closes[-5:]) / 5  # 5-period MA
        long_ma = sum(closes[-10:]) / 10  # 10-period MA
        
        # Simple RSI calculation
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) > 0:
            avg_gain = sum(gains[-14:]) / min(14, len(gains))
            avg_loss = sum(losses[-14:]) / min(14, len(losses))
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Generate signal based on MA crossover and RSI
                if short_ma > long_ma and rsi < 70:
                    return "buy"
                elif short_ma < long_ma and rsi > 30:
                    return "sell"
        
        return None
        
    except Exception as e:
        print(f"Fallback prediction failed: {e}")
        return None
    except Exception as e:
        print(f"[predict.py] Prediction error: {e}")
        return None
