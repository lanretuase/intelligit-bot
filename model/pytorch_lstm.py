# model/pytorch_lstm.py
# PyTorch LSTM Model for Trading Signal Prediction

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for binary trading signal prediction
    """
    def __init__(self, input_size=6, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out

class TradingLSTM:
    """
    Wrapper class for LSTM model with training and prediction capabilities
    """
    def __init__(self, input_size=6, hidden_size=50, num_layers=2, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = None
        
        logger.info(f"LSTM model initialized on {self.device}")
    
    def prepare_data(self, candles, sequence_length=50):
        """
        Prepare candle data for LSTM training/prediction
        """
        try:
            # Extract features from candles
            features = []
            for candle in candles:
                features.append([
                    float(candle.get('open', 0)),
                    float(candle.get('max', candle.get('high', 0))),
                    float(candle.get('min', candle.get('low', 0))),
                    float(candle.get('close', 0)),
                    float(candle.get('volume', 0)),
                    float(candle.get('from', candle.get('timestamp', 0)))
                ])
            
            data = np.array(features, dtype=np.float32)
            
            # Normalize data
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                normalized_data = self.scaler.fit_transform(data)
            else:
                normalized_data = self.scaler.transform(data)
            
            # Create sequences
            if len(normalized_data) < sequence_length:
                # Pad with zeros if not enough data
                padding = np.zeros((sequence_length - len(normalized_data), data.shape[1]))
                normalized_data = np.vstack([padding, normalized_data])
            
            # Take last sequence_length candles
            sequence = normalized_data[-sequence_length:]
            
            return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def predict(self, candles):
        """
        Predict trading signal from candle data
        Returns probability (0-1), where >0.5 suggests 'buy', <0.5 suggests 'sell'
        """
        try:
            self.model.eval()
            
            # Prepare input data
            input_tensor = self.prepare_data(candles)
            if input_tensor is None:
                return None
            
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                probability = prediction.item()
            
            logger.info(f"LSTM prediction probability: {probability:.4f}")
            return probability
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def save_model(self, path="model/lstm_trained.pth"):
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path="model/lstm_trained.pth"):
        """Load a trained model"""
        try:
            # Fix for PyTorch 2.6+ security issue
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint.get('scaler', None)
            
            self.model.eval()
            logger.info(f"✅ Model loaded successfully from {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

def create_sample_model():
    """
    Create and save a sample trained model for immediate use
    """
    try:
        logger.info("Creating sample LSTM model...")
        
        # Create model
        lstm = TradingLSTM()
        
        # Generate some dummy training data to initialize the model properly
        dummy_candles = []
        for i in range(100):
            dummy_candles.append({
                'open': 1.0 + np.random.normal(0, 0.01),
                'max': 1.0 + np.random.normal(0, 0.01),
                'min': 1.0 + np.random.normal(0, 0.01),
                'close': 1.0 + np.random.normal(0, 0.01),
                'volume': 1000 + np.random.normal(0, 100),
                'from': i
            })
        
        # Prepare data to initialize scaler
        lstm.prepare_data(dummy_candles)
        
        # Save the initialized model
        import os
        os.makedirs("model", exist_ok=True)
        lstm.save_model("model/lstm_trained.pth")
        
        logger.info("✅ Sample LSTM model created and saved")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample model: {e}")
        return False

if __name__ == "__main__":
    # Create sample model if run directly
    create_sample_model()
