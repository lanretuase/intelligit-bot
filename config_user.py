#!/usr/bin/env python3
"""
User Configuration for Intelligit Bot
Centralized configuration management
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BotConfig:
    """Bot configuration class"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        
        # ===== IQ Option Credentials =====
        self.IQ_EMAIL = os.getenv('IQ_OPTION_EMAIL', '')
        self.IQ_PASSWORD = os.getenv('IQ_OPTION_PASSWORD', '')
        self.ACCOUNT_TYPE = os.getenv('ACCOUNT_TYPE', 'PRACTICE')  # PRACTICE or REAL
        
        # ===== Trading Parameters =====
        self.INITIAL_STAKE = float(os.getenv('INITIAL_STAKE', '1.0'))
        self.MAX_STAKE = float(os.getenv('MAX_STAKE', '50.0'))
        self.SIGNAL_THRESHOLD = float(os.getenv('SIGNAL_THRESHOLD', '85.0'))  # Minimum confidence %
        
        # ===== Risk Management =====
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '100.0'))
        self.MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
        
        # ===== Trading Assets =====
        default_assets = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
            'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY'
        ]
        self.ASSETS = os.getenv('TRADING_ASSETS', ','.join(default_assets)).split(',')
        self.ASSETS = [asset.strip() for asset in self.ASSETS]  # Clean whitespace
        
        # ===== Bot Timing =====
        self.SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', '60'))  # Seconds between scans
        self.TRADE_DURATION = int(os.getenv('TRADE_DURATION', '1'))  # Minutes per trade
        
        # ===== ML Model Settings =====
        self.MODEL_DIR = os.getenv('MODEL_DIR', 'model/ensemble')
        self.USE_ENSEMBLE = os.getenv('USE_ENSEMBLE', 'True').lower() == 'true'
        
        # ===== Telegram Notifications (Optional) =====
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
        self.ENABLE_TELEGRAM = bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)
        
        # ===== Dashboard Settings (Optional) =====
        self.DASHBOARD_USERNAME = os.getenv('DASHBOARD_USERNAME', 'admin')
        self.DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', '')
        self.DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))
        
        # ===== Data Storage =====
        self.PERFORMANCE_FILE = os.getenv('PERFORMANCE_FILE', 'data/performance_history.json')
        self.LOG_FILE = os.getenv('LOG_FILE', 'intelligit_bot.log')
        
        # ===== Advanced Settings =====
        self.ENABLE_ADAPTIVE_LEARNING = os.getenv('ENABLE_ADAPTIVE_LEARNING', 'True').lower() == 'true'
        self.ENABLE_MARTINGALE = os.getenv('ENABLE_MARTINGALE', 'True').lower() == 'true'
        self.MARTINGALE_MULTIPLIER = float(os.getenv('MARTINGALE_MULTIPLIER', '2.0'))
        
        # Technical Analysis
        self.USE_TECHNICAL_ANALYSIS = os.getenv('USE_TECHNICAL_ANALYSIS', 'True').lower() == 'true'
        self.TA_TIMEFRAME = int(os.getenv('TA_TIMEFRAME', '60'))  # Seconds
        self.TA_CANDLES = int(os.getenv('TA_CANDLES', '100'))  # Number of candles
        
        # Signal filtering
        self.MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', '70.0'))
        self.REQUIRE_CONFIRMATION = os.getenv('REQUIRE_CONFIRMATION', 'True').lower() == 'true'
        
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Check required credentials
        if not self.IQ_EMAIL:
            errors.append("IQ_OPTION_EMAIL is required")
        if not self.IQ_PASSWORD:
            errors.append("IQ_OPTION_PASSWORD is required")
        
        # Validate account type
        if self.ACCOUNT_TYPE not in ['PRACTICE', 'REAL']:
            errors.append("ACCOUNT_TYPE must be 'PRACTICE' or 'REAL'")
        
        # Validate stake values
        if self.INITIAL_STAKE <= 0:
            errors.append("INITIAL_STAKE must be greater than 0")
        if self.MAX_STAKE < self.INITIAL_STAKE:
            errors.append("MAX_STAKE must be greater than or equal to INITIAL_STAKE")
        
        # Validate thresholds
        if not (0 <= self.SIGNAL_THRESHOLD <= 100):
            errors.append("SIGNAL_THRESHOLD must be between 0 and 100")
        
        # Validate risk management
        if self.MAX_DAILY_LOSS <= 0:
            errors.append("MAX_DAILY_LOSS must be greater than 0")
        if self.MAX_CONSECUTIVE_LOSSES <= 0:
            errors.append("MAX_CONSECUTIVE_LOSSES must be greater than 0")
        
        # Validate assets
        if not self.ASSETS:
            errors.append("At least one trading asset must be specified")
        
        # Validate timing
        if self.SCAN_INTERVAL <= 0:
            errors.append("SCAN_INTERVAL must be greater than 0")
        if self.TRADE_DURATION <= 0:
            errors.append("TRADE_DURATION must be greater than 0")
        
        # Print errors if any
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration (hiding sensitive data)"""
        print("\n" + "="*60)
        print("🤖 INTELLIGIT BOT CONFIGURATION")
        print("="*60)
        
        print("\n📧 IQ Option Account:")
        print(f"  Email: {self.IQ_EMAIL[:3]}***@{self.IQ_EMAIL.split('@')[1] if '@' in self.IQ_EMAIL else '***'}")
        print(f"  Account Type: {self.ACCOUNT_TYPE}")
        
        print("\n💰 Trading Parameters:")
        print(f"  Initial Stake: ${self.INITIAL_STAKE}")
        print(f"  Max Stake: ${self.MAX_STAKE}")
        print(f"  Signal Threshold: {self.SIGNAL_THRESHOLD}%")
        
        print("\n⚠️ Risk Management:")
        print(f"  Max Daily Loss: ${self.MAX_DAILY_LOSS}")
        print(f"  Max Consecutive Losses: {self.MAX_CONSECUTIVE_LOSSES}")
        
        print("\n📊 Trading Assets:")
        print(f"  Assets ({len(self.ASSETS)}): {', '.join(self.ASSETS)}")
        
        print("\n⏱️ Timing:")
        print(f"  Scan Interval: {self.SCAN_INTERVAL}s")
        print(f"  Trade Duration: {self.TRADE_DURATION}m")
        
        print("\n🧠 ML Settings:")
        print(f"  Model Directory: {self.MODEL_DIR}")
        print(f"  Use Ensemble: {self.USE_ENSEMBLE}")
        
        print("\n🔔 Notifications:")
        print(f"  Telegram: {'Enabled' if self.ENABLE_TELEGRAM else 'Disabled'}")
        
        print("\n⚙️ Advanced:")
        print(f"  Adaptive Learning: {self.ENABLE_ADAPTIVE_LEARNING}")
        print(f"  Martingale: {self.ENABLE_MARTINGALE}")
        print(f"  Technical Analysis: {self.USE_TECHNICAL_ANALYSIS}")
        
        print("\n" + "="*60 + "\n")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            'account_type': self.ACCOUNT_TYPE,
            'initial_stake': self.INITIAL_STAKE,
            'max_stake': self.MAX_STAKE,
            'signal_threshold': self.SIGNAL_THRESHOLD,
            'max_daily_loss': self.MAX_DAILY_LOSS,
            'max_consecutive_losses': self.MAX_CONSECUTIVE_LOSSES,
            'assets': self.ASSETS,
            'scan_interval': self.SCAN_INTERVAL,
            'trade_duration': self.TRADE_DURATION,
            'model_dir': self.MODEL_DIR,
            'use_ensemble': self.USE_ENSEMBLE,
            'enable_telegram': self.ENABLE_TELEGRAM,
            'enable_adaptive_learning': self.ENABLE_ADAPTIVE_LEARNING,
            'enable_martingale': self.ENABLE_MARTINGALE,
            'use_technical_analysis': self.USE_TECHNICAL_ANALYSIS
        }


# Quick test
if __name__ == "__main__":
    config = BotConfig()
    
    if config.validate():
        config.print_config()
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration validation failed!")
