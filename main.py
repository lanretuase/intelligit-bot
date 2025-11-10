#!/usr/bin/env python3
"""
INTELLIGIT BOT - Main Orchestrator
Coordinates trading operations with ML predictions and risk management
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intelligit_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import core components
from config_user import BotConfig
from core_iqconnect import IQConnectionManager
from utils.signal_engine import SignalEngine
from utils.adaptive_learning import AdaptiveLearning
from utils.martingale_manager import MartingaleManager
from iq_connect.trade_executor import TradeExecutor
from model.predict import ModelPredictor


class IntelligitBot:
    """Main bot orchestrator coordinating all components"""
    
    def __init__(self, config: BotConfig):
        """
        Initialize bot with configuration
        
        Args:
            config: Bot configuration instance
        """
        self.config = config
        self.running = False
        
        # Core components
        self.connection_manager = None
        self.signal_engine = None
        self.adaptive_learning = None
        self.martingale_manager = None
        self.trade_executor = None
        self.model_predictor = None
        
        # Trading state
        self.active_trades = {}
        self.performance_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0
        }
        
        logger.info("🤖 Intelligit Bot initialized")
    
    async def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("🔧 Initializing bot components...")
            
            # Initialize IQ Option connection
            self.connection_manager = IQConnectionManager(
                email=self.config.IQ_EMAIL,
                password=self.config.IQ_PASSWORD,
                account_type=self.config.ACCOUNT_TYPE
            )
            
            if not await self.connection_manager.connect():
                raise Exception("Failed to connect to IQ Option")
            
            # Initialize model predictor
            self.model_predictor = ModelPredictor(
                model_dir=self.config.MODEL_DIR
            )
            
            # Initialize signal engine
            self.signal_engine = SignalEngine(
                connection_manager=self.connection_manager,
                model_predictor=self.model_predictor,
                threshold=self.config.SIGNAL_THRESHOLD
            )
            
            # Initialize adaptive learning
            self.adaptive_learning = AdaptiveLearning(
                performance_file=self.config.PERFORMANCE_FILE
            )
            
            # Initialize risk management
            self.martingale_manager = MartingaleManager(
                initial_stake=self.config.INITIAL_STAKE,
                max_stake=self.config.MAX_STAKE,
                max_consecutive_losses=self.config.MAX_CONSECUTIVE_LOSSES
            )
            
            # Initialize trade executor
            self.trade_executor = TradeExecutor(
                connection_manager=self.connection_manager
            )
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def scan_and_trade(self):
        """Main trading loop - scan for signals and execute trades"""
        logger.info(f"🔍 Scanning {len(self.config.ASSETS)} assets for signals...")
        
        for asset in self.config.ASSETS:
            try:
                # Generate signal
                signal = await self.signal_engine.generate_signal(asset)
                
                if signal and signal['confidence'] >= self.config.SIGNAL_THRESHOLD:
                    logger.info(f"📊 Signal found for {asset}: {signal['direction']} "
                              f"({signal['confidence']:.2f}% confidence)")
                    
                    # Check if we should trade (risk management)
                    if self.should_trade(asset):
                        await self.execute_trade(asset, signal)
                    else:
                        logger.info(f"⏭️ Skipping {asset} - Risk management")
                        
            except Exception as e:
                logger.error(f"❌ Error scanning {asset}: {e}")
                continue
    
    def should_trade(self, asset: str) -> bool:
        """Check if we should trade based on risk management"""
        # Check daily loss limit
        if abs(self.performance_stats['profit']) >= self.config.MAX_DAILY_LOSS:
            logger.warning(f"🛑 Daily loss limit reached: ${self.performance_stats['profit']:.2f}")
            return False
        
        # Check if asset is on cooldown
        if asset in self.active_trades:
            logger.info(f"⏸️ {asset} has active trade")
            return False
        
        return True
    
    async def execute_trade(self, asset: str, signal: Dict):
        """Execute a trade based on signal"""
        try:
            # Get stake from martingale manager
            stake = self.martingale_manager.get_stake()
            
            logger.info(f"💰 Executing trade: {asset} {signal['direction']} - ${stake}")
            
            # Execute trade
            trade_result = await self.trade_executor.execute(
                asset=asset,
                direction=signal['direction'],
                stake=stake,
                duration=signal.get('duration', 1)  # Default 1 minute
            )
            
            if trade_result:
                self.active_trades[asset] = {
                    'signal': signal,
                    'stake': stake,
                    'start_time': datetime.now(),
                    'trade_id': trade_result.get('id')
                }
                
                logger.info(f"✅ Trade executed successfully: {trade_result.get('id')}")
                
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    async def check_trade_results(self):
        """Check results of active trades"""
        completed_trades = []
        
        for asset, trade_info in self.active_trades.items():
            try:
                # Check if trade duration has passed
                elapsed = (datetime.now() - trade_info['start_time']).total_seconds()
                duration_seconds = trade_info['signal'].get('duration', 1) * 60
                
                if elapsed >= duration_seconds:
                    # Get trade result
                    result = await self.trade_executor.get_trade_result(
                        trade_info['trade_id']
                    )
                    
                    if result:
                        await self.process_trade_result(asset, trade_info, result)
                        completed_trades.append(asset)
                        
            except Exception as e:
                logger.error(f"❌ Error checking trade result for {asset}: {e}")
        
        # Remove completed trades
        for asset in completed_trades:
            del self.active_trades[asset]
    
    async def process_trade_result(self, asset: str, trade_info: Dict, result: Dict):
        """Process completed trade result"""
        is_win = result.get('win', False)
        profit = result.get('profit', 0.0)
        
        # Update performance stats
        self.performance_stats['total_trades'] += 1
        self.performance_stats['profit'] += profit
        
        if is_win:
            self.performance_stats['wins'] += 1
            logger.info(f"✅ WIN: {asset} - Profit: ${profit:.2f}")
            self.martingale_manager.reset()
        else:
            self.performance_stats['losses'] += 1
            logger.warning(f"❌ LOSS: {asset} - Loss: ${profit:.2f}")
            self.martingale_manager.increment()
        
        # Update adaptive learning
        self.adaptive_learning.record_trade(
            asset=asset,
            signal=trade_info['signal'],
            result=is_win
        )
        
        # Log current statistics
        win_rate = (self.performance_stats['wins'] / self.performance_stats['total_trades'] * 100) if self.performance_stats['total_trades'] > 0 else 0
        logger.info(f"📈 Stats - Trades: {self.performance_stats['total_trades']}, "
                   f"Win Rate: {win_rate:.1f}%, "
                   f"Profit: ${self.performance_stats['profit']:.2f}")
    
    async def run(self):
        """Main bot run loop"""
        self.running = True
        logger.info("🚀 Intelligit Bot started!")
        logger.info(f"📊 Trading {len(self.config.ASSETS)} assets")
        logger.info(f"⏰ Scan interval: {self.config.SCAN_INTERVAL}s")
        
        try:
            while self.running:
                # Scan and execute trades
                await self.scan_and_trade()
                
                # Check active trade results
                await self.check_trade_results()
                
                # Wait before next scan
                await asyncio.sleep(self.config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown bot"""
        logger.info("🔌 Shutting down bot...")
        self.running = False
        
        # Close connection
        if self.connection_manager:
            await self.connection_manager.disconnect()
        
        # Save performance data
        if self.adaptive_learning:
            self.adaptive_learning.save()
        
        logger.info("👋 Intelligit Bot stopped")


async def main():
    """Main entry point"""
    try:
        # Load configuration
        config = BotConfig()
        
        # Validate configuration
        if not config.validate():
            logger.error("❌ Invalid configuration. Please check your settings.")
            return
        
        # Create and initialize bot
        bot = IntelligitBot(config)
        
        if await bot.initialize():
            # Run bot
            await bot.run()
        else:
            logger.error("❌ Bot initialization failed")
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
