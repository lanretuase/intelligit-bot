#!/usr/bin/env python3
"""
Core IQ Option Connection Manager
Handles connection, authentication, and API communication
"""

import logging
import asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import time

# Apply websocket patch if available
try:
    from websocket_patch import apply_websocket_patch
    apply_websocket_patch()
except ImportError:
    pass

try:
    from iqoptionapi.stable_api import IQ_Option
except ImportError:
    print("❌ iqoptionapi not installed. Install with: pip install iqoptionapi")
    exit(1)

logger = logging.getLogger(__name__)


class IQConnectionManager:
    """Manages IQ Option connection with reliability features"""
    
    def __init__(self, email: str, password: str, account_type: str = 'PRACTICE'):
        """
        Initialize connection manager
        
        Args:
            email: IQ Option account email
            password: IQ Option account password
            account_type: 'PRACTICE' or 'REAL'
        """
        self.email = email
        self.password = password
        self.account_type = account_type.upper()
        
        self.api = None
        self.connected = False
        self.last_connection_time = None
        self.connection_attempts = 0
        self.max_reconnect_attempts = 3
        
        logger.info(f"🔧 Connection Manager initialized for {account_type} account")
    
    async def connect(self) -> bool:
        """
        Establish connection to IQ Option
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("🔌 Connecting to IQ Option...")
            
            # Create API instance
            self.api = IQ_Option(self.email, self.password)
            
            # Attempt connection
            check, reason = self.api.connect()
            
            if check:
                # Set account balance type
                self.api.change_balance(self.account_type)
                
                self.connected = True
                self.last_connection_time = datetime.now()
                self.connection_attempts = 0
                
                # Get account info
                balance = self.api.get_balance()
                
                logger.info(f"✅ Connected to IQ Option ({self.account_type})")
                logger.info(f"💰 Balance: ${balance:.2f}")
                
                return True
            else:
                logger.error(f"❌ Connection failed: {reason}")
                self.connected = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IQ Option"""
        try:
            if self.api and self.connected:
                logger.info("🔌 Disconnecting from IQ Option...")
                # The IQ API doesn't have explicit disconnect, just clean up
                self.api = None
                self.connected = False
                logger.info("✅ Disconnected")
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")
    
    async def reconnect(self) -> bool:
        """
        Reconnect to IQ Option
        
        Returns:
            bool: True if reconnection successful
        """
        self.connection_attempts += 1
        
        if self.connection_attempts > self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        logger.info(f"🔄 Reconnecting (attempt {self.connection_attempts}/{self.max_reconnect_attempts})...")
        
        await self.disconnect()
        await asyncio.sleep(2)  # Wait before reconnecting
        
        return await self.connect()
    
    def is_connected(self) -> bool:
        """Check if connected to IQ Option"""
        return self.connected and self.api is not None
    
    def get_balance(self) -> float:
        """
        Get current account balance
        
        Returns:
            float: Account balance
        """
        try:
            if not self.is_connected():
                logger.error("❌ Not connected to IQ Option")
                return 0.0
            
            balance = self.api.get_balance()
            return float(balance)
            
        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return 0.0
    
    async def get_candles(self, asset: str, timeframe: int = 60, count: int = 100) -> Optional[List[Dict]]:
        """
        Get candle data for an asset
        
        Args:
            asset: Asset name (e.g., 'EURUSD')
            timeframe: Candle timeframe in seconds
            count: Number of candles to fetch
            
        Returns:
            List of candle dictionaries or None
        """
        try:
            if not self.is_connected():
                logger.error("❌ Not connected to IQ Option")
                return None
            
            # Get candles from API
            candles = self.api.get_candles(asset, timeframe, count, time.time())
            
            if candles and len(candles) > 0:
                return candles
            else:
                logger.warning(f"⚠️ No candles received for {asset}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting candles for {asset}: {e}")
            return None
    
    async def check_asset_available(self, asset: str) -> bool:
        """
        Check if an asset is available for trading
        
        Args:
            asset: Asset name
            
        Returns:
            bool: True if asset is available
        """
        try:
            if not self.is_connected():
                return False
            
            # Get all available assets
            all_assets = self.api.get_all_open_time()
            
            if not all_assets:
                return False
            
            # Check if asset is in the list and open
            binary_data = all_assets.get('binary', {})
            
            if asset in binary_data:
                is_open = binary_data[asset].get('open', False)
                return is_open
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking asset {asset}: {e}")
            return False
    
    async def get_available_assets(self) -> List[str]:
        """
        Get list of available trading assets
        
        Returns:
            List of asset names
        """
        try:
            if not self.is_connected():
                return []
            
            all_assets = self.api.get_all_open_time()
            
            if not all_assets:
                return []
            
            binary_data = all_assets.get('binary', {})
            
            # Filter only open assets
            available = [
                asset for asset, data in binary_data.items()
                if data.get('open', False)
            ]
            
            return available
            
        except Exception as e:
            logger.error(f"❌ Error getting available assets: {e}")
            return []
    
    async def buy(self, asset: str, amount: float, direction: str, duration: int = 1) -> Optional[Dict]:
        """
        Place a binary options trade
        
        Args:
            asset: Asset name
            amount: Stake amount
            direction: 'call' or 'put'
            duration: Trade duration in minutes
            
        Returns:
            Trade result dictionary or None
        """
        try:
            if not self.is_connected():
                logger.error("❌ Not connected to IQ Option")
                return None
            
            # Validate direction
            direction = direction.lower()
            if direction not in ['call', 'put']:
                logger.error(f"❌ Invalid direction: {direction}")
                return None
            
            # Place trade
            check, trade_id = self.api.buy(amount, asset, direction, duration)
            
            if check:
                logger.info(f"✅ Trade placed: {asset} {direction.upper()} ${amount} - ID: {trade_id}")
                return {
                    'success': True,
                    'id': trade_id,
                    'asset': asset,
                    'amount': amount,
                    'direction': direction,
                    'duration': duration,
                    'timestamp': datetime.now()
                }
            else:
                logger.error(f"❌ Trade failed for {asset}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error placing trade: {e}")
            return None
    
    async def check_win(self, trade_id: int, max_wait: int = 300) -> Optional[Tuple[bool, float]]:
        """
        Check if a trade won
        
        Args:
            trade_id: Trade ID to check
            max_wait: Maximum wait time in seconds
            
        Returns:
            Tuple of (is_win, profit) or None
        """
        try:
            if not self.is_connected():
                logger.error("❌ Not connected to IQ Option")
                return None
            
            start_time = time.time()
            
            while (time.time() - start_time) < max_wait:
                # Check trade result
                result = self.api.check_win_v4(trade_id)
                
                if result is not None:
                    is_win = result > 0
                    profit = float(result)
                    return (is_win, profit)
                
                await asyncio.sleep(1)
            
            logger.warning(f"⚠️ Timeout waiting for trade result: {trade_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking trade result: {e}")
            return None
    
    def get_api(self):
        """
        Get the underlying API instance (for advanced usage)
        
        Returns:
            IQ_Option API instance
        """
        return self.api


# Quick test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_connection():
        """Test connection manager"""
        email = os.getenv('IQ_OPTION_EMAIL', '')
        password = os.getenv('IQ_OPTION_PASSWORD', '')
        
        if not email or not password:
            print("❌ Please set IQ_OPTION_EMAIL and IQ_OPTION_PASSWORD in .env file")
            return
        
        manager = IQConnectionManager(email, password, 'PRACTICE')
        
        if await manager.connect():
            print("✅ Connection successful!")
            
            # Get balance
            balance = manager.get_balance()
            print(f"💰 Balance: ${balance:.2f}")
            
            # Get available assets
            assets = await manager.get_available_assets()
            print(f"📊 Available assets: {len(assets)}")
            print(f"   First 10: {', '.join(assets[:10])}")
            
            # Get candles
            candles = await manager.get_candles('EURUSD', 60, 10)
            if candles:
                print(f"📈 Retrieved {len(candles)} candles for EURUSD")
            
            await manager.disconnect()
        else:
            print("❌ Connection failed!")
    
    asyncio.run(test_connection())
