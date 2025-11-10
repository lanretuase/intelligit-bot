from typing import Optional, Tuple, Dict, Any, List
import time
import logging
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option
from .connection_manager import ConnectionManager
from utils.martingale_manager import martingale_manager
from config.settings import USE_BLITZ, BLITZ_DURATION_SECONDS, ENFORCE_TURBO_ONLY

log = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.last_trade_time: Dict[str, float] = {}
        self.min_trade_interval = 2  # seconds between trades on same asset
        self.trade_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

    def _next_expiration_epoch(self, seconds: int) -> int:
        """Compute next expiration epoch aligned to 30s/60s boundary for Blitz."""
        now = int(time.time())
        if seconds == 30:
            return ((now // 30) + 1) * 30
        return ((now // 60) + 1) * 60

    def _can_trade_asset(self, asset: str) -> bool:
        """Check if we can trade this asset based on rate limiting."""
        current_time = time.time()
        last_time = self.last_trade_time.get(asset, 0)
        if current_time - last_time < self.min_trade_interval:
            log.warning(f"Rate limited: {asset} - waiting {self.min_trade_interval}s between trades")
            return False
        return True
        
    def _place_blitz_trade(self, asset: str, direction: str, amount: float, duration: int) -> Tuple[bool, Optional[str]]:
        """
        Place a Blitz (turbo) trade using raw expirations.
        
        Args:
            asset: Asset to trade (e.g., 'EURUSD-OTC')
            direction: 'call' or 'put'
            amount: Trade amount in USD
            duration: Duration in minutes
            
        Returns:
            Tuple of (success, trade_id or error_message)
        """
        try:
            if not self.connection_manager.ensure_connection():
                return False, "Not connected to IQ Option"
                
            # Get active ID for the asset
            active_id = self.connection_manager.get_active_id(asset)
            if not active_id:
                return False, f"Could not find active ID for {asset}"
                
            # Convert direction to IQ Option format
            iq_direction = 'call' if direction.lower() == 'call' else 'put'
            
            # Place the trade
            result = self.connection_manager.iq.buy(
                amount=amount,
                asset_name=asset,
                direction=iq_direction,
                duration=duration * 60,  # Convert minutes to seconds
                is_turbo=True
            )
            
            if isinstance(result, dict) and 'id' in result:
                self.last_trade_time[asset] = time.time()
                self._add_to_history(asset, direction, amount, duration, True, str(result['id']))
                return True, str(result['id'])
                
            return False, f"Trade failed: {result}"
            
        except Exception as e:
            log.error(f"Error in _place_blitz_trade for {asset}: {str(e)}")
            return False, str(e)

    def _add_to_history(self, asset: str, direction: str, amount: float, duration: int, success: bool, trade_id: str):
        """Add trade to history with size limit."""
        trade_record = {
            'timestamp': datetime.now(),
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'duration': duration,
            'success': success,
            'trade_id': trade_id,
            'result': None,  # Will be updated when result is known
            'profit': 0.0,   # Will be updated when result is known
            'status': 'pending'  # pending, won, lost
        }
        
        self.trade_history.append(trade_record)
        
        # Keep history size manageable
        if len(self.trade_history) > self.max_history_size:
            self.trade_history = self.trade_history[-self.max_history_size:]
    
    def check_trade_result(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Check the result of a specific trade"""
        try:
            if not self.connection_manager.ensure_connection():
                return None
            
            iq = self.connection_manager.iq
            
            # Try to get trade result from IQ Option
            result = iq.check_win_v3(trade_id)
            
            if result is not None:
                # Find the trade in history and update it
                for trade in self.trade_history:
                    if trade['trade_id'] == trade_id:
                        if result > 0:
                            trade['status'] = 'won'
                            trade['profit'] = result
                            trade['result'] = True
                        else:
                            trade['status'] = 'lost'
                            trade['profit'] = result
                            trade['result'] = False
                        
                        log.info(f"Trade {trade_id} result: {'WON' if result > 0 else 'LOST'} (${result:.2f})")
                        return trade
                        
            return None
            
        except Exception as e:
            log.error(f"Error checking trade result for {trade_id}: {e}")
            return None
    
    def get_pending_trades(self) -> List[Dict[str, Any]]:
        """Get all pending trades that need result checking"""
        return [trade for trade in self.trade_history if trade['status'] == 'pending']
    
    def update_trade_result(self, trade_id: str, won: bool, profit: float):
        """Manually update a trade result"""
        for trade in self.trade_history:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'won' if won else 'lost'
                trade['result'] = won
                trade['profit'] = profit
                log.info(f"Updated trade {trade_id}: {'WON' if won else 'LOST'} (${profit:.2f})")
                return trade
        return None

    def place_blitz_trade(
        self, 
        asset: str, 
        direction: str, 
        amount: float, 
        seconds: int
    ) -> Tuple[bool, Optional[str]]:
        """Place a Blitz (turbo) trade with connection management."""
        if not self.connection_manager.ensure_connection():
            log.error("Cannot place trade: No active connection")
            self._add_to_history(asset, direction, amount, seconds, False)
            return False, None

        if not self.connection_manager.is_blitz_available(asset):
            log.warning(f"Blitz not available for {asset}")
            self._add_to_history(asset, direction, amount, seconds, False)
            return False, None

        if not self._can_trade_asset(asset):
            self._add_to_history(asset, direction, amount, seconds, False)
            return False, None

        try:
            iq = self.connection_manager.iq
            if not iq:
                raise ConnectionError("No active IQ Option connection")
                
            expiration = self._next_expiration_epoch(seconds)
            result = iq.buy(amount, asset, direction, seconds)
            
            if isinstance(result, dict) and result.get('id'):
                self.last_trade_time[asset] = time.time()
                trade_id = str(result['id'])
                log.info(f"✅ Placed {direction.upper()} trade on {asset} for {seconds}s (ID: {trade_id})")
                self._add_to_history(asset, direction, amount, seconds, True, trade_id)
                return True, trade_id
                
            log.error(f"Failed to place trade: {result}")
            self._add_to_history(asset, direction, amount, seconds, False)
            return False, None
            
        except Exception as e:
            log.error(f"Error placing Blitz trade: {e}")
            self._add_to_history(asset, direction, amount, seconds, False)
            return False, None

# Global instance for backward compatibility
_global_executor = None

def init_global_executor(connection_manager: ConnectionManager) -> None:
    """Initialize the global trade executor."""
    global _global_executor
    _global_executor = TradeExecutor(connection_manager)

def place_trade(iq, asset: str, direction: str, amount: float, duration: int) -> Tuple[bool, Optional[str]]:
    """
    Place a trade with proper Blitz/Binary handling.
    Enhanced with better error handling and logging.
    
    Args:
        iq: IQ Option API instance
        asset: Asset to trade (e.g., 'EURUSD-OTC')
        direction: 'call' or 'put'
        amount: Trade amount in USD
        duration: Duration in minutes
        
    Returns:
        Tuple of (success, trade_id or error_message)
    """
    global _global_executor
    
    # Input validation
    if not _global_executor or not _global_executor.connection_manager:
        log.error("❌ Trade executor not properly initialized")
        return False, "Trade executor not initialized"
        
    if not isinstance(amount, (int, float)) or amount <= 0:
        log.error(f"❌ Invalid trade amount: {amount}")
        return False, f"Invalid trade amount: {amount}"
        
    # Ensure we have a valid direction
    direction = direction.lower()
    if direction not in ['call', 'put']:
        log.error(f"❌ Invalid trade direction: {direction}")
        return False, f"Invalid direction: {direction}. Must be 'call' or 'put'"
    
    # Check if asset is available for trading using connection manager
    if not _global_executor.connection_manager or not hasattr(_global_executor.connection_manager, 'is_asset_available'):
        log.warning("⚠️ Connection manager doesn't support asset availability check, skipping...")
    else:
        if not _global_executor.connection_manager.is_asset_available(asset):
            log.error(f"❌ Asset {asset} is not available for trading")
            return False, f"Asset {asset} is not available for trading"
    
    # Get active ID for the asset
    active_id = _global_executor.connection_manager.get_active_id(asset)
    if not active_id:
        log.error(f"❌ Could not get active ID for {asset}")
        return False, f"Could not get active ID for {asset}"
    
    # Log trade attempt
    log.info(f"🚀 Attempting to place {direction.upper()} trade for {asset} (${amount}, {duration}min)")
    
    # Try Blitz first if enabled (Blitz/Turbo seconds-based)
    if USE_BLITZ:
        log.info(f"🔵 Using Blitz mode for {asset} ({BLITZ_DURATION_SECONDS}s)")
        try:
            # Enforce seconds-based turbo trade regardless of incoming duration minutes
            success, result = _global_executor.place_blitz_trade(
                asset=asset,
                direction=direction,
                amount=amount,
                seconds=BLITZ_DURATION_SECONDS,
            )
            if success:
                log.info(f"✅ Blitz trade placed successfully: {result}")
                return True, result
            log.warning(f"⚠️ Blitz trade failed: {result}")
            if ENFORCE_TURBO_ONLY:
                return False, f"Blitz trade failed and turbo enforcement is on: {result}"
        except Exception as e:
            log.error(f"❌ Error in Blitz trade execution: {str(e)}")
            if ENFORCE_TURBO_ONLY:
                return False, f"Blitz trade error and turbo enforcement is on: {str(e)}"
    
    # Fall back to standard binary options if Blitz fails and not enforcing
    log.info(f"🔄 Falling back to standard binary options for {asset}")
    try:
        # Convert direction to IQ Option format
        iq_direction = 'call' if direction == 'call' else 'put'
        
        # Place the trade
        log.debug(f"Placing binary trade: {iq_direction} {asset} ${amount} for {duration}min")
        result = iq.buy(
            amount=amount,
            active=active_id,
            direction=iq_direction,
            duration=duration * 60,  # Convert minutes to seconds
            is_turbo=False
        )
        
        # Process the result
        if isinstance(result, dict) and 'id' in result:
            trade_id = str(result['id'])
            _global_executor.last_trade_time[asset] = time.time()
            _global_executor._add_to_history(asset, direction, amount, duration, True, trade_id)
            log.info(f"✅ Binary trade placed successfully: {trade_id}")
            return True, trade_id
            
        log.error(f"❌ Binary trade failed: {result}")
        return False, f"Trade failed: {result}"
        
    except Exception as e:
        error_msg = f"Error placing binary trade: {str(e)}"
        log.error(f"❌ {error_msg}")
        return False, error_msg

def _is_turbo_open(iq, asset: str) -> bool:
    """Legacy function for backward compatibility."""
    if hasattr(iq, '_connection_manager'):
        return iq._connection_manager.is_blitz_available(asset)
    return True


# Removed duplicate place_trade function - using the first implementation only

def place_martingale_trade(iq, asset: str, direction: str, confidence: float, duration: int) -> Tuple[bool, Optional[str]]:
    """Place a trade with 3-level martingale support"""
    
    # Check if we can start a new martingale sequence
    can_start, reason = martingale_manager.can_start_sequence(asset)
    
    if not can_start:
        log.warning(f"🚫 Martingale blocked for {asset}: {reason}")
        # Fall back to regular trade
        amount = martingale_manager.base_amount
        return place_trade(iq, asset, direction, amount, duration)
    
    # Start new martingale sequence
    sequence_id = martingale_manager.start_sequence(asset, direction, confidence)
    
    if not sequence_id:
        log.error(f"❌ Failed to start martingale sequence for {asset}")
        # Fall back to regular trade
        amount = martingale_manager.base_amount
        return place_trade(iq, asset, direction, amount, duration)
    
    # Place the first trade (Level 1)
    trade_id = f"MART_L1_{asset}_{int(time.time())}"
    success = martingale_manager.place_martingale_trade(iq, sequence_id, trade_id, duration)
    
    if success:
        log.info(f"🎯 Martingale Level 1 trade started for {asset}: {direction.upper()} @ {confidence:.1f}%")
        return True, sequence_id
    else:
        log.error(f"❌ Failed to place martingale Level 1 trade for {asset}")
        return False, None

def continue_martingale_sequence(iq, sequence_id: str, duration: int) -> bool:
    """Continue a martingale sequence to the next level after a loss"""
    
    if not martingale_manager.should_place_next_level(sequence_id):
        log.warning(f"⚠️ Cannot continue martingale sequence {sequence_id}")
        return False
    
    # Get sequence details
    sequence = martingale_manager.active_sequences.get(sequence_id)
    if not sequence:
        log.error(f"❌ Sequence {sequence_id} not found")
        return False
    
    next_level = len(sequence.trades) + 1
    trade_id = f"MART_L{next_level}_{sequence.asset}_{int(time.time())}"
    
    success = martingale_manager.place_martingale_trade(iq, sequence_id, trade_id, duration)
    
    if success:
        log.info(f"🎯 Martingale Level {next_level} trade placed for {sequence.asset}")
        return True
    else:
        log.error(f"❌ Failed to place martingale Level {next_level} trade")
        return False

def update_martingale_result(trade_id: str, won: bool, payout: float = 0.0) -> bool:
    """Update the result of a martingale trade"""
    
    success = martingale_manager.update_trade_result(trade_id, won, payout)
    
    if success and won:
        log.info(f"🎉 Martingale trade {trade_id} WON: +${payout:.2f}")
    elif success and not won:
        log.info(f"📉 Martingale trade {trade_id} LOST")
    
    return success

def get_martingale_stats() -> dict:
    """Get current martingale statistics"""
    return martingale_manager.get_statistics()

def cleanup_martingale_sequences():
    """Clean up expired martingale sequences"""
    martingale_manager.cleanup_expired_sequences()

def save_martingale_state(filepath: str = "data/martingale_state.json"):
    """Save martingale state to file"""
    martingale_manager.save_state(filepath)

def load_martingale_state(filepath: str = "data/martingale_state.json"):
    """Load martingale state from file"""
    martingale_manager.load_state(filepath)
