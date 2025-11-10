# utils/martingale_manager.py

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from config.settings import USE_BLITZ, BLITZ_DURATION_SECONDS, ENFORCE_TURBO_ONLY

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    EXPIRED = "expired"

@dataclass
class MartingaleTrade:
    """Represents a single trade in a martingale sequence"""
    trade_id: str
    asset: str
    direction: str  # 'call' or 'put'
    amount: float
    level: int  # 1, 2, or 3
    timestamp: float
    duration: int  # in minutes
    status: TradeStatus = TradeStatus.PENDING
    result: Optional[str] = None
    payout: float = 0.0
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'asset': self.asset,
            'direction': self.direction,
            'amount': self.amount,
            'level': self.level,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'status': self.status.value,
            'result': self.result,
            'payout': self.payout
        }

@dataclass
class MartingaleSequence:
    """Represents a complete martingale sequence (up to 3 trades)"""
    sequence_id: str
    asset: str
    original_signal: str  # Original signal direction
    original_confidence: float
    base_amount: float
    trades: List[MartingaleTrade]
    total_invested: float = 0.0
    total_payout: float = 0.0
    net_result: float = 0.0
    status: str = "active"  # active, completed, failed
    created_at: float = 0.0
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self):
        return {
            'sequence_id': self.sequence_id,
            'asset': self.asset,
            'original_signal': self.original_signal,
            'original_confidence': self.original_confidence,
            'base_amount': self.base_amount,
            'trades': [trade.to_dict() for trade in self.trades],
            'total_invested': self.total_invested,
            'total_payout': self.total_payout,
            'net_result': self.net_result,
            'status': self.status,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }

class MartingaleManager:
    """Manages 3-level martingale trading sequences"""
    
    def __init__(self, base_amount: float = 1.0, multiplier: float = 2.5, max_concurrent: int = 5):
        self.base_amount = base_amount
        self.multiplier = multiplier  # Amount multiplier for each level
        self.max_concurrent = max_concurrent  # Max concurrent sequences
        
        # Active sequences tracking
        self.active_sequences: Dict[str, MartingaleSequence] = {}  # sequence_id -> sequence
        self.asset_sequences: Dict[str, str] = {}  # asset -> active_sequence_id
        
        # Historical data
        self.completed_sequences: List[MartingaleSequence] = []
        
        # Statistics
        self.stats = {
            'total_sequences': 0,
            'successful_sequences': 0,
            'failed_sequences': 0,
            'level_1_wins': 0,
            'level_2_wins': 0,
            'level_3_wins': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"🎯 Martingale Manager initialized: base=${base_amount}, multiplier={multiplier}x, max_concurrent={max_concurrent}")
    
    def can_start_sequence(self, asset: str) -> Tuple[bool, str]:
        """Check if we can start a new martingale sequence for an asset"""
        
        # Check if asset already has active sequence
        if asset in self.asset_sequences:
            return False, f"Asset {asset} already has active martingale sequence"
        
        # Check concurrent limit
        if len(self.active_sequences) >= self.max_concurrent:
            return False, f"Maximum concurrent sequences reached ({self.max_concurrent})"
        
        return True, "OK"
    
    def start_sequence(self, asset: str, signal: str, confidence: float) -> Optional[str]:
        """Start a new martingale sequence"""
        
        can_start, reason = self.can_start_sequence(asset)
        if not can_start:
            logger.warning(f"🚫 Cannot start martingale for {asset}: {reason}")
            return None
        
        # Generate sequence ID
        sequence_id = f"MART_{asset}_{int(time.time())}"
        
        # Create new sequence
        sequence = MartingaleSequence(
            sequence_id=sequence_id,
            asset=asset,
            original_signal=signal,
            original_confidence=confidence,
            base_amount=self.base_amount,
            trades=[]
        )
        
        # Register sequence
        self.active_sequences[sequence_id] = sequence
        self.asset_sequences[asset] = sequence_id
        self.stats['total_sequences'] += 1
        
        logger.info(f"🎯 Started martingale sequence {sequence_id} for {asset}: {signal.upper()} @ {confidence:.1f}%")
        return sequence_id
    
    def get_next_trade_amount(self, sequence_id: str) -> float:
        """Calculate the amount for the next trade in the sequence"""
        
        if sequence_id not in self.active_sequences:
            return self.base_amount
        
        sequence = self.active_sequences[sequence_id]
        level = len(sequence.trades) + 1
        
        if level == 1:
            return self.base_amount
        elif level == 2:
            return self.base_amount * self.multiplier
        elif level == 3:
            return self.base_amount * (self.multiplier ** 2)
        else:
            logger.error(f"❌ Invalid martingale level {level} for sequence {sequence_id}")
            return self.base_amount
    
    def place_martingale_trade(self, iq, sequence_id: str, trade_id: str, duration: int) -> bool:
        """Place the next trade in a martingale sequence"""
        
        if sequence_id not in self.active_sequences:
            logger.error(f"❌ Sequence {sequence_id} not found")
            return False
        
        sequence = self.active_sequences[sequence_id]
        level = len(sequence.trades) + 1
        
        if level > 3:
            logger.error(f"❌ Cannot place level {level} trade - max is 3")
            return False
        
        # Calculate trade amount
        amount = self.get_next_trade_amount(sequence_id)
        
        # Create trade record
        trade = MartingaleTrade(
            trade_id=trade_id,
            asset=sequence.asset,
            direction=sequence.original_signal,
            amount=amount,
            level=level,
            timestamp=time.time(),
            duration=duration
        )
        
        # Blitz helpers
        def _next_expiration_epoch(seconds: int) -> int:
            now = int(time.time())
            if seconds == 30:
                return ((now // 30) + 1) * 30
            return ((now // 60) + 1) * 60

        def _is_turbo_open(iq, asset: str) -> bool:
            """Best-effort check if turbo is open for given asset. Returns True if unknown."""
            try:
                norm = asset.replace("-OTC", "")
                if not hasattr(iq, "get_all_open_time"):
                    return True
                open_info = iq.get_all_open_time()
                turbo_info = open_info.get("turbo") if isinstance(open_info, dict) else None
                if not turbo_info:
                    return True
                active_id = None
                if hasattr(iq, "get_active_id"):
                    try:
                        active_id = iq.get_active_id(norm)
                    except Exception:
                        active_id = None
                if active_id is None:
                    return True
                status = turbo_info.get(active_id)
                if isinstance(status, dict) and "open" in status:
                    return bool(status["open"])  # True if open
            except Exception:
                return True
            return True

        # Place the actual trade
        try:
            if USE_BLITZ or ENFORCE_TURBO_ONLY:
                if not _is_turbo_open(iq, sequence.asset):
                    logger.warning(f"🚫 Turbo is closed/suspended for {sequence.asset}. Skipping martingale trade.")
                    if ENFORCE_TURBO_ONLY:
                        return False
                    # If not enforcing turbo only, fall back to binary below
                    raise RuntimeError("Turbo closed; fallback to binary path")
                expired = _next_expiration_epoch(BLITZ_DURATION_SECONDS)
                option = "turbo"
                norm_asset = sequence.asset.replace("-OTC", "")
                logger.info(
                    f"⚡ Blitz L{level}: {sequence.original_signal} {sequence.asset} -> {norm_asset} ${amount:.2f} for {BLITZ_DURATION_SECONDS}s exp={expired}"
                )
                success = iq.buy_by_raw_expirations(
                    amount, norm_asset, sequence.original_signal, option, expired
                )
                actual_trade_id = None
                if ENFORCE_TURBO_ONLY and not success:
                    logger.error("⛔ Turbo enforcement active: refusing to fallback to Binary for martingale trade.")
                    return False
            else:
                success, actual_trade_id = iq.buy(amount, sequence.asset, sequence.original_signal, duration)
            
            if success:
                trade.trade_id = actual_trade_id or trade_id
                sequence.trades.append(trade)
                sequence.total_invested += amount
                
                logger.info(f"🚀 Martingale L{level} trade placed: {sequence.asset} {sequence.original_signal.upper()} ${amount:.2f} (ID: {trade.trade_id})")
                return True
            else:
                logger.error(f"❌ Failed to place martingale L{level} trade for {sequence.asset}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error placing martingale trade: {e}")
            return False
    
    def update_trade_result(self, trade_id: str, won: bool, payout: float = 0.0) -> bool:
        """Update the result of a trade and handle martingale logic"""
        
        # Find the trade in active sequences
        sequence_id = None
        trade_index = None
        
        for seq_id, sequence in self.active_sequences.items():
            for i, trade in enumerate(sequence.trades):
                if trade.trade_id == trade_id:
                    sequence_id = seq_id
                    trade_index = i
                    break
            if sequence_id:
                break
        
        if not sequence_id:
            logger.warning(f"⚠️ Trade {trade_id} not found in active martingale sequences")
            return False
        
        sequence = self.active_sequences[sequence_id]
        trade = sequence.trades[trade_index]
        
        # Update trade result
        trade.status = TradeStatus.WON if won else TradeStatus.LOST
        trade.result = "win" if won else "loss"
        trade.payout = payout
        
        sequence.total_payout += payout
        sequence.net_result = sequence.total_payout - sequence.total_invested
        
        if won:
            # Trade won - sequence successful!
            self._complete_sequence(sequence_id, True)
            logger.info(f"🎉 Martingale L{trade.level} WIN! Sequence {sequence_id} completed: +${sequence.net_result:.2f}")
            return True
        else:
            # Trade lost - check if we can continue
            if trade.level < 3:
                # Can place next level
                logger.info(f"📉 Martingale L{trade.level} LOSS. Preparing L{trade.level + 1} for {sequence.asset}")
                return True
            else:
                # All 3 levels failed - sequence failed
                self._complete_sequence(sequence_id, False)
                logger.error(f"💸 Martingale sequence {sequence_id} FAILED after 3 levels: -${sequence.total_invested:.2f}")
                return False
    
    def _complete_sequence(self, sequence_id: str, successful: bool):
        """Complete a martingale sequence and update statistics"""
        
        if sequence_id not in self.active_sequences:
            return
        
        sequence = self.active_sequences[sequence_id]
        sequence.status = "completed" if successful else "failed"
        sequence.completed_at = time.time()
        
        # Update statistics
        if successful:
            self.stats['successful_sequences'] += 1
            self.stats['total_profit'] += sequence.net_result
            
            # Track which level won
            winning_level = len(sequence.trades)
            if winning_level == 1:
                self.stats['level_1_wins'] += 1
            elif winning_level == 2:
                self.stats['level_2_wins'] += 1
            elif winning_level == 3:
                self.stats['level_3_wins'] += 1
        else:
            self.stats['failed_sequences'] += 1
            self.stats['total_loss'] += sequence.total_invested
        
        # Update win rate
        total_completed = self.stats['successful_sequences'] + self.stats['failed_sequences']
        if total_completed > 0:
            self.stats['win_rate'] = (self.stats['successful_sequences'] / total_completed) * 100
        
        # Move to completed and cleanup
        self.completed_sequences.append(sequence)
        del self.active_sequences[sequence_id]
        if sequence.asset in self.asset_sequences:
            del self.asset_sequences[sequence.asset]
        
        logger.info(f"📊 Martingale stats: {self.stats['successful_sequences']}/{total_completed} sequences won ({self.stats['win_rate']:.1f}%)")
    
    def get_pending_trades(self) -> List[MartingaleTrade]:
        """Get all pending trades across all active sequences"""
        pending_trades = []
        
        for sequence in self.active_sequences.values():
            for trade in sequence.trades:
                if trade.status == TradeStatus.PENDING:
                    pending_trades.append(trade)
        
        return pending_trades
    
    def should_place_next_level(self, sequence_id: str) -> bool:
        """Check if we should place the next level trade"""
        
        if sequence_id not in self.active_sequences:
            return False
        
        sequence = self.active_sequences[sequence_id]
        
        # Check if last trade lost and we haven't reached level 3
        if sequence.trades:
            last_trade = sequence.trades[-1]
            if last_trade.status == TradeStatus.LOST and last_trade.level < 3:
                return True
        
        return False
    
    def get_sequence_by_asset(self, asset: str) -> Optional[MartingaleSequence]:
        """Get active martingale sequence for an asset"""
        
        if asset in self.asset_sequences:
            sequence_id = self.asset_sequences[asset]
            return self.active_sequences.get(sequence_id)
        
        return None
    
    def cleanup_expired_sequences(self, max_age_hours: int = 24):
        """Clean up old sequences that may have stalled"""
        
        current_time = time.time()
        expired_sequences = []
        
        for sequence_id, sequence in self.active_sequences.items():
            age_hours = (current_time - sequence.created_at) / 3600
            if age_hours > max_age_hours:
                expired_sequences.append(sequence_id)
        
        for sequence_id in expired_sequences:
            logger.warning(f"🧹 Cleaning up expired martingale sequence: {sequence_id}")
            self._complete_sequence(sequence_id, False)
    
    def get_statistics(self) -> dict:
        """Get comprehensive martingale statistics"""
        
        return {
            **self.stats,
            'active_sequences': len(self.active_sequences),
            'net_profit': self.stats['total_profit'] - self.stats['total_loss'],
            'level_distribution': {
                'level_1': self.stats['level_1_wins'],
                'level_2': self.stats['level_2_wins'],
                'level_3': self.stats['level_3_wins']
            }
        }
    
    def save_state(self, filepath: str):
        """Save martingale manager state to file"""
        
        state = {
            'active_sequences': {k: v.to_dict() for k, v in self.active_sequences.items()},
            'asset_sequences': self.asset_sequences,
            'completed_sequences': [seq.to_dict() for seq in self.completed_sequences[-100:]],  # Last 100
            'stats': self.stats,
            'config': {
                'base_amount': self.base_amount,
                'multiplier': self.multiplier,
                'max_concurrent': self.max_concurrent
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"💾 Martingale state saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save martingale state: {e}")
    
    def load_state(self, filepath: str):
        """Load martingale manager state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore configuration
            config = state.get('config', {})
            self.base_amount = config.get('base_amount', self.base_amount)
            self.multiplier = config.get('multiplier', self.multiplier)
            self.max_concurrent = config.get('max_concurrent', self.max_concurrent)
            
            # Restore statistics
            self.stats = state.get('stats', self.stats)
            
            # Restore asset sequences mapping
            self.asset_sequences = state.get('asset_sequences', {})
            
            logger.info(f"📂 Martingale state loaded from {filepath}")
            
        except FileNotFoundError:
            logger.info(f"📂 No existing martingale state file found at {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load martingale state: {e}")

# Global martingale manager instance with configuration
try:
    from config.martingale_settings import (
        BASE_TRADE_AMOUNT, MARTINGALE_MULTIPLIER, MAX_CONCURRENT_SEQUENCES,
        validate_martingale_settings
    )
    
    # Validate settings before creating manager
    validate_martingale_settings()
    
    martingale_manager = MartingaleManager(
        base_amount=BASE_TRADE_AMOUNT,
        multiplier=MARTINGALE_MULTIPLIER,
        max_concurrent=MAX_CONCURRENT_SEQUENCES
    )
    
    logger.info(f"🎯 Martingale Manager configured: ${BASE_TRADE_AMOUNT} base, {MARTINGALE_MULTIPLIER}x multiplier")
    
except ImportError:
    # Fallback to default settings if config not available
    logger.warning("⚠️ Martingale config not found, using defaults")
    martingale_manager = MartingaleManager(
        base_amount=1.0,
        multiplier=2.5,
        max_concurrent=5
    )
except Exception as e:
    logger.error(f"❌ Martingale configuration error: {e}")
    logger.info("🔄 Using safe default martingale settings")
    martingale_manager = MartingaleManager(
        base_amount=1.0,
        multiplier=2.0,  # Conservative multiplier
        max_concurrent=3  # Conservative concurrent limit
    )
