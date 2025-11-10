# iq_connect/candle_manager.py

import time
import logging
from iq_connect.connection_manager import ensure_connection
from typing import List, Dict

log = logging.getLogger(__name__)

def _get_candles_with_fallbacks(iq, asset: str, duration: int, count: int) -> List[Dict]:
    """Enhanced candle fetching with multiple fallback methods"""
    methods = [
        ('primary', lambda: iq.get_candles(asset, duration * 60, count, time.time())),
        ('recent', lambda: iq.get_candles(asset, duration * 60, min(count, 50), time.time() - 3600)),
        ('minimal', lambda: iq.get_candles(asset, duration * 60, 20, time.time() - 1800))
    ]
    
    for method_name, method in methods:
        try:
            log.debug(f"Trying {method_name} candle fetch for {asset}")
            
            # Check connection before each attempt
            if hasattr(iq, 'check_connect') and not iq.check_connect():
                log.warning(f"Connection lost before {method_name} fetch for {asset}")
                continue
                
            candles = method()
            if candles and len(candles) >= 10:  # Minimum viable candles
                log.info(f"✅ Fetched {len(candles)} candles for {asset} via {method_name} method")
                return candles
            elif candles:
                log.warning(f"⚠️ Only {len(candles)} candles for {asset} via {method_name}")
        except Exception as e:
            log.debug(f"❌ {method_name} method failed for {asset}: {str(e)[:100]}")
            continue
    
    log.error(f"❌ All candle fetch methods failed for {asset}")
    return []

def get_candles(connection_manager, asset: str, duration: int, count: int = 100) -> List[Dict]:
    """
    Get candles with enhanced error handling and retry logic
    """
    if not connection_manager:
        log.error("❌ No connection manager provided")
        return []
    
    # Check connection manager type and use appropriate method
    connection_type = type(connection_manager).__name__
    
    if connection_type == 'RobustConnectionManager':
        # Enhanced robust connection manager with multiple fallback methods
        if hasattr(connection_manager, 'iq') and connection_manager.iq:
            return _get_candles_with_fallbacks(connection_manager.iq, asset, duration, count)
        else:
            log.error(f"❌ RobustConnectionManager has no active IQ connection")
            return []
    elif connection_type == 'SyncHTTPConnection':
        # Synchronous HTTP connection manager - use get_candles method directly
        try:
            # Check connection status first
            if not connection_manager.is_connected():
                log.warning(f"HTTP connection not active for {asset}, attempting reconnect")
                if not connection_manager.connect():
                    log.error(f"Failed to reconnect HTTP connection for {asset}")
                    return []
            
            candles = connection_manager.get_candles(asset, duration * 60, count, time.time())
            if candles and len(candles) > 0:
                log.info(f"✅ Fetched {len(candles)} candles for {asset} via HTTP connection")
                return candles
            else:
                log.warning(f"⚠️ No candles returned for {asset}")
                return []
        except Exception as e:
            log.error(f"❌ Failed to fetch candles via HTTP connection for {asset}: {str(e)[:100]}")
            return []
    elif connection_type == 'WebSocketConnectionManager':
        # WebSocket connection manager - use trader's get_candles method directly
        try:
            candles = connection_manager.trader.get_candles(asset, duration * 60, count, time.time())
            if candles and len(candles) > 0:
                log.info(f"✅ Fetched {len(candles)} candles for {asset} via WebSocket trader")
                return candles
            else:
                log.warning(f"⚠️ No candles returned for {asset}")
                return []
        except Exception as e:
            log.error(f"❌ Failed to fetch candles via WebSocket trader for {asset}: {e}")
            return []
    elif hasattr(connection_manager, 'iq') and connection_manager.iq:
        # Legacy robust connection manager check
        return _get_candles_robust(connection_manager, asset, duration, count)
    elif hasattr(connection_manager, 'trader') and connection_manager.trader:
        # Legacy WebSocket connection manager check
        try:
            candles = connection_manager.trader.get_candles(asset, duration * 60, count, time.time())
            if candles and len(candles) > 0:
                log.info(f"✅ Fetched {len(candles)} candles for {asset} via WebSocket trader")
                return candles
            else:
                log.warning(f"⚠️ No candles returned for {asset}")
                return []
        except Exception as e:
            log.error(f"❌ Failed to fetch candles via WebSocket trader for {asset}: {e}")
            return []
    else:
        log.error(f"❌ Unknown connection manager type: {type(connection_manager)}")
        return []

def _get_candles_robust(connection_manager, asset, duration, count):
    max_retries = 5  # Increased from 3 to 5
    retry_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            # Handle different connection manager types (prefer WebSocket/Blitz)
            if hasattr(connection_manager, 'trader') and connection_manager.trader is not None:
                # WebSocket connection manager (Blitz)
                if not connection_manager.ensure_connection():
                    log.error("❌ Cannot connect to IQ Option WebSocket")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                iq = connection_manager.trader
                log.debug("[CANDLES] Using WebSocket trader for Blitz candles")
            elif hasattr(connection_manager, 'iq') and connection_manager.iq is not None:
                # Standard connection manager
                iq = connection_manager.iq
                if not ensure_connection(iq):
                    log.error("❌ Cannot connect to IQ Option API")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                log.debug("[CANDLES] Using standard IQ client for candles (fallback)")
            else:
                log.debug(f"Using fallback method for connection manager type: {type(connection_manager)}")
                # Try direct IQ connection as fallback
                try:
                    candles = connection_manager.iq.get_candles(asset, duration * 60, count, time.time())
                    if candles and len(candles) > 0:
                        log.info(f"✅ Fetched {len(candles)} candles for {asset} via fallback method")
                        return candles
                    else:
                        log.warning(f"⚠️ No candles returned for {asset} via fallback")
                        return None
                except Exception as e:
                    log.error(f"❌ Fallback candle fetch failed for {asset}: {e}")
                    return None
            
            # Check if asset is available
            try:
                is_open = True
                if hasattr(iq, 'check_asset_open'):
                    is_open = bool(iq.check_asset_open(asset))
                if not is_open:
                    log.warning(f"⚠️ Asset {asset} is not currently available for trading")
                    return None
            except Exception:
                # If API does not expose this, proceed optimistically
                pass
                
            # Get active ID for the asset
            try:
                active_id = None
                manager = getattr(iq, '_connection_manager', None)
                if manager is not None and hasattr(manager, 'get_active_id'):
                    active_id = manager.get_active_id(asset)
                # If still None, continue without blocking; some API calls don't require it here
            except Exception:
                active_id = None
                
            # Fetch candles with proper timestamp
            end_time = int(time.time())
            # IQ Option API expects duration in seconds; convert minutes to seconds for consistency
            candles = iq.get_candles(asset, duration * 60, count, end_time)
            
            # Validate candle data
            if not candles or not isinstance(candles, list) or len(candles) == 0:
                log.warning(f"⚠️ Attempt {attempt + 1}: No valid candle data for {asset}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
                
            # Validate candle structure - support both formats
            sample_candle = candles[0]
            try:
                # Check if it's a valid candle with required OHLC data
                if isinstance(sample_candle, dict):
                    # WebSocket format: open, close, high, low, volume, time
                    # Standard format: open, close, max, min, volume, from
                    has_ohlc = ('open' in sample_candle and 'close' in sample_candle and 
                               (('high' in sample_candle and 'low' in sample_candle) or 
                                ('max' in sample_candle and 'min' in sample_candle)) and
                               ('time' in sample_candle or 'from' in sample_candle))
                    
                    if not has_ohlc:
                        log.warning(f"⚠️ Attempt {attempt + 1}: Invalid candle structure for {asset}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
                    # Normalize candle format
                    normalized_candles = []
                    for candle in candles:
                        normalized = {
                            'open': float(candle.get('open', 0)),
                            'close': float(candle.get('close', 0)),
                            'high': float(candle.get('high', candle.get('max', 0))),
                            'low': float(candle.get('low', candle.get('min', 0))),
                            'volume': float(candle.get('volume', 1000000)),
                            'time': int(candle.get('time', candle.get('from', end_time)))
                        }
                        normalized_candles.append(normalized)
                    
                    log.info(f"✅ Fetched {len(normalized_candles)} candles for {asset}")
                    return normalized_candles
                    
                else:
                    log.warning(f"⚠️ Attempt {attempt + 1}: Invalid candle format for {asset}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                    
            except Exception as validation_error:
                log.warning(f"⚠️ Candle validation failed for {asset}: {validation_error}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
                
        except Exception as e:
            log.error(f"❌ Failed to fetch candles for {asset} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    log.error(f"❌ Failed to retrieve candles for {asset} after {max_retries} attempts")
    return []


# Remove duplicate function - using the enhanced version above
