import os
import time
import json
import requests
import websocket
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    success: bool
    trade_id: Optional[str] = None
    message: Optional[str] = None

class IQOptionWebSocketTrader:
    LOGIN_URL = 'https://api.iqoption.com/v2/login'
    LOGOUT_URL = "https://auth.iqoption.com/api/v1.0/logout"
    WS_URL = 'wss://ws.iqoption.com/echo/websocket'
    
    # Constants for account types
    ACCOUNT_REAL = 1
    ACCOUNT_TOURNAMENT = 2
    ACCOUNT_DEMO = 4
    ACCOUNT_CFD = 6

    def __init__(self, email: str, password: str, account_mode: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.account_mode = account_mode.upper()
        self.session = requests.Session()
        self.session.trust_env = False  # Bypass proxy for IQ Option
        self.websocket: Optional[websocket.WebSocketApp] = None
        self.ws_is_active = False
        self.profile_msg: Optional[dict] = None
        self.balance_data: Optional[dict] = None
        self.active_balance_id: Optional[int] = None
        self._ws_thread: Optional[threading.Thread] = None
        self.trade_responses: Dict[str, dict] = {}
        self.connected = False
        # Capture latest candle-generated events by (active_id, size)
        self.last_candle_events: Dict[str, dict] = {}
        # Dynamic active_id map loaded/saved to disk
        self.dynamic_active_ids: Dict[str, int] = {}
        self._load_active_id_map()

    def connect(self) -> bool:
        """Connect to IQ Option and authenticate"""
        try:
            logger.info("🔗 Connecting to IQ Option WebSocket...")
            
            # Login via HTTP
            response = self.login()
            if response.status_code != 200:
                logger.error(f"❌ Login failed: {response.status_code}")
                return False
            
            logger.info("✅ HTTP login successful")
            
            # Initialize WebSocket
            self._initialize_websocket()
            
            # Authenticate WebSocket
            self._authenticate_websocket()
            
            # Switch to appropriate account
            target_account = 'demo' if self.account_mode == 'PRACTICE' else 'real'
            self.switch_account(target_account)
            
            self.connected = True
            logger.info(f"✅ Connected to IQ Option ({target_account} account)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def login(self) -> requests.Response:
        """Authenticate with IQOption API"""
        data = {
            'identifier': self.email,
            'password': self.password
        }
        response = self.session.post(url=self.LOGIN_URL, data=data)
        response.raise_for_status()
        return response

    def _initialize_websocket(self) -> None:
        """Setup and start the WebSocket connection with proper handshake"""
        # Reset connection state
        self.ws_is_active = False
        self.connected = False
        
        # Create WebSocket connection with proper headers for IQ Option
        self.websocket = websocket.WebSocketApp(
            self.WS_URL,
            on_message=lambda ws, msg: self._on_message(ws, msg),
            on_open=lambda ws: self._on_open(ws),
            on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg),
            on_error=lambda ws, error: self._on_error(ws, error),
            header=[
                'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Origin: https://iqoption.com',
                'Pragma: no-cache',
                'Cache-Control: no-cache'
            ]
        )

        # Start WebSocket in a separate thread
        self._ws_thread = threading.Thread(target=self.websocket.run_forever, kwargs={
            'ping_interval': 20,
            'ping_timeout': 10
        })
        self._ws_thread.daemon = True
        self._ws_thread.start()

        # Wait for WebSocket connection to be established
        timeout = 15  # Increased timeout for connection
        start_time = time.time()
        while not self.ws_is_active and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.ws_is_active:
            raise ConnectionError("WebSocket connection timeout")

    def _authenticate_websocket(self) -> None:
        """Authenticate the WebSocket connection using session cookies"""
        try:
            ssid = self.session.cookies.get('ssid')
            if not ssid:
                raise ConnectionError("SSID not found in session cookies")
            
            # First send the SSID for authentication
            auth_request = {
                'name': 'ssid',
                'msg': ssid,
                'request_id': f'auth_{int(time.time())}'
            }
            self.websocket.send(json.dumps(auth_request))
            logger.info("🔑 Sent WebSocket authentication request")
            
            # Then send the profile request
            profile_request = {
                'name': 'get-profile',
                'msg': {},
                'request_id': f'profile_{int(time.time())}'
            }
            self.websocket.send(json.dumps(profile_request))
            
            # Wait for profile to arrive to confirm successful authentication
            timeout = 15  # Increased timeout for authentication
            start_time = time.time()
            while self.profile_msg is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if self.profile_msg is None:
                raise ConnectionError("WebSocket authentication timeout")
                
            logger.info("✅ WebSocket authentication successful")
            
        except Exception as e:
            logger.error(f"❌ WebSocket authentication failed: {e}")
            raise

    def _send_websocket_request(self, name: str, msg, request_id: Optional[int] = None) -> None:
        """Send a request through WebSocket"""
        if not self.websocket:
            raise ConnectionError("WebSocket is not initialized")
            
        if request_id is None:
            request_id = int(str(time.time()).split('.')[1])

        data = json.dumps({
            'name': name,
            'msg': msg,
            'request_id': request_id
        })
        self.websocket.send(data)

    def is_market_open(self) -> bool:
        """Check if the market is currently open for trading"""
        # Get current time in UTC
        now_utc = datetime.utcnow()
        # Convert to server time (assuming server is in UTC+0)
        server_hour = now_utc.hour
        # Market is open 24/5 for forex, but let's be safe
        # Check if it's a weekday (0=Monday, 6=Sunday)
        is_weekday = now_utc.weekday() < 5
        # Check if it's within trading hours (24/5 for forex)
        return is_weekday

    def is_asset_tradable(self, asset: str) -> bool:
        """Check if an asset is currently tradable"""
        try:
            # Get the active ID for the asset
            active_id = self._get_asset_id(asset)
            if not active_id:
                logger.warning(f"⚠️ Could not find active ID for {asset}")
                return False
                
            # Check if we have recent asset info
            if active_id in self.assets_info and 'suspended' in self.assets_info[active_id]:
                if self.assets_info[active_id]['suspended']:
                    logger.warning(f"⚠️ Asset {asset} is currently suspended")
                    return False
                return True
                
            # If we don't have asset info, assume it's tradable
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking asset tradability: {e}")
            return True  # Assume tradable if we can't check

    def place_binary_trade(self, asset: str, direction: str, amount: float, duration_minutes: int = 3) -> TradeResult:
        """Place a binary options trade via WebSocket"""
        try:
            if not self.connected or not self.websocket:
                logger.error("❌ Not connected to WebSocket")
                return TradeResult(False, None, "Not connected to WebSocket")
                
            # Check if market is open
            if not self.is_market_open():
                logger.warning("⚠️ Market is currently closed (weekend or outside trading hours)")
                return TradeResult(False, None, "Market is currently closed")
                
            # Check if asset is tradable
            if not self.is_asset_tradable(asset):
                return TradeResult(False, None, f"Asset {asset} is not currently tradable")
            
            # Get the active ID for the asset
            active_id = self._get_asset_id(asset)
            if not active_id:
                return TradeResult(False, None, f"Could not find active ID for {asset}")
                
            # Generate a unique request ID
            request_id = f"trade_{int(time.time() * 1000)}"
            
            # Calculate expiration time (next candle close)
            current_time = int(time.time())
            expiration_timestamp = current_time - (current_time % (duration_minutes * 60)) + (duration_minutes * 60)
            
            # Prepare the trade request with the exact format expected by IQ Option API
            trade_request = {
                "name": "sendMessage",
                "request_id": request_id,
                "msg": {
                    "name": "binary-options.open-option",
                    "version": "1.0",
                    "body": {
                        "user_balance_id": int(self.active_balance_id),
                        "active_id": int(active_id),
                        "option_type_id": 1,  # Binary option
                        "direction": direction.lower(),
                        "expired": expiration_timestamp,
                        "price": amount,  # Keep as is, let json.dumps handle the number format
                        "type": "turbo" if duration_minutes <= 5 else "binary"
                    }
                }
            }
            
            # Log the trade request for debugging
            logger.debug(f"Sending trade request: {json.dumps(trade_request, indent=2)}")
            
            logger.info(f"📈 Placing {direction.upper()} trade: {asset} ${amount} for {duration_minutes}min")
            
            # Send the request
            logger.debug(f"Sending trade request: {json.dumps(trade_request, indent=2)}")
            self.websocket.send(json.dumps(trade_request))
            
            # Wait for response with timeout
            start_time = time.time()
            timeout = 10  # seconds
            
            while time.time() - start_time < timeout:
                if request_id in self.trade_responses:
                    response = self.trade_responses.pop(request_id)
                    logger.debug(f"Received trade response: {response}")
                    
                    # Check for successful trade placement
                    if response.get('msg', {}).get('status') == 'success':
                        trade_id = response.get('msg', {}).get('result', {}).get('id')
                        if trade_id:
                            logger.info(f"✅ Trade placed successfully! ID: {trade_id}")
                            return TradeResult(True, str(trade_id), "Trade placed successfully")
                        else:
                            logger.error(f"❌ Trade response missing trade ID: {response}")
                            return TradeResult(False, None, "Invalid trade response: missing trade ID")
                    else:
                        error_msg = response.get('msg', {}).get('message', 'Unknown error')
                        if 'active is suspended' in str(error_msg).lower():
                            logger.warning(f"⚠️ Asset {asset} is currently suspended")
                            return TradeResult(False, None, f"Asset {asset} is currently suspended")
                        else:
                            logger.error(f"❌ Trade failed: {error_msg}")
                            return TradeResult(False, None, error_msg)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # If we get here, we timed out waiting for a response
            logger.error(f"❌ Timeout waiting for trade response (request_id: {request_id})")
            return TradeResult(False, None, "Timeout waiting for trade response")
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return TradeResult(False, None, str(e))

    def place_turbo_trade_seconds(self, asset: str, direction: str, amount: float, seconds: int = 30) -> TradeResult:
        """Place a turbo (Blitz) trade using seconds-based expiration (e.g., 30s)."""
        try:
            if not self.connected or not self.websocket:
                logger.error("❌ Not connected to WebSocket")
                return TradeResult(False, None, "Not connected to WebSocket")

            # Check if market is open
            if not self.is_market_open():
                logger.warning("⚠️ Market is currently closed (weekend or outside trading hours)")
                return TradeResult(False, None, "Market is currently closed")

            # Check if asset is tradable
            if not self.is_asset_tradable(asset):
                return TradeResult(False, None, f"Asset {asset} is not currently tradable")

            # Get the active ID for the asset
            active_id = self._get_asset_id(asset)
            if not active_id:
                return TradeResult(False, None, f"Could not find active ID for {asset}")

            # Generate a unique request ID
            request_id = f"trade_{int(time.time() * 1000)}"

            # Calculate expiration time aligned to the next N-second boundary
            current_time = int(time.time())
            step = max(5, int(seconds))  # guardrail
            expiration_timestamp = current_time - (current_time % step) + step

            trade_request = {
                "name": "sendMessage",
                "request_id": request_id,
                "msg": {
                    "name": "binary-options.open-option",
                    "version": "1.0",
                    "body": {
                        "user_balance_id": int(self.active_balance_id),
                        "active_id": int(active_id),
                        "option_type_id": 1,  # Binary option API with turbo type
                        "direction": direction.lower(),
                        "expired": expiration_timestamp,
                        "price": amount,
                        "type": "turbo"
                    }
                }
            }

            logger.debug(f"Sending turbo trade request: {json.dumps(trade_request, indent=2)}")
            logger.info(f"⚡ Placing TURBO trade {direction.upper()}: {asset} ${amount} for {seconds}s (expires {expiration_timestamp})")

            # Send the request
            self.websocket.send(json.dumps(trade_request))

            # Wait for response with timeout
            start_time = time.time()
            timeout = 10  # seconds

            while time.time() - start_time < timeout:
                if request_id in self.trade_responses:
                    response = self.trade_responses.pop(request_id)
                    logger.debug(f"Received turbo trade response: {response}")

                    if response.get('msg', {}).get('status') == 'success':
                        trade_id = response.get('msg', {}).get('result', {}).get('id')
                        if trade_id:
                            logger.info(f"✅ Turbo trade placed successfully! ID: {trade_id}")
                            return TradeResult(True, str(trade_id), "Trade placed successfully")
                        else:
                            logger.error(f"❌ Turbo trade response missing trade ID: {response}")
                            return TradeResult(False, None, "Invalid trade response: missing trade ID")
                    else:
                        error_msg = response.get('msg', {}).get('message', 'Unknown error')
                        if 'active is suspended' in str(error_msg).lower():
                            logger.warning(f"⚠️ Asset {asset} is currently suspended")
                            return TradeResult(False, None, f"Asset {asset} is currently suspended")
                        else:
                            logger.error(f"❌ Turbo trade failed: {error_msg}")
                            return TradeResult(False, None, error_msg)

                time.sleep(0.1)

            logger.error(f"❌ Timeout waiting for turbo trade response (request_id: {request_id})")
            return TradeResult(False, None, "Timeout waiting for trade response")

        except Exception as e:
            logger.error(f"❌ Turbo trade execution error: {e}")
            return TradeResult(False, None, str(e))

    def _get_active_id(self, asset: str) -> int:
        """Resolve asset to active ID using Blitz/Turbo mapping only."""
        # Delegate to the Blitz-oriented mapping defined in `_get_asset_id` below
        return self._get_asset_id(asset)

    def _wait_for_trade_response(self, request_id: int, timeout: int = 10) -> Optional[dict]:
        """Wait for trade response"""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if str(request_id) in self.trade_responses:
                return self.trade_responses.pop(str(request_id))
            time.sleep(0.1)
        return None

    def get_balance(self) -> Optional[float]:
        """Get current account balance"""
        try:
            if not self.balance_data:
                # Try to fetch balance if not available
                self.fetch_account_balances_v2()
                if not self.balance_data:
                    logger.warning("⚠️ Balance data not available")
                    return None
            
            # Handle different balance data structures
            if isinstance(self.balance_data, list):
                # Find the balance for the current account mode
                for balance in self.balance_data:
                    if (self.account_mode == 'REAL' and balance.get('type') == 1) or \
                       (self.account_mode == 'PRACTICE' and balance.get('type') == 4):
                        return float(balance.get('amount', 0.0))
                logger.warning("⚠️ No balance found for current account mode")
                return None
            elif isinstance(self.balance_data, dict):
                # Direct balance amount
                return float(self.balance_data.get('amount', 0.0))
            else:
                logger.warning(f"⚠️ Unexpected balance data format: {type(self.balance_data)}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get balance: {e}")
            return None

    def get_candles(self, asset: str, duration: int, count: int, end_time: Optional[float] = None) -> List[dict]:
        """
        Get historical candles for analysis. Falls back to HTTP API if WebSocket fails.
        """
        # Ensure connection is alive; attempt quick reconnect if needed
        try:
            if not self.connected or not self.websocket:
                logger.warning("[CANDLES] WebSocket not connected; attempting reconnect before fetching candles")
                if not self.connect():
                    logger.error("[CANDLES] Reconnect failed; proceeding to HTTP fallback")
        except Exception as _:
            logger.error("[CANDLES] Exception during reconnect attempt; proceeding to HTTP fallback")
        # First try WebSocket approach
        ws_candles = self._get_candles_websocket(asset, duration, count, end_time)
        if ws_candles and len(ws_candles) >= min(30, count // 2):
            return ws_candles
        
        # Fallback to HTTP API for historical data
        logger.info(f"[CANDLES][HTTP] Falling back to HTTP API for {asset}")
        return self._get_candles_http(asset, duration, count, end_time)
    
    def _get_candles_websocket(self, asset: str, duration: int, count: int, end_time: Optional[float] = None) -> List[dict]:
        """Get candle data for an asset via WebSocket"""
        try:
            if not self.connected or not self.websocket:
                logger.error("❌ WebSocket not connected")
                return []
                
            if end_time is None:
                end_time = int(time.time())
            else:
                end_time = int(end_time)
            # Use seconds for timestamps
            end_time_s = int(end_time)
            from_time_s = int(end_time_s - (count * duration * 60))
            
            # Generate a unique request ID
            request_id = f"candles_{int(time.time() * 1000)}"
            
            # Prepare the request - try get-candles first as it's more reliable for historical data
            active_id = self._get_asset_id(asset)
            request = {
                "name": "sendMessage",
                "request_id": request_id,
                "msg": {
                    "name": "get-candles",
                    "version": "1.0",
                    "body": {
                        "active_id": active_id,
                        "size": duration * 60,  # Convert minutes to seconds
                        "to": end_time_s,
                        "count": count
                    }
                }
            }
            
            logger.info(f"[CANDLES][REQ] asset={asset} id={active_id} size={duration*60} to={end_time_s} count={count} req_id={request_id}")
            # Send the request
            self.websocket.send(json.dumps(request))
            
            # Wait for the response
            start_time = time.time()
            timeout = 15  # seconds
            
            while time.time() - start_time < timeout:
                if request_id in self.trade_responses:
                    response = self.trade_responses.pop(request_id)
                    msg = response.get('msg', {}) or {}
                elif 'candles_latest' in self.trade_responses:
                    response = self.trade_responses.pop('candles_latest')
                    msg = response.get('msg', {}) or {}
                    # Support multiple shapes: msg.candles, msg.result.candles, msg.data.candles
                    raw_candles = None
                    if isinstance(msg, dict):
                        if 'candles' in msg and isinstance(msg['candles'], list):
                            raw_candles = msg['candles']
                        elif isinstance(msg.get('result'), dict) and isinstance(msg['result'].get('candles'), list):
                            raw_candles = msg['result']['candles']
                        elif isinstance(msg.get('data'), dict) and isinstance(msg['data'].get('candles'), list):
                            raw_candles = msg['data']['candles']

                    if raw_candles is not None:
                        candles: List[dict] = []
                        for candle in raw_candles:
                            # Handle both 'max/min' and 'high/low' keys
                            high = candle.get('max', candle.get('high'))
                            low = candle.get('min', candle.get('low'))
                            ts = candle.get('from', candle.get('time', 0))
                            try:
                                candles.append({
                                    'open': float(candle['open']),
                                    'high': float(high),
                                    'low': float(low),
                                    'close': float(candle['close']),
                                    'volume': float(candle.get('volume', 0)),
                                    'time': int(ts)
                                })
                            except Exception:
                                # Skip malformed candle
                                continue
                        if candles:
                            logger.info(f"[CANDLES][OK] Received {len(candles)} candles for {asset}")
                            return candles
                        else:
                            logger.warning(f"[CANDLES] Empty candle list parsed for {asset}")
                            return []

                    # If we reach here, no candles were found in known fields; log diagnostics
                    top_keys = list(msg.keys()) if isinstance(msg, dict) else type(msg)
                    err_text = msg.get('message') if isinstance(msg, dict) else None
                    if err_text:
                        logger.warning(f"[CANDLES] No candle data for {asset}; keys={top_keys}; message={err_text}")
                    else:
                        logger.warning(f"[CANDLES] No candle data in response for {asset}; keys={top_keys}")
                    # Elevate response snippet to WARNING for visibility
                    try:
                        snippet = json.dumps(response)[:800]
                    except Exception:
                        snippet = str(response)[:800]
                    logger.warning(f"[CANDLES][RESP] {snippet}")
                    return []
                time.sleep(0.1)
            
            logger.warning(f"[CANDLES] Timeout waiting for candle data for {asset} using get-candles; trying fallback history-candles")
            # Probe active_id via subscription to detect routing/id mismatch
            try:
                probe_ok = self._probe_active_id(active_id, duration * 60)
                logger.info(f"[PROBE] Result for {asset} active_id={active_id}: {'OK' if probe_ok else 'FAILED'}")
            except Exception as e:
                logger.warning(f"[PROBE] Exception for {asset}: {e}")
                probe_ok = False
            # If probe fails, try discover a working active_id and retry once
            if not probe_ok:
                logger.info(f"[DISCOVER] Starting discovery for {asset} due to probe failure")
                try:
                    discovered_id = self._discover_active_id(asset, duration * 60)
                    if discovered_id and discovered_id != active_id:
                        logger.info(f"[DISCOVER] Using discovered active_id={discovered_id} for {asset}; retrying history-candles once")
                        active_id = discovered_id
                        self._save_active_id_map()
                        # Rebuild and resend history-candles once with new id
                        request_id_retry = f"candles_{int(time.time() * 1000)}"
                        request_retry = {
                            "name": "sendMessage",
                            "request_id": request_id_retry,
                            "msg": {
                                "name": "history-candles",
                                "version": "1.0",
                                "body": {
                                    "active_id": active_id,
                                    "size": duration * 60,
                                    "from": from_time_s,
                                    "to": end_time_s,
                                    "only_closed": True
                                },
                                "routingFilters": {
                                    "instrument_type": "turbo-option",
                                    "active_id": active_id,
                                    "size": duration * 60
                                }
                            }
                        }
                        logger.info(f"[CANDLES][RETRY][REQ] asset={asset} id={active_id} size={duration*60} from={from_time_s} to={end_time_s} req_id={request_id_retry}")
                        self.websocket.send(json.dumps(request_retry))
                        t0 = time.time()
                        while time.time() - t0 < 10:
                            if request_id_retry in self.trade_responses:
                                response = self.trade_responses.pop(request_id_retry)
                                msg = response.get('msg', {}) or {}
                                raw = None
                                if isinstance(msg, dict):
                                    if 'candles' in msg and isinstance(msg['candles'], list):
                                        raw = msg['candles']
                                    elif isinstance(msg.get('result'), dict) and isinstance(msg['result'].get('candles'), list):
                                        raw = msg['result']['candles']
                                    elif isinstance(msg.get('data'), dict) and isinstance(msg['data'].get('candles'), list):
                                        raw = msg['data']['candles']
                                if raw is not None:
                                    candles = []
                                    for c in raw:
                                        try:
                                            candles.append({
                                                'open': float(c['open']),
                                                'high': float(c.get('max', c.get('high'))),
                                                'low': float(c.get('min', c.get('low'))),
                                                'close': float(c['close']),
                                                'volume': float(c.get('volume', 0)),
                                                'time': int(c.get('from', c.get('time', 0)))
                                            })
                                        except Exception:
                                            continue
                                    if candles:
                                        logger.info(f"[CANDLES][RETRY][OK] Received {len(candles)} candles for {asset}")
                                        return candles
                                break
                            elif 'candles_latest' in self.trade_responses:
                                response = self.trade_responses.pop('candles_latest')
                                # Let normal fallback path handle single/latest forms
                                break
                            time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"[DISCOVER] Failed to discover active_id for {asset}: {e}")

            # Fallback: try history-candles with time range
            request_id_fb = f"candles_{int(time.time() * 1000)}"
            request_fb = {
                "name": "sendMessage",
                "request_id": request_id_fb,
                "msg": {
                    "name": "history-candles",
                    "version": "1.0",
                    "body": {
                        "active_id": active_id,
                        "size": duration * 60,
                        "from": from_time_s,
                        "to": end_time_s,
                        "only_closed": True
                    }
                }
            }
            logger.info(f"[CANDLES][FALLBACK][REQ] asset={asset} id={active_id} size={duration*60} to={end_time_s} count={count} req_id={request_id_fb}")
            self.websocket.send(json.dumps(request_fb))

            start_time = time.time()
            timeout = 12
            while time.time() - start_time < timeout:
                if request_id_fb in self.trade_responses:
                    response = self.trade_responses.pop(request_id_fb)
                    msg = response.get('msg', {}) or {}
                elif 'candles_latest' in self.trade_responses:
                    response = self.trade_responses.pop('candles_latest')
                    msg = response.get('msg', {}) or {}
                else:
                    time.sleep(0.1)
                    continue

                raw_candles = None
                if isinstance(msg, dict):
                    if 'candles' in msg and isinstance(msg['candles'], list):
                        raw_candles = msg['candles']
                    elif isinstance(msg.get('result'), dict) and isinstance(msg['result'].get('candles'), list):
                        raw_candles = msg['result']['candles']
                    elif isinstance(msg.get('data'), dict) and isinstance(msg['data'].get('candles'), list):
                        raw_candles = msg['data']['candles']

                if raw_candles is not None:
                    candles: List[dict] = []
                    for candle in raw_candles:
                        high = candle.get('max', candle.get('high'))
                        low = candle.get('min', candle.get('low'))
                        ts = candle.get('from', candle.get('time', 0))
                        try:
                            candles.append({
                                'open': float(candle['open']),
                                'high': float(high),
                                'low': float(low),
                                'close': float(candle['close']),
                                'volume': float(candle.get('volume', 0)),
                                'time': int(ts)
                            })
                        except Exception:
                            continue
                    if candles:
                        logger.info(f"[CANDLES][FALLBACK][OK] Received {len(candles)} candles for {asset}")
                        return candles
                    else:
                        logger.warning(f"[CANDLES][FALLBACK] Empty candle list parsed for {asset}")
                        return []

                top_keys = list(msg.keys()) if isinstance(msg, dict) else type(msg)
                # If server returned a single candle (e.g., candle-generated), it's not sufficient for analysis
                # We need historical data, not just live candles
                if isinstance(msg, dict):
                    candidate = None
                    if all(k in msg for k in ('open', 'close')):
                        candidate = msg
                    elif isinstance(msg.get('data'), dict) and all(k in msg['data'] for k in ('open', 'close')):
                        candidate = msg['data']
                    if candidate is not None:
                        logger.warning(f"[CANDLES][FALLBACK] Received single candle-generated event for {asset}, but need historical data")
                err_text = msg.get('message') if isinstance(msg, dict) else None
                if err_text:
                    logger.warning(f"[CANDLES][FALLBACK] No candle data for {asset}; keys={top_keys}; message={err_text}")
                else:
                    logger.warning(f"[CANDLES][FALLBACK] No candle data in response for {asset}; keys={top_keys}")
                try:
                    snippet = json.dumps(response)[:800]
                except Exception:
                    snippet = str(response)[:800]
                logger.warning(f"[CANDLES][FALLBACK][RESP] {snippet}")

                # Additional variants to probe server routing
                variants = [
                    {"name": "get-candles", "version": "1.0", "body": {"active_id": active_id, "size": duration*60, "to": end_time_s, "count": count}},
                    {"name": "history-candles", "version": "1.0", "body": {"active_id": active_id, "size": duration*60, "from": from_time_s, "to": end_time_s, "only_closed": True}},
                    {"name": "get-candles", "version": "2.0", "body": {"active_id": active_id, "size": duration*60, "to": end_time_s, "count": count}},
                ]
                for var in variants:
                    rid = f"candles_{int(time.time() * 1000)}"
                    req = {"name": "sendMessage", "request_id": rid, "msg": var}
                    logger.info(f"[CANDLES][ALT][REQ] {var['name']} v{var['version']} id={active_id} size={duration*60}")
                    self.websocket.send(json.dumps(req))
                    t0 = time.time()
                    while time.time() - t0 < 10:
                        if rid in self.trade_responses:
                            resp = self.trade_responses.pop(rid)
                            m = resp.get('msg', {}) or {}
                            raw = None
                            if isinstance(m, dict):
                                if 'candles' in m and isinstance(m['candles'], list):
                                    raw = m['candles']
                                elif isinstance(m.get('result'), dict) and isinstance(m['result'].get('candles'), list):
                                    raw = m['result']['candles']
                                elif isinstance(m.get('data'), dict) and isinstance(m['data'].get('candles'), list):
                                    raw = m['data']['candles']
                            if raw is None and isinstance(m, dict):
                                # Try single-candle shape
                                cand = None
                                if all(k in m for k in ('open', 'close')):
                                    cand = m
                                elif isinstance(m.get('data'), dict) and all(k in m['data'] for k in ('open', 'close')):
                                    cand = m['data']
                                if cand is not None:
                                    raw = [cand]
                            if raw is not None:
                                out = []
                                for c in raw:
                                    try:
                                        out.append({
                                            'open': float(c['open']),
                                            'high': float(c.get('max', c.get('high'))),
                                            'low': float(c.get('min', c.get('low'))),
                                            'close': float(c['close']),
                                            'volume': float(c.get('volume', 0)),
                                            'time': int(c.get('from', c.get('time', 0)))
                                        })
                                    except Exception:
                                        continue
                                if out:
                                    logger.info(f"[CANDLES][ALT][OK] Received {len(out)} candles for {asset}")
                                    return out
                            # log alt resp
                            try:
                                snip = json.dumps(resp)[:800]
                            except Exception:
                                snip = str(resp)[:800]
                            logger.warning(f"[CANDLES][ALT][RESP] {snip}")
                            break
                        elif 'candles_latest' in self.trade_responses:
                            resp = self.trade_responses.pop('candles_latest')
                            m = resp.get('msg', {}) or {}
                            raw = None
                            if isinstance(m, dict):
                                if 'candles' in m and isinstance(m['candles'], list):
                                    raw = m['candles']
                                elif isinstance(m.get('result'), dict) and isinstance(m['result'].get('candles'), list):
                                    raw = m['result']['candles']
                                elif isinstance(m.get('data'), dict) and isinstance(m['data'].get('candles'), list):
                                    raw = m['data']['candles']
                                if raw is None:
                                    cand = None
                                    if all(k in m for k in ('open', 'close')):
                                        cand = m
                                    elif isinstance(m.get('data'), dict) and all(k in m['data'] for k in ('open', 'close')):
                                        cand = m['data']
                                    if cand is not None:
                                        raw = [cand]
                            if raw is not None:
                                out = []
                                for c in raw:
                                    try:
                                        out.append({
                                            'open': float(c['open']),
                                            'high': float(c.get('max', c.get('high'))),
                                            'low': float(c.get('min', c.get('low'))),
                                            'close': float(c['close']),
                                            'volume': float(c.get('volume', 0)),
                                            'time': int(c.get('from', c.get('time', 0)))
                                        })
                                    except Exception:
                                        continue
                                if out:
                                    logger.info(f"[CANDLES][ALT-LATEST][OK] Received {len(out)} candles for {asset}")
                                    return out
                            try:
                                snip = json.dumps(resp)[:800]
                            except Exception:
                                snip = str(resp)[:800]
                            logger.warning(f"[CANDLES][ALT-LATEST][RESP] {snip}")
                            break
                        time.sleep(0.1)

                return []
            
        except Exception as e:
            logger.error(f"❌ Error getting candles for {asset}: {e}")
            return []
    
    def _get_candles_http(self, asset: str, duration: int, count: int, end_time: Optional[float] = None) -> List[dict]:
        """Get historical candles via HTTP API as fallback"""
        try:
            if end_time is None:
                end_time = int(time.time())
            else:
                end_time = int(end_time)
            
            from_time = end_time - (count * duration * 60)
            active_id = self._get_asset_id(asset)
            
            # Use IQ Option HTTP API for historical data
            url = "https://iqoption.com/api/candles/get-candles"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "active_id": active_id,
                "size": duration * 60,
                "from": from_time,
                "to": end_time
            }
            
            logger.info(f"[CANDLES][HTTP][REQ] {asset} id={active_id} from={from_time} to={end_time}")
            
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'candles' in data:
                    raw_candles = data['candles']
                elif isinstance(data, list):
                    raw_candles = data
                else:
                    logger.warning(f"[CANDLES][HTTP] Unexpected response format for {asset}")
                    return []
                
                candles = []
                for candle in raw_candles:
                    try:
                        candles.append({
                            'open': float(candle['open']),
                            'high': float(candle.get('max', candle.get('high', candle['open']))),
                            'low': float(candle.get('min', candle.get('low', candle['open']))),
                            'close': float(candle['close']),
                            'volume': float(candle.get('volume', 0)),
                            'time': int(candle.get('from', candle.get('time', 0)))
                        })
                    except (KeyError, ValueError, TypeError):
                        continue
                
                if candles:
                    logger.info(f"[CANDLES][HTTP][OK] Retrieved {len(candles)} candles for {asset}")
                    return candles
                else:
                    logger.warning(f"[CANDLES][HTTP] No valid candles parsed for {asset}")
                    return []
            else:
                logger.warning(f"[CANDLES][HTTP] HTTP {response.status_code} for {asset}")
                return []
                
        except Exception as e:
            logger.error(f"[CANDLES][HTTP] Error for {asset}: {e}")
            return []
    
    def is_asset_tradable(self, asset: str) -> bool:
        """Check if an asset is currently available for trading"""
        try:
            # First check if we're connected
            if not self.connected or not self.websocket:
                logger.warning("⚠️ Not connected to check asset status")
                return False
                
            # Get the active ID for the asset
            active_id = self._get_asset_id(asset)
            if not active_id:
                logger.warning(f"⚠️ Could not find active ID for {asset}")
                return False
                
            # Prepare the request to check asset status
            request_id = f"asset_status_{int(time.time() * 1000)}"
            request = {
                "name": "sendMessage",
                "request_id": request_id,
                "msg": {
                    "name": "get-asset-info",
                    "version": "1.0",
                    "body": {
                        "active_id": active_id
                    }
                }
            }
            
            # Send the request
            self.websocket.send(json.dumps(request))
            
            # Wait for the response
            start_time = time.time()
            timeout = 5  # seconds
            
            while time.time() - start_time < timeout:
                if request_id in self.trade_responses:
                    response = self.trade_responses.pop(request_id)
                    # Check if the asset is enabled for trading
                    is_enabled = response.get('msg', {}).get('enabled', False)
                    if is_enabled:
                        return True
                    else:
                        logger.warning(f"⚠️ Asset {asset} is currently suspended")
                        return False
                time.sleep(0.1)
                
            logger.warning(f"⚠️ Timeout checking status for {asset}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking asset status for {asset}: {e}")
            return False
            
    def _get_asset_id(self, asset: str) -> int:
        """Get asset ID for API calls - BLITZ/DIGITAL OPTIONS mapping"""
        # BLITZ/DIGITAL asset ID mapping (1-76 range)
        asset_ids = {
            # Forex Digital/Blitz Pairs (1-30)
            'EURUSD-OTC': 1, 'GBPUSD-OTC': 2, 'USDJPY-OTC': 3, 'USDCHF-OTC': 4,
            'AUDUSD-OTC': 5, 'NZDUSD-OTC': 6, 'USDCAD-OTC': 7, 'EURGBP-OTC': 8,
            'EURJPY-OTC': 9, 'GBPJPY-OTC': 10, 'AUDCAD-OTC': 11, 'AUDCHF-OTC': 12,
            'AUDJPY-OTC': 13, 'AUDNZD-OTC': 14, 'CADCHF-OTC': 15, 'CADJPY-OTC': 16,
            'CHFJPY-OTC': 17, 'EURAUD-OTC': 18, 'EURCAD-OTC': 19, 'EURCHF-OTC': 20,
            'EURNZD-OTC': 21, 'GBPAUD-OTC': 22, 'GBPCAD-OTC': 23, 'GBPCHF-OTC': 24,
            'GBPNZD-OTC': 25, 'NZDCAD-OTC': 26, 'NZDCHF-OTC': 27, 'NZDJPY-OTC': 28,
            
            # Cryptocurrency Digital/Blitz Pairs (31-45)
            'BTCUSD-OTC': 31, 'ETHUSD-OTC': 32, 'LTCUSD-OTC': 33, 'XRPUSD-OTC': 34,
            'BCHUSD-OTC': 35, 'XLMUSD-OTC': 36, 'EOSUSD-OTC': 37, 'TRXUSD-OTC': 38,
            
            # Stock Indices Digital/Blitz (46-60)
            'US500-OTC': 46, 'US30-OTC': 47, 'USTEC-OTC': 48,
            'UK100-OTC': 49, 'GER30-OTC': 50, 'FRA40-OTC': 51,
            'JPN225-OTC': 52, 'AUS200-OTC': 53, 'ESP35-OTC': 54,
            'EUSTX50-OTC': 55, 'HKG33-OTC': 56
        }
        # Prefer dynamically discovered map first (but only if it's a valid Blitz ID within 1..76)
        a = asset.upper()
        if a in self.dynamic_active_ids:
            try:
                did = int(self.dynamic_active_ids[a])
                if 1 <= did <= 76:
                    return did
                else:
                    logger.warning(f"[ACTIVE-ID] Ignoring previously discovered non-Blitz id {did} for {a}; using Blitz mapping instead")
            except Exception:
                pass
        # Default to BLITZ mapping or EURUSD if not found
        return asset_ids.get(a, 1)

    # ---- Candle subscription utilities for probing active_id routing ----
    def _subscribe_candles(self, active_id: int, size: int) -> None:
        try:
            sub = {
                "name": "subscribeMessage",
                "msg": {
                    "name": "candle-generated",
                    "params": {
                        "routingFilters": {
                            "active_id": int(active_id),
                            "size": int(size)
                        }
                    }
                }
            }
            self.websocket.send(json.dumps(sub))
            logger.info(f"[SUBSCRIBE] candle-generated active_id={active_id} size={size}")
        except Exception as e:
            logger.warning(f"[SUBSCRIBE] Failed: {e}")

    def _unsubscribe_candles(self, active_id: int, size: int) -> None:
        try:
            unsub = {
                "name": "unsubscribeMessage",
                "msg": {
                    "name": "candle-generated",
                    "params": {
                        "routingFilters": {
                            "active_id": int(active_id),
                            "size": int(size)
                        }
                    }
                }
            }
            self.websocket.send(json.dumps(unsub))
            logger.info(f"[UNSUBSCRIBE] candle-generated active_id={active_id} size={size}")
        except Exception as e:
            logger.warning(f"[UNSUBSCRIBE] Failed: {e}")

    def _probe_active_id(self, active_id: int, size: int, wait_s: float = 5.0) -> bool:
        key = f"{active_id}:{size}"
        self.last_candle_events.pop(key, None)
        self._subscribe_candles(active_id, size)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if key in self.last_candle_events:
                logger.info(f"[PROBE][OK] candle-generated events received for active_id={active_id} size={size}")
                self._unsubscribe_candles(active_id, size)
                return True
            time.sleep(0.1)
        logger.warning(f"[PROBE][MISS] No candle-generated events for active_id={active_id} size={size} within {wait_s}s")
        self._unsubscribe_candles(active_id, size)
        return False

    # ---- Dynamic active_id discovery & persistence ----
    def _load_active_id_map(self) -> None:
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'active_id_map.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.dynamic_active_ids = json.load(f)
                # Sanitize: keep only Blitz/Digital IDs in 1..76
                try:
                    before = len(self.dynamic_active_ids)
                    self.dynamic_active_ids = {
                        str(k).upper(): int(v)
                        for k, v in self.dynamic_active_ids.items()
                        if isinstance(v, (int, float)) and 1 <= int(v) <= 76
                    }
                    after = len(self.dynamic_active_ids)
                    if after < before:
                        logger.warning(f"[ACTIVE-ID] Filtered non-Blitz IDs from cache: {before-after} removed")
                except Exception:
                    pass
                logger.info(f"[ACTIVE-ID] Loaded {len(self.dynamic_active_ids)} discovered mappings from {path}")
        except Exception as e:
            logger.warning(f"[ACTIVE-ID] Failed to load active id map: {e}")

    def _save_active_id_map(self) -> None:
        try:
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, 'active_id_map.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic_active_ids, f, ensure_ascii=False, indent=2)
            logger.info(f"[ACTIVE-ID] Saved mappings to {path}")
        except Exception as e:
            logger.warning(f"[ACTIVE-ID] Failed to save active id map: {e}")

    def _discover_active_id(self, asset: str, size: int) -> Optional[int]:
        try:
            a = asset.upper()
            base_id = int(self._get_asset_id(a))  # Blitz mapping base (1..76)
            candidates: List[int] = []
            for delta in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
                cid = base_id + delta
                if 1 <= cid <= 76 and cid not in candidates:
                    candidates.append(cid)

            for cid in candidates:
                if self._probe_active_id(cid, size, wait_s=3.0):
                    self.dynamic_active_ids[a] = int(cid)
                    logger.info(f"[DISCOVER][OK] {a} -> active_id={cid}")
                    return int(cid)
            logger.warning(f"[DISCOVER][MISS] Could not find working Blitz active_id for {a} in candidate set")
            return None
        except Exception as e:
            logger.warning(f"[DISCOVER] Error discovering active_id for {asset}: {e}")
            return None

    def fetch_account_balances_v2(self) -> List[dict]:
        """Fetch all account balances with detailed information"""
        self.balance_data = None
        request = {
            "name": "sendMessage",
            "msg": {
                "name": "internal-billing.get-balances",
                "version": "1.0",
                "body": {
                    "types_ids": [1, 4, 2, 6],
                    "tournaments_statuses_ids": [3, 2]
                }
            }
        }
        self.websocket.send(json.dumps(request))
        
        # Wait for balance data
        timeout = 5
        start_time = time.time()
        while self.balance_data is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        return self.balance_data or []

    def switch_account(self, account_type: str) -> None:
        """Switch between real and demo accounts"""
        account_type = account_type.lower()
        if account_type not in ('real', 'demo'):
            raise ValueError("Account type must be either 'real' or 'demo'")

        accounts = self.fetch_account_balances_v2()
        account_id = self._find_account_id(accounts, account_type)
        
        if account_id:
            self._set_active_account(account_id)
            logger.info(f"✅ Switched to {account_type} account (ID: {account_id})")
        else:
            raise ValueError(f"{account_type.capitalize()} account not found")

    def _find_account_id(self, accounts: List[dict], account_type: str) -> Optional[int]:
        """Find account ID by type"""
        target_type = self.ACCOUNT_REAL if account_type == 'real' else self.ACCOUNT_DEMO
        for account in accounts:
            if account['type'] == target_type:
                return account['id']
        return None

    def _set_active_account(self, account_id: int) -> None:
        """Set the active account"""
        self.active_balance_id = account_id

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages"""
        try:
            # Log raw message for debugging
            if isinstance(message, str):
                logger.debug(f"[WS] Received message: {message[:500]}")
            else:
                logger.debug("[WS] Received binary message")
            
            try:
                message_data = json.loads(message) if isinstance(message, str) else message
                if not isinstance(message_data, dict):
                    logger.debug(f"Unexpected message format: {message_data}")
                    return
            except (json.JSONDecodeError, TypeError):
                # Handle non-JSON messages (like binary data or plain text)
                logger.debug(f"Received non-JSON message: {str(message)[:200]}...")
                return
                
            # Handle different message types
            msg_name = message_data.get('name', '')
            inner = message_data.get('msg', {}) if isinstance(message_data, dict) else {}
            inner_name = inner.get('name') if isinstance(inner, dict) else None
            
            if msg_name == 'timeSync':
                logger.debug("[TIME] Time sync received")
                return
                
            elif msg_name == 'profile':
                logger.debug("[PROFILE] Data received")
                self._handle_profile_message(message_data)
                return
                
            elif msg_name == 'balances':
                logger.debug("[BALANCE] Data received")
                self._handle_balance_message(message_data)
                return
                
            elif msg_name in ['candles', 'candle-generated'] or inner_name in ['candles', 'candle-generated', 'get-candles', 'history-candles']:
                # Handle candle data responses
                request_id = str(message_data.get('request_id', '') or '')
                if request_id and request_id.startswith('candles_'):
                    logger.debug(f"[CANDLES] Data received for request {request_id}")
                    self.trade_responses[request_id] = message_data
                    return
                # Some servers omit request_id or wrap candles within msg/result/data
                # Store as latest if candles-like content is present
                try:
                    container = message_data
                    # If wrapped, prefer inner
                    if isinstance(inner, dict) and any(k in inner for k in ('candles', 'result', 'data')):
                        container = {'msg': inner, 'request_id': message_data.get('request_id')}
                    self.trade_responses['candles_latest'] = container
                    logger.debug("[CANDLES] Data received (no/unknown request_id); stored as candles_latest")
                except Exception:
                    self.trade_responses['candles_latest'] = message_data
                # Also capture candle-generated events for probe
                try:
                    if (msg_name == 'candle-generated') or (inner_name == 'candle-generated'):
                        # Try to extract active_id and size from multiple possible locations
                        aid = None
                        sz = None
                        # Direct fields
                        if isinstance(message_data, dict):
                            aid = message_data.get('active_id') or aid
                            sz = message_data.get('size') or sz
                        if isinstance(inner, dict):
                            aid = inner.get('active_id') or aid
                            sz = inner.get('size') or sz
                            params = inner.get('params') or {}
                            rf = params.get('routingFilters') if isinstance(params, dict) else {}
                            if isinstance(rf, dict):
                                aid = rf.get('active_id') or aid
                                sz = rf.get('size') or sz
                        # Sometimes nested under 'data'
                        data = inner.get('data') if isinstance(inner, dict) else None
                        if isinstance(data, dict):
                            aid = data.get('active_id') or aid
                            sz = data.get('size') or sz
                        if aid and sz:
                            key = f"{int(aid)}:{int(sz)}"
                            self.last_candle_events[key] = message_data
                            logger.info(f"[CANDLE][EVENT] candle-generated captured for active_id={aid} size={sz}")
                except Exception:
                    pass
                return
                
            elif msg_name in ['option', 'option-closed', 'option-changed']:
                # Correlate response by request_id when available so place_binary_trade() can resolve
                req_id = str(message_data.get('request_id', ''))
                if req_id:
                    self.trade_responses[req_id] = message_data
                self._handle_trade_response(message_data)
                
                # Check for suspension or trading errors
                if 'msg' in message_data and 'message' in message_data['msg']:
                    msg_text = str(message_data['msg']['message']).lower()
                    if 'active is suspended' in msg_text or 'cannot purchase' in msg_text:
                        # Try to extract asset name from the request
                        request_id = message_data.get('request_id', '')
                        if isinstance(request_id, str) and request_id.startswith('trade_'):
                            parts = request_id.split('_')
                            if len(parts) > 1:
                                asset_name = parts[1]
                                logger.warning(f"⚠️ Asset {asset_name} is currently suspended or cannot be traded")
                                
                                # Mark asset as suspended in our cache
                                active_id = self._get_asset_id(asset_name)
                                if active_id:
                                    if active_id not in getattr(self, 'assets_info', {}):
                                        # Lazily create assets_info if missing
                                        if not hasattr(self, 'assets_info'):
                                            self.assets_info = {}
                                        self.assets_info[active_id] = {}
                                    self.assets_info[active_id]['suspended'] = True
                                    logger.debug(f"Marked {asset_name} (ID: {active_id}) as suspended")
                return
                
            # Handle responses by request_id
            elif 'request_id' in message_data:
                request_id = str(message_data['request_id'])
                logger.debug(f"[RESPONSE] For request {request_id}")
                
                # Handle asset info responses
                if request_id.startswith('asset_status_'):
                    logger.debug(f"[ASSET STATUS] Response for {request_id}")
                    self.trade_responses[request_id] = message_data
                    return
                    
                # Handle other responses
                self.trade_responses[request_id] = message_data
                return
                
            else:
                # Generic catch: if payload contains candles list anywhere, store as latest
                try:
                    def find_candles(d):
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if k == 'candles' and isinstance(v, list):
                                    return True
                                if find_candles(v):
                                    return True
                        elif isinstance(d, list):
                            for item in d:
                                if find_candles(item):
                                    return True
                        return False
                    if find_candles(message_data):
                        self.trade_responses['candles_latest'] = message_data
                        logger.info("[CANDLES][CAPTURE] Captured candles from unrecognized message type")
                        return
                except Exception:
                    pass
                logger.debug(f"[UNHANDLED] Message type: {message_data.get('name')}")
                return
                
            # Log unknown message format
            logger.debug(f"Unknown message format: {type(message_data)}")

        except Exception as e:
            logger.error(f"❌ Error handling WebSocket message: {e}")
            logger.debug(f"Message that caused error: {message[:500]}")

    def _handle_profile_message(self, message: dict) -> None:
        """Process profile information"""
        self.profile_msg = message
        self._update_active_balance(message['msg']['balances'])

    def _update_active_balance(self, balances: List[dict]) -> None:
        """Update the active balance ID from profile data"""
        target_type = self.ACCOUNT_DEMO if self.account_mode == 'PRACTICE' else self.ACCOUNT_REAL
        for balance in balances:
            if balance['type'] == target_type:
                self.active_balance_id = balance['id']
                break

    def _handle_balance_message(self, message: dict) -> None:
        """Process balance information"""
        self.balance_data = message['msg']

    def _handle_trade_response(self, response_data: dict) -> None:
        """Handle trade response and update trade status"""
        try:
            if not isinstance(response_data, dict):
                return
                
            # Check for trade result in the response
            if 'msg' in response_data and 'result' in response_data['msg']:
                result = response_data['msg']['result']
                if 'id' in result:
                    trade_id = str(result['id'])
                    logger.info(f"✅ Trade {trade_id} executed successfully")
                    
                    # Update trade status if we're tracking this trade
                    if hasattr(self, 'active_trades') and trade_id in self.active_trades:
                        self.active_trades[trade_id]['status'] = 'executed'
                        self.active_trades[trade_id]['result'] = result
                        
            # Handle error responses
            elif 'msg' in response_data and 'message' in response_data['msg']:
                error_msg = response_data['msg']['message']
                logger.error(f"❌ Trade failed: {error_msg}")
                
                # Check for asset suspension
                if 'active is suspended' in str(error_msg).lower() or 'cannot purchase' in str(error_msg).lower():
                    # The main message handler will handle suspension marking
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing trade response: {e}")

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket errors"""
        logger.error(f"❌ WebSocket error: {error}")
        self.ws_is_active = False
        self.connected = False
        
        # Attempt to reconnect if this was an active connection
        if self.connected:
            logger.info("🔄 Attempting to reconnect...")
            try:
                self.connect()
            except Exception as e:
                logger.error(f"❌ Reconnection failed: {e}")

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opening"""
        logger.info("🔗 WebSocket connection established")
        self.ws_is_active = True
        
        # Send initial handshake message if needed
        try:
            handshake = {
                'name': 'timeSync',
                'msg': int(time.time() * 1000),
                'request_id': f'time_{int(time.time())}'
            }
            ws.send(json.dumps(handshake))
        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket handshake: {e}")

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket connection closing"""
        logger.info(f"🔌 WebSocket connection closed (code: {close_status_code}): {close_msg}")
        self.ws_is_active = False
        self.connected = False
        
        # Reset state for potential reconnection
        self.profile_msg = None
        self.balance_data = None

    def close(self) -> None:
        """Close the WebSocket connection"""
        if self.websocket:
            self.websocket.close()
        self.connected = False

    def check_connect(self) -> bool:
        """Check if connected"""
        return self.connected and self.ws_is_active
