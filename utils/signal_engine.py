# signal_engine.py - ULTIMATE SIGNAL FUSION ENGINE

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from features.enhanced_ta_system import enhanced_ta

# Import dashboard integration
try:
    from utils.dashboard_integration import send_signal_to_dashboard
except ImportError:
    def send_signal_to_dashboard(*args, **kwargs):
        pass  # Fallback if dashboard not available

from features.ultimate_signal_fusion import ultimate_fusion
from features.practical_balanced_signals import PracticalBalancedSignals
from features.ultimate_signal_fusion import UltimateSignalFusion
from features.simple_signal_generator import SimpleSignalGenerator
from features.ml_signal_generator import MLSignalGenerator
from features.intelligent_martingale import IntelligentMartingale
from features.blitz_optimizer import blitz_optimizer
from utils.adaptive_ensemble_learning import adaptive_ensemble_learner
try:
    from config.settings import TEACHER_MODE, TEACHER_CONF_MIN, USE_BLITZ, BLITZ_DURATION_SECONDS
except Exception:
    TEACHER_MODE = False
    TEACHER_CONF_MIN = 80
    USE_BLITZ = True
    BLITZ_DURATION_SECONDS = 30

logger = logging.getLogger(__name__)

# Global instances for signal generation
practical_signals = PracticalBalancedSignals()
ultimate_fusion = UltimateSignalFusion()
ml_generator = MLSignalGenerator()

# Global variables for confidence tracking
last_confidence_score = 85
last_signal_details = {}

def predict_signal(asset: str, candles: List[Dict]) -> Optional[Tuple[str, float]]:
    """Get balanced trading signal with PUT bias fix.
    Returns 'buy', 'sell', or None with confidence score (0-100).
    """
    global last_signal_details
    
    try:
        # PRIORITY 1: Use optimized ML signal generator first (93.1% accuracy)
        logger.info(f"🤖 Trying optimized ML signal generator for {asset}")
        
        # Initialize default values
        ml_result = None
        
        try:
            ml_result = ml_generator.generate_ml_signal(asset, candles)
        except Exception as e:
            logger.warning(f"⚠️ ML signal generation failed for {asset}: {e}")
            ml_result = None
        
        if ml_result and isinstance(ml_result, dict) and ml_result.get('signal'):
            try:
                direction = 'buy' if str(ml_result['signal']).lower() in ['call', 'buy'] else 'sell'
                confidence = float(ml_result.get('confidence', 0))
                
                if confidence >= 55:  # More aggressive ML signal threshold
                    logger.info(f"✅ ML signal accepted for {asset}: {direction.upper()} ({confidence:.1f}%)")
                    last_signal_details = {
                        'source': 'ML_Generator',
                        'confidence': confidence,
                        'details': ml_result
                    }
                    
                    # Send ML signal to dashboard
                    send_signal_to_dashboard(
                        pair=asset,
                        direction='call' if direction == 'buy' else 'put',
                        confidence=confidence,
                        ml_data={'prediction': direction, 'confidence': confidence},
                        ta_data={},
                        ensemble_data={},
                        executed=False
                    )
                    
                    return direction, confidence
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"⚠️ Error processing ML result for {asset}: {e}")
                # Continue to fallback methods
            
            last_signal_details = {
                'asset': asset,
                'fusion_result': ml_result,
                'dynamic_threshold': confidence,
                'decision_source': 'ML_Generator_Primary',
                'strength_category': 'ml_predicted',
                'bet_size_multiplier': 1.2,  # Higher multiplier for ML signals
                'entry_timing': 'immediate',
                'analysis': ml_result.get('analysis', {})
            }
            
            logger.info(
                f"🚀 ML PRIMARY SIGNAL for {asset}: {direction.upper()} "
                f"(Confidence: {confidence:.1f}%, Model: {ml_result.get('model_type', 'Unknown')}, "
                f"Accuracy: 93.1%)"
            )
            
            return direction, confidence
        
        # PRIORITY 2: Practical balanced signals as backup
        logger.info(f"📊 Trying practical balanced signals for {asset}")
        
        try:
            balanced_result = practical_signals.generate_balanced_signal(asset, candles)
            
            if balanced_result and isinstance(balanced_result, dict) and balanced_result.get('signal'):
                try:
                    direction = 'buy' if str(balanced_result['signal']).lower() in ['call', 'buy'] else 'sell'
                    confidence = float(balanced_result.get('confidence', 0))
                    
                    if confidence >= 50:  # More aggressive practical signal threshold
                        logger.info(f"✅ Practical signal accepted for {asset}: {direction.upper()} ({confidence:.1f}%)")
                        last_signal_details = {
                            'source': 'Practical_Balanced',
                            'confidence': confidence,
                            'details': balanced_result
                        }
                        
                        # Send practical signal to dashboard
                        send_signal_to_dashboard(
                            pair=asset,
                            direction='call' if direction == 'buy' else 'put',
                            confidence=confidence,
                            ml_data={},
                            ta_data={'score': confidence, 'details': balanced_result.get('analysis', {})},
                            ensemble_data={},
                            executed=False
                        )
                        
                        # Send to Telegram
                        try:
                            from utils.telegram_notifier import TelegramNotifier
                            telegram = TelegramNotifier()
                            
                            signal_emoji = "📈" if direction == 'buy' else "📉"
                            direction_label = "CALL (BUY)" if direction == 'buy' else "PUT (SELL)"
                            
                            message = (
                                f"{signal_emoji} *INTELLIGIT SIGNAL*\n"
                                f"📍 *Pair:* `{asset}`\n"
                                f"🔁 *Direction:* *{direction_label}*\n"
                                f"📊 *Confidence:* `{confidence:.1f}%`\n"
                                f"🕐 *Duration:* `3 minutes`\n"
                                f"🎯 *Source:* Practical Balanced Signals\n"
                                f"🚀 *Entry:* NOW"
                            )
                            
                            telegram.send_telegram_signal(message)
                            logger.info(f"📱 Signal sent to Telegram: {asset} {direction.upper()}")
                            
                        except Exception as telegram_err:
                            logger.error(f"Failed to send Telegram signal: {telegram_err}")
                        
                        return direction, confidence
                        
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Error processing practical signal for {asset}: {e}")
                    # Continue to next fallback
            else:
                logger.warning(f"⚠️ No valid signal from practical balanced system for {asset}")
                
        except Exception as e:
            logger.error(f"❌ Error in practical balanced signal generation for {asset}: {e}")
            # Continue to next fallback
            
            last_signal_details = {
                'asset': asset,
                'fusion_result': balanced_result,
                'dynamic_threshold': confidence,
                'decision_source': 'Practical_Balanced_Backup',
                'strength_category': balanced_result.get('strength_category', 'medium'),
                'bet_size_multiplier': balanced_result.get('bet_size_multiplier', 1.0),
                'entry_timing': balanced_result.get('entry_timing', 'immediate'),
                'analysis': balanced_result.get('analysis', {})
            }
            
            logger.info(
                f"✅ BALANCED BACKUP SIGNAL for {asset}: {direction.upper()} "
                f"(Confidence: {confidence:.1f}%, Category: {balanced_result.get('strength_category', 'N/A')}, "
                f"Multiplier: {balanced_result.get('bet_size_multiplier', 1.0):.1f}x)"
            )
            
            return direction, confidence
        
        # PRIORITY 3: Ultimate Fusion fallback with enhanced error handling
        logger.info(f"🔄 Falling back to Ultimate Fusion for {asset}")
        
        try:
            fusion_result = ultimate_fusion.generate_ultimate_signal(asset, candles)
            
            # Safely extract confidence with validation
            conf_raw = fusion_result.get('confidence', 0) if isinstance(fusion_result, dict) else 0
            try:
                conf_val = max(0, min(100, float(conf_raw)))  # Ensure confidence is between 0-100
            except (ValueError, TypeError):
                conf_val = 0.0
            
            # Validate signal and confidence
            if (isinstance(fusion_result, dict) and 
                fusion_result.get('signal') and 
                str(fusion_result['signal']).lower() in ['call', 'put', 'buy', 'sell'] and 
                conf_val >= 65):  # Moderate confidence fusion signal
                
                direction = 'buy' if str(fusion_result['signal']).lower() in ['call', 'buy'] else 'sell'
                confidence = conf_val
                
                logger.info(f"✅ Ultimate Fusion signal accepted for {asset}: {direction.upper()} ({confidence:.1f}%)")
                last_signal_details = {
                    'source': 'Ultimate_Fusion',
                    'confidence': confidence,
                    'details': fusion_result
                }
                
                # Send fusion signal to dashboard
                try:
                    send_signal_to_dashboard(
                        pair=asset,
                        direction='call' if direction == 'buy' else 'put',
                        confidence=confidence,
                        ml_data={},
                        ta_data={},
                        ensemble_data={'prediction': direction, 'confidence': confidence},
                        executed=False
                    )
                    
                    # Send to Telegram
                    try:
                        from utils.telegram_notifier import TelegramNotifier
                        telegram = TelegramNotifier()
                        
                        signal_emoji = "📈" if direction == 'buy' else "📉"
                        direction_label = "CALL (BUY)" if direction == 'buy' else "PUT (SELL)"
                        
                        message = (
                            f"{signal_emoji} *INTELLIGIT ULTIMATE SIGNAL*\n"
                            f"📍 *Pair:* `{asset}`\n"
                            f"🔁 *Direction:* *{direction_label}*\n"
                            f"📊 *Confidence:* `{confidence:.1f}%`\n"
                            f"🕐 *Duration:* `3 minutes`\n"
                            f"🎯 *Source:* Ultimate Fusion Engine\n"
                            f"🚀 *Entry:* NOW"
                        )
                        
                        telegram.send_telegram_signal(message)
                        logger.info(f"📱 Ultimate signal sent to Telegram: {asset} {direction.upper()}")
                        
                    except Exception as telegram_err:
                        logger.error(f"Failed to send Ultimate Telegram signal: {telegram_err}")
                        
                except Exception as e:
                    logger.error(f"⚠️ Failed to send Ultimate Fusion signal to dashboard: {e}")
                
                return direction, confidence
                
        except Exception as e:
            logger.error(f"❌ Error in Ultimate Fusion fallback for {asset}: {e}", exc_info=True)
            # Continue to next fallback method
            
            last_signal_details = {
                'asset': asset,
                'fusion_result': fusion_result,
                'dynamic_threshold': confidence,
                'decision_source': 'Ultimate_Fusion_Fallback',
                'strength_category': fusion_result.get('strength_category', 'medium'),
                'bet_size_multiplier': fusion_result.get('bet_size_multiplier', 1.0),
                'entry_timing': fusion_result.get('entry_timing', 'immediate'),
                'analysis': fusion_result.get('analysis', {})
            }
            
            logger.info(
                f"🚀 FALLBACK SIGNAL for {asset}: {direction.upper()} "
                f"(Confidence: {confidence:.1f}%, Category: {fusion_result.get('strength_category', 'N/A')})"
            )
            
            return direction, confidence
            
    except Exception as e:
        logger.error(f"Signal generation failed for {asset}: {e}")
        
    # Final fallback: Simple signal generator
    try:
        logger.info(f"🎯 Trying simple signal generator for {asset}")
        simple_generator = SimpleSignalGenerator()
        simple_result = simple_generator.generate_signal(asset, candles)
        
        if simple_result and simple_result.get('signal'):
            signal = simple_result['signal']
            confidence = simple_result.get('confidence', 65)
            
            # Convert to standard format
            direction = 'buy' if signal == 'call' else 'sell'
            
            # Store signal details
            last_signal_details[asset] = {
                'signal': direction,
                'confidence': confidence,
                'dynamic_threshold': confidence,
                'decision_source': 'Simple_Generator_Fallback',
                'strength_category': 'basic',
                'bet_size_multiplier': 1.0,
                'entry_timing': 'immediate',
                'analysis': simple_result.get('analysis', {})
            }
            
            logger.info(
                f"🚀 SIMPLE SIGNAL for {asset}: {direction.upper()} "
                f"(Confidence: {confidence:.1f}%, Source: Simple Generator)"
            )
            
            return direction, confidence
            
    except Exception as e:
        logger.error(f"Simple signal generation failed for {asset}: {e}")
    
    logger.info(f"📊 No signal generated for {asset} from any source")
    return None, 0.0

def get_signal_confidence_info(asset: Optional[str] = None) -> Dict:
    """
    Get the confidence information from the last generated signal.
    
    Args:
        asset: Optional asset name to filter the signal details
        
    Returns:
        Dictionary with confidence score and signal details
    """
    global last_confidence_score, last_signal_details
    
    # If an asset is specified and it doesn't match the last signal's asset, return empty
    if asset and last_signal_details.get('asset') != asset:
        return {
            'confidence_score': 0,
            'signal_details': {},
            'timestamp': datetime.now().isoformat()
        }
    
    # Backward/forward compatible fields for other modules
    details = last_signal_details.copy()
    # Ensure numeric outputs
    try:
        conf_out = float(last_confidence_score)
    except Exception:
        conf_out = 0.0
    dyn_raw = details.get('dynamic_threshold', conf_out)
    try:
        dyn_out = float(dyn_raw)
    except Exception:
        dyn_out = conf_out
    acc_raw = details.get('asset_accuracy', 0.0)
    try:
        acc_out = float(acc_raw)
    except Exception:
        acc_out = 0.0
    return {
        'confidence_score': conf_out,
        'signal_details': details,
        # commonly expected aliases in other modules
        'dynamic_threshold': dyn_out,
        'asset_accuracy': acc_out,
        'timestamp': datetime.now().isoformat()
    }

def determine_best_entry_time(candle_data: List[Dict], signal_type: str = None) -> str:
    """
    Determine a precise entry time (hh:mm:ss) aligned to the next candle open with a
    small volatility-aware buffer. If immediate-entry conditions are met, return a
    near-immediate timestamp with a slight latency offset.

    Returns a local-time string formatted as HH:MM:SS.
    """
    from datetime import datetime, timedelta

    # Helper: compute a safe precise timestamp string
    def _fmt(dt: datetime) -> str:
        return dt.strftime('%H:%M:%S')

    now = datetime.now()

    # If very little data, default to next minute + small buffer
    if not candle_data or len(candle_data) < 5:
        next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return _fmt(next_minute + timedelta(seconds=10))

    try:
        # Initialize Smart Money Strategy for precise context (order blocks / FVG)
        sms = SmartMoneyStrategy()
        analysis = sms.analyze_order_flow(candle_data)

        # Current price and last candle
        last = candle_data[-1]
        current_price = float(last.get('close', last.get('price', 0)))

        # Estimate volatility using recent ranges (prefer high/low, fallback to |close-open|)
        def _rng(c):
            try:
                hi = float(c.get('max', c.get('high', c.get('close', 0))))
                lo = float(c.get('min', c.get('low', c.get('open', 0))))
                return max(0.0, hi - lo)
            except Exception:
                oc = abs(float(c.get('close', 0)) - float(c.get('open', 0)))
                return max(0.0, oc)

        recent = candle_data[-20:]
        ranges = [_rng(c) for c in recent]
        atr = float(np.mean(ranges[-14:])) if len(ranges) >= 14 else (float(np.mean(ranges)) if ranges else 0.0)
        curr_rng = _rng(last)
        vf = (curr_rng / (atr + 1e-8)) if atr > 0 else 1.0  # volatility factor

        # Volatility-aware buffer seconds at next candle open
        if vf < 1.2:
            buffer_sec = 5
        elif vf < 1.8:
            buffer_sec = 12
        else:
            buffer_sec = 20

        # Slight latency safety for immediate entries
        latency_sec = 2

        # Immediate entry conditions using SMC context
        if signal_type == 'buy':
            for block in analysis.get('order_blocks', []):
                try:
                    if block.get('type') == 'bullish' and float(block['low']) <= current_price <= float(block['high']) * 1.002:
                        return _fmt(now + timedelta(seconds=latency_sec))
                except Exception:
                    continue
            for fvg in analysis.get('fair_value_gaps', []):
                try:
                    if fvg.get('type') == 'bullish' and float(fvg['low']) <= current_price <= float(fvg['high']):
                        if float(last.get('close', 0)) > float(last.get('open', 0)):
                            return _fmt(now + timedelta(seconds=latency_sec))
                except Exception:
                    continue
        elif signal_type == 'sell':
            for block in analysis.get('order_blocks', []):
                try:
                    if block.get('type') == 'bearish' and float(block['low']) * 0.998 <= current_price <= float(block['high']):
                        return _fmt(now + timedelta(seconds=latency_sec))
                except Exception:
                    continue
            for fvg in analysis.get('fair_value_gaps', []):
                try:
                    if fvg.get('type') == 'bearish' and float(fvg['low']) <= current_price <= float(fvg['high']):
                        if float(last.get('close', 0)) < float(last.get('open', 0)):
                            return _fmt(now + timedelta(seconds=latency_sec))
                except Exception:
                    continue

        # Default: align to next minute open with volatility-aware buffer
        next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return _fmt(next_minute + timedelta(seconds=buffer_sec))

    except Exception as e:
        logger.error(f"Error in entry timing: {e}")
        # Fallback to next minute + 10s buffer
        next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return _fmt(next_minute + timedelta(seconds=10))

def generate_signal_for_asset(asset: str, candles: List[Dict]):
    """
    Generate trading signal combining ML prediction and TA confirmation.
    Returns 'call', 'put', or None
    """
    logger.info(f"=== Generating signal for {asset} ===")
    # Ensure globals are declared before any use inside this function
    global last_confidence_score, last_signal_details
    if not candles or len(candles) < 30:
        logger.warning(f"Insufficient candle data for {asset}: {len(candles) if candles else 0} candles (need ≥30)")
        return None
    
    try:
        # OPTIMIZED DATA REQUIREMENTS - ALLOW MORE SIGNALS
        if len(candles) < 30:  # Minimum requirement
            logger.warning(f"Insufficient candle data for {asset}: {len(candles)} < 30")
            return None
        
        # 1. Get Enhanced TA Analysis (before ML to pass context into ensemble)
        try:
            ta_analysis = enhanced_ta.analyze_market_conditions(candles)
            
            logger.info(f"TA Analysis for {asset}: Score={ta_analysis['overall_score']:.1f}, "
                       f"Confidence={ta_analysis['confidence_level']}, "
                       f"Trend={ta_analysis['trend_analysis']['trend_direction']}, "
                       f"Regime={ta_analysis['market_regime']}")
            
            # Get current price from latest candle
            current_price = float(candles[-1].get('close', 0))
            
            # 2. Get Ensemble ML prediction with adaptive learner (auto-trains if missing)
            try:
                ens_signal, ens_confidence, signal_info = adaptive_ensemble_learner.get_enhanced_signal(asset, candles, ta_analysis)
                if ens_signal is None:
                    ml_prediction, ml_confidence = None, 0.0
                    logger.info(f"No Ensemble ML signal for {asset} (below threshold)")
                else:
                    # Map ensemble label to internal ML direction
                    if ens_signal in ('call', 'buy'):
                        ml_prediction = 'buy'
                    elif ens_signal in ('put', 'sell'):
                        ml_prediction = 'sell'
                    else:
                        ml_prediction = None
                    # Confidence comes back on 0-100 scale from learner
                    ml_confidence = float(ens_confidence or 0.0)
                    logger.info(f"Ensemble ML: {asset} -> {ens_signal.upper()} @ {ml_confidence:.1f}% (dyn thr: {signal_info.get('dynamic_threshold')}), feat#: {signal_info.get('features_count')}")
            except Exception as e:
                logger.error(f"Ensemble ML failed for {asset}: {e}")
                ml_prediction, ml_confidence = None, 0.0

            # REVOLUTIONARY TA SIGNAL LOGIC - QUANTUM-ENHANCED GRADE
            ta_signal = None
            # Increase base confidence to generate more signals
            confidence_score = int(ml_confidence) if ml_confidence else 80  # Increased base confidence
            ta_score = 0  # track TA score used to decide
            
            # QUANTUM-ENHANCED SIGNAL GENERATION
            quantum_analysis = ta_analysis.get('quantum_analysis', {})
            fractal_analysis = ta_analysis.get('fractal_analysis', {})
            neural_analysis = ta_analysis.get('neural_analysis', {})
            balanced_signals = ta_analysis.get('balanced_signals', {})
            transcendence_score = ta_analysis.get('dimensional_transcendence', 50)
            
            # Log revolutionary analysis
            logger.info(f"🌌 QUANTUM STATE: {quantum_analysis.get('quantum_state', 'unknown')} | "
                       f"FRACTAL DIM: {fractal_analysis.get('fractal_dimension', 1.5):.2f} | "
                       f"NEURAL FREQ: {neural_analysis.get('neural_frequency', 0.5):.2f} | "
                       f"TRANSCENDENCE: {transcendence_score:.1f}/100")
            
            # Extract key TA metrics from enhanced analysis
            trend_analysis = ta_analysis['trend_analysis']
            momentum_analysis = ta_analysis['momentum_analysis']
            volatility_analysis = ta_analysis['volatility_analysis']
            overall_score = ta_analysis['overall_score']
            confidence_level = ta_analysis['confidence_level']
            market_regime = ta_analysis['market_regime']
            # Initialize trend_direction early for safe logging in all branches
            trend_direction = trend_analysis.get('trend_direction', 'neutral')
            ema_slopes = trend_analysis.get('ema_slopes', {})
            price_vs_emas_pct = trend_analysis.get('price_vs_emas_pct', {})
            ema8_slope = float(ema_slopes.get('ema8', 0.0) or 0.0)
            ema21_slope = float(ema_slopes.get('ema21', 0.0) or 0.0)
            ema55_slope = float(ema_slopes.get('ema55', 0.0) or 0.0)
            price_vs_ema21 = float(price_vs_emas_pct.get('ema21', 0.0) or 0.0)
            
            # Adjust confidence based on TA analysis
            ta_confidence_multiplier = {
                'very_high': 1.2,
                'high': 1.1,
                'medium': 1.0,
                'low': 0.9
            }.get(confidence_level, 1.0)
            
            # ENHANCED SIGNAL GENERATION USING PROFESSIONAL TA
            # Compute BOTH CALL and PUT TA scores irrespective of ML to avoid directional bias
            rsi_condition = momentum_analysis.get('rsi_condition', 'neutral')
            macd_condition = momentum_analysis.get('macd_condition', 'neutral')
            volume_condition = ta_analysis.get('volume_analysis', {}).get('volume_condition', 'normal')

            # CALL score
            call_score = 0
            call_reasons = []
            if trend_direction == 'bullish':
                call_score += 35  # Reduced from 40
                call_reasons.append(f"Bullish trend ({trend_analysis.get('alignment_score', 0):.2f})")
            elif trend_direction == 'neutral':
                call_score += 8  # Further reduce neutral boost to curb CALL bias
                call_reasons.append("Neutral trend")
            elif trend_direction == 'bearish':
                # Penalize CALL in bearish trend context to reduce bias
                call_score -= 10
                call_reasons.append("Bearish trend penalty for CALL")
            if rsi_condition in ['oversold', 'bullish'] or macd_condition == 'bullish':
                momentum_score = 30 if rsi_condition == 'oversold' else 20  # Reduced
                call_score += momentum_score
                call_reasons.append(f"Momentum: RSI={rsi_condition}, MACD={macd_condition}")
            if market_regime in ['normal', 'low_volatility']:
                call_score += 12  # Reduced from 15
                call_reasons.append(f"Favorable regime: {market_regime}")
            elif market_regime == 'high_volatility':
                call_score += 5
                call_reasons.append("High volatility regime")
            if volume_condition == 'high':
                call_score += 8  # Reduced from 10
                call_reasons.append("High volume confirmation")
            elif volume_condition == 'normal':
                call_score += 5

            # PUT score - ENHANCED FOR PROFITABILITY (Data-driven approach)
            put_score = 0
            put_reasons = []
            rsi_value = float(momentum_analysis.get('rsi', 50) or 50)
            macd_value = float(momentum_analysis.get('macd', 0) or 0)
            macd_signal = float(momentum_analysis.get('macd_signal', 0) or 0)
            
            # STRICT PUT CONDITIONS - Only high-probability reversals
            if trend_direction == 'bearish':
                put_score += 50  # Strong bearish trend
                put_reasons.append(f"Strong bearish trend ({trend_analysis.get('alignment_score', 0):.2f})")
            elif trend_direction == 'neutral':
                # Neutral trends require additional confirmation for PUT
                if rsi_value >= 70:  # Only very overbought in neutral
                    put_score += 30
                    put_reasons.append("Neutral trend + severe overbought")
                else:
                    put_score += 10  # Reduced from 25 - neutral not ideal for PUT
                    put_reasons.append("Neutral trend (weak PUT setup)")
            
            # ENHANCED RSI-BASED PUT LOGIC (Proven reversal patterns)
            if rsi_value >= 75:  # Severely overbought - high probability reversal
                put_score += 45
                put_reasons.append(f"Severely overbought RSI ({rsi_value:.1f}) - strong reversal signal")
            elif rsi_value >= 70:  # Standard overbought
                put_score += 35
                put_reasons.append(f"Overbought RSI ({rsi_value:.1f}) - reversal signal")
            elif rsi_value >= 65:  # Early overbought warning
                put_score += 20
                put_reasons.append(f"Early overbought RSI ({rsi_value:.1f}) - caution zone")
            elif rsi_value <= 45:  # RSI too low for PUT signals
                put_score -= 20
                put_reasons.append(f"RSI too low for PUT ({rsi_value:.1f}) - penalty")
            
            # MACD DIVERGENCE DETECTION (Critical for PUT success)
            if macd_condition == 'bearish':
                if macd_value < macd_signal:  # MACD below signal line
                    put_score += 25
                    put_reasons.append("MACD bearish divergence confirmed")
                else:
                    put_score += 15
                    put_reasons.append("MACD bearish momentum")
            elif macd_condition == 'bullish':
                put_score -= 15  # Penalty for conflicting MACD
                put_reasons.append("MACD bullish (conflicts with PUT) - penalty")
            
            # MARKET REGIME ANALYSIS (Critical for PUT timing)
            if market_regime == 'low_volatility':
                # Low volatility = harder to reverse, need stronger signals
                if rsi_value >= 72:  # Only very overbought in low vol
                    put_score += 15
                    put_reasons.append("Low volatility + extreme overbought")
                else:
                    put_score += 5
                    put_reasons.append("Low volatility (requires stronger signals)")
            elif market_regime == 'normal':
                put_score += 12
                put_reasons.append("Normal volatility regime")
            elif market_regime == 'high_volatility':
                # High volatility = better for reversals but need confirmation
                if rsi_value >= 65:
                    put_score += 20
                    put_reasons.append("High volatility + overbought (ideal PUT setup)")
                else:
                    put_score += 8
                    put_reasons.append("High volatility regime")
            
            # VOLUME CONFIRMATION (Essential for PUT success)
            if volume_condition == 'high':
                if rsi_value >= 68:  # High volume + overbought = strong reversal
                    put_score += 20
                    put_reasons.append("High volume + overbought (strong reversal setup)")
                else:
                    put_score += 10
                    put_reasons.append("High volume confirmation")
            elif volume_condition == 'normal':
                put_score += 5

            # Bearish bias conditions: short/medium EMA slopes negative and price below EMA21
            bearish_bias = (ema8_slope < 0 and ema21_slope < 0)
            price_below_ema21 = (price_vs_ema21 < -0.05)  # below EMA21 by >0.05%
            if bearish_bias and price_below_ema21:
                # ENHANCED bearish bias with RSI confirmation
                if rsi_value >= 65:  # Only with overbought confirmation
                    put_score += 15  # Increased bonus for confirmed setup
                    call_score = max(0, call_score - 10)
                    put_reasons.append("Bearish bias + overbought RSI - premium setup")
                    logger.info(
                        f"PREMIUM TA bearish-bias for {asset}: slopes negative, price below EMA21, RSI={rsi_value:.1f} -> put+15"
                    )
                else:
                    put_score += 5  # Reduced bonus without RSI confirmation
                    call_score = max(0, call_score - 5)
                    put_reasons.append("Bearish bias without RSI confirmation")
                    logger.info(
                        f"Basic TA bearish-bias for {asset}: slopes negative, price below EMA21, but RSI={rsi_value:.1f} -> put+5"
                    )

            # ENHANCED MOMENTUM CONFIRMATION for PUT profitability
            if macd_condition == 'bearish' and rsi_value >= 60:  # Only with overbought RSI
                put_score += 8  # Increased bonus
                call_score = max(0, call_score - 5)
                put_reasons.append("Bearish MACD + overbought RSI - strong PUT setup")
            elif macd_condition == 'bearish' and rsi_value >= 50:
                put_score += 5
                call_score = max(0, call_score - 3)
                put_reasons.append("Bearish MACD + neutral RSI")
            
            # PRICE ACTION CONFIRMATION (New addition)
            price_vs_ema8 = float(price_vs_emas_pct.get('ema8', 0) or 0)
            if price_vs_ema8 > 0.15 and rsi_value >= 65:  # Price extended above EMA8 + overbought
                put_score += 15
                put_reasons.append("Price overextended + overbought - reversal setup")
            elif price_vs_ema8 > 0.10 and rsi_value >= 68:
                put_score += 10
                put_reasons.append("Price extended + overbought")

            # DYNAMIC THRESHOLDS - QUALITY OVER QUANTITY for PUT signals
            call_threshold = 65  # Raise CALL threshold to reduce CALL bias
            put_threshold = 70   # RAISED PUT threshold for better quality
            
            # STRICT PUT CONDITIONS for profitability
            if bearish_bias and price_below_ema21:
                if rsi_value >= 68:  # Only strong overbought in bearish bias
                    put_threshold = 65  # Slightly easier with strong setup
                    put_reasons.append("Bearish bias + overbought - quality setup")
                else:
                    put_threshold = 75  # Stricter without overbought
                call_threshold = 65
            elif rsi_value >= 72:  # Severely overbought
                put_threshold = 60  # Allow easier PUT for extreme overbought
                put_reasons.append("Severely overbought - high probability reversal")
                call_threshold = 70  # Stricter CALL when overbought
            elif rsi_value >= 68:  # Standard overbought
                put_threshold = 65  # Moderate threshold
                call_threshold = 65
            else:
                put_threshold = 75  # High threshold for weak setups
                call_threshold = 65  # Keep CALL threshold elevated even in weak setups

            # REVOLUTIONARY MULTI-DIMENSIONAL SIGNAL DECISION
            selected_ta_signal = None
            selected_ta_score = 0
            
            # Check if quantum-enhanced balanced signals are available
            quantum_signal = balanced_signals.get('primary_signal', 'neutral')
            quantum_strength = balanced_signals.get('signal_strength', 0)
            dimensional_consensus = balanced_signals.get('dimensional_consensus', False)
            
            # Multi-dimensional signal fusion
            if quantum_signal != 'neutral' and transcendence_score > 60 and dimensional_consensus:
                # Use revolutionary quantum-enhanced signals
                selected_ta_signal = quantum_signal
                selected_ta_score = quantum_strength
                
                # Boost confidence with transcendence score
                confidence_score = int(confidence_score * (1 + transcendence_score / 200))
                
                logger.info(f"🚀 REVOLUTIONARY SIGNAL: {quantum_signal.upper()} from quantum analysis")
                logger.info(f"   📊 Quantum Strength: {quantum_strength}/100 | Transcendence: {transcendence_score:.1f}/100")
                logger.info(f"   🌌 PUT Reasons: {', '.join(balanced_signals.get('put_reasons', []))}")
                logger.info(f"   ⭐ CALL Reasons: {', '.join(balanced_signals.get('call_reasons', []))}")
                logger.info(f"   ⚖️ Balance Ratio: {balanced_signals.get('balance_ratio', 1.0):.2f}")
                
            elif call_score >= call_threshold or put_score >= put_threshold:
                # Fallback to traditional TA with quantum enhancement
                if call_score >= call_threshold and put_score >= put_threshold:
                    # Apply quantum weighting to traditional signals
                    quantum_put_boost = quantum_analysis.get('collapse_probability', 0.5) * 20
                    quantum_call_boost = (1 - quantum_analysis.get('collapse_probability', 0.5)) * 20
                    
                    adjusted_put_score = put_score + quantum_put_boost
                    adjusted_call_score = call_score + quantum_call_boost
                    
                    # Prefer ML agreement if available, otherwise quantum-weighted stronger signal
                    if ml_prediction == 'buy' and adjusted_call_score >= 50:
                        selected_ta_signal, selected_ta_score = 'call', adjusted_call_score
                    elif ml_prediction == 'sell' and adjusted_put_score >= 50:
                        selected_ta_signal, selected_ta_score = 'put', adjusted_put_score
                    else:
                        if adjusted_call_score >= adjusted_put_score:
                            selected_ta_signal, selected_ta_score = 'call', adjusted_call_score
                        else:
                            selected_ta_signal, selected_ta_score = 'put', adjusted_put_score
                    
                    logger.info(f"🔮 QUANTUM-ENHANCED TA: {selected_ta_signal.upper()} "
                               f"(Original: CALL={call_score}, PUT={put_score} | "
                               f"Quantum-Adjusted: CALL={adjusted_call_score:.1f}, PUT={adjusted_put_score:.1f})")
                    
                elif call_score >= call_threshold:
                    selected_ta_signal, selected_ta_score = 'call', call_score
                else:
                    selected_ta_signal, selected_ta_score = 'put', put_score

            # Log TA outcome
            if selected_ta_signal:
                ta_signal = selected_ta_signal
                ta_score = selected_ta_score
                if ta_signal == 'call':
                    logger.info(f"✅ CALL signal for {asset} - Score: {call_score}/100 (threshold: {call_threshold})")
                    logger.info(f"   Reasons: {', '.join(call_reasons)}")
                    if bearish_bias and price_below_ema21:
                        logger.info(f"   ⚠️ Note: Generated CALL despite bearish bias (slopes negative, price below EMA21)")
                else:
                    logger.info(f"✅ PUT signal for {asset} - Score: {put_score}/100 (threshold: {put_threshold})")
                    logger.info(f"   Reasons: {', '.join(put_reasons)}")
                    if bearish_bias and price_below_ema21:
                        logger.info(f"   ✅ PUT signal aligned with bearish bias (slopes negative, price below EMA21)")
            else:
                logger.info(
                    f"❌ TA rejected for {asset} - CALL {call_score}/100 (≥{call_threshold}), "
                    f"PUT {put_score}/100 (≥{put_threshold})"
                )
                
                # Log detailed scoring breakdown for analysis
                logger.info(
                    f"📊 TA Scoring Details for {asset}: CALL reasons: {', '.join(call_reasons)} | PUT reasons: {', '.join(put_reasons)}"
                )
            
            # If TA signal was set via limited-data fallback, ensure selected_ta_score exists
            if ta_signal and 'selected_ta_score' not in locals():
                selected_ta_score = ta_score
            
            # Calculate base confidence based on TA score
            if selected_ta_score >= 70:
                base_confidence = 82
            elif selected_ta_score >= 55:
                base_confidence = 80
            else:
                base_confidence = 78  # Lower scoring signals get reduced base
                
                # Adjust confidence based on TA analysis quality
                confidence_score = int(base_confidence * ta_confidence_multiplier)
                
                # Bonus for extreme conditions
                rsi_value = momentum_analysis.get('rsi', 50)
                if ta_signal == "put" and rsi_value >= 75:  # Severely overbought
                    confidence_score = min(confidence_score + 10, 95)
                elif ta_signal == "call" and rsi_value <= 25:  # Severely oversold  
                    confidence_score = min(confidence_score + 10, 95)
                elif ta_signal == "put" and rsi_value >= 70:  # Overbought
                    confidence_score = min(confidence_score + 5, 90)
                elif ta_signal == "call" and rsi_value <= 30:  # Oversold
                    confidence_score = min(confidence_score + 5, 90)
                
                # Adjust for market regime
                if market_regime == 'high_volatility':
                    confidence_score = max(confidence_score - 5, 70)  # Reduce confidence in high volatility

                # Signal Quality Enhancement: Apply additional filters for stability
                # 1. Confidence boost for strong agreement between ML and TA
                if ta_signal and ml_prediction:
                    ml_ta_agreement = (
                        (ta_signal == 'call' and ml_prediction == 'buy') or
                        (ta_signal == 'put' and ml_prediction == 'sell')
                    )
                    if ml_ta_agreement and confidence_score >= 75:
                        confidence_score = min(100, confidence_score + 5)
                        logger.info(f"🤝 ML-TA agreement bonus: +5% confidence for {asset}")
                
                # 2. Penalize signals in choppy/sideways markets (low ADX) - relaxed
                adx_value = float(trend_analysis.get('adx', 25) or 25)
                if adx_value < 15:  # Very weak trend (lowered threshold)
                    confidence_score = max(65, confidence_score - 5)  # Reduced penalty
                    logger.info(f"⚠️ Weak trend penalty: ADX={adx_value:.1f}, confidence reduced by 5%")
                
                # 3. Boost confidence for signals in strong trending markets
                elif adx_value > 30:  # Strong trend (lowered threshold)
                    confidence_score = min(100, confidence_score + 5)  # Increased bonus
                    logger.info(f"📈 Strong trend bonus: ADX={adx_value:.1f}, confidence +5%")
                
                # Clamp confidence to [0,100]
                confidence_score = max(0, min(100, confidence_score))

                # Teacher-mode: if TA is confident enough, record as teacher signal
                if TEACHER_MODE and confidence_score >= int(TEACHER_CONF_MIN or 0):
                    try:
                        ta_ctx = {
                            'trend_direction': trend_direction,
                            'market_regime': market_regime,
                        }
                        adaptive_ensemble_learner.record_teacher_signal(
                            asset=asset,
                            label=ta_signal,
                            confidence=float(confidence_score),
                            ta_context=ta_ctx,
                            ml_prediction=ml_prediction
                        )
                        # add teacher metadata for downstream consumers
                        last_signal_details['teacher_label'] = ta_signal
                        last_signal_details['teacher_confidence'] = confidence_score
                    except Exception as e:
                        logger.error(f"Teacher record failed for {asset}: {e}")

            # ENHANCED SIGNAL QUALITY CHECKS - Different thresholds for CALL vs PUT
            if ta_signal == 'put':
                # STRICTER requirements for PUT signals to improve profitability
                # Relax threshold slightly in limited-data mode
                min_put_confidence = 70 if 'limited_data_mode' in locals() and limited_data_mode else 75
                if confidence_score < min_put_confidence:
                    logger.info(f"🚫 PUT signal quality check failed for {asset}: confidence {confidence_score}% < {min_put_confidence}% minimum")
                    return None
                
                # Additional PUT-specific quality checks
                if not ('limited_data_mode' in locals() and limited_data_mode):
                    if rsi_value < 60:  # PUT signals need overbought conditions
                        logger.info(f"🚫 PUT signal rejected for {asset}: RSI {rsi_value:.1f} too low (need ≥60 for PUT)")
                        return None
                
                if trend_direction == 'bullish' and confidence_score < (80 if 'limited_data_mode' in locals() and limited_data_mode else 85):
                    logger.info(f"🚫 PUT signal rejected for {asset}: bullish trend requires {(80 if 'limited_data_mode' in locals() and limited_data_mode else 85)}%+ confidence, got {confidence_score}%")
                    return None
                
                logger.info(f"✅ PUT signal quality check passed for {asset}: confidence {confidence_score}%, RSI {rsi_value:.1f}")
            else:
                # CALL-specific quality checks to reduce CALL bias
                if ta_signal == 'call':
                    # Require reasonably oversold RSI for CALL unless extremely strong
                    rsi_value = momentum_analysis.get('rsi', 50)
                    if rsi_value > 40 and confidence_score < (80 if 'limited_data_mode' in locals() and limited_data_mode else 85):
                        logger.info(
                            f"🚫 CALL rejected for {asset}: RSI {rsi_value:.1f} not oversold (≤40) and confidence {confidence_score}% < {(80 if 'limited_data_mode' in locals() and limited_data_mode else 85)}%"
                        )
                        return None
                    # Avoid CALLs in bearish bias unless very strong
                    if trend_direction == 'bearish' and confidence_score < (82 if 'limited_data_mode' in locals() and limited_data_mode else 85):
                        logger.info(
                            f"🚫 CALL rejected for {asset}: bearish trend requires ≥{(82 if 'limited_data_mode' in locals() and limited_data_mode else 85)}% confidence, got {confidence_score}%"
                        )
                        return None
                    if bearish_bias and price_below_ema21 and confidence_score < (85 if 'limited_data_mode' in locals() and limited_data_mode else 88):
                        logger.info(
                            f"🚫 CALL rejected for {asset}: bearish bias (EMA slopes negative, price below EMA21) requires ≥{(85 if 'limited_data_mode' in locals() and limited_data_mode else 88)}% confidence"
                        )
                        return None
                    # Baseline minimum confidence for CALL
                    if confidence_score < (68 if 'limited_data_mode' in locals() and limited_data_mode else 70):
                        logger.info(
                            f"🚫 CALL signal quality check failed for {asset}: confidence {confidence_score}% < {(68 if 'limited_data_mode' in locals() and limited_data_mode else 70)}% minimum"
                        )
                        return None
                else:
                    # If not explicitly a PUT (and not a CALL), reject
                    if confidence_score < 65:
                        logger.info(
                            f"🚫 Signal quality check failed for {asset}: confidence {confidence_score}% < 65% minimum"
                        )
                        return None
            
            # Store confidence information globally (ensure variables exist)
            # clamp stored confidence as well
            last_confidence_score = max(0, min(100, confidence_score))
            last_signal_details = {
                'asset': asset,
                'signal_type': ta_signal if ta_signal else ('call' if ml_prediction == 'buy' else ('put' if ml_prediction == 'sell' else None)),
                'decision_source': ('TA' if ta_signal else 'ML'),
                'overall_score': overall_score,
                'confidence_level': confidence_level,
                'market_regime': market_regime,
                'ml_prediction': ml_prediction,
                'ml_confidence': ml_confidence,
                'ta_signal': ta_signal,
                'ta_score': ta_score,
                'ta_analysis': {
                    'trend_direction': trend_direction,
                    'rsi': momentum_analysis.get('rsi'),
                    'macd_signal': momentum_analysis.get('macd_signal'),
                    'volatility_regime': volatility_analysis.get('volatility_regime')
                }
            }
            
            # Check signal agreement and score
            ml_ta_agreement = (ml_prediction == 'buy' and ta_signal == 'call') or (ml_prediction == 'sell' and ta_signal == 'put')
            
            if ml_ta_agreement:
                # When ML and TA agree, boost confidence
                confidence_score = min(95, confidence_score + 15)
                logger.info(f"✅ STRONG SIGNAL - ML & TA AGREE on {ta_signal.upper()} for {asset}")
                logger.info(f"   📊 Confidence: {confidence_score}% (TA Level: {confidence_level})")
                logger.info(f"   🎯 Overall Score: {overall_score:.1f}/100, Regime: {market_regime}")
                # Regime-specific gates
                if market_regime == 'high_volatility':
                    # require stronger TA or very strong ML+TA agreement
                    if (ta_signal and ta_score >= 78) or (ml_confidence and ml_confidence >= 92):
                        return ta_signal
                    logger.info("🛑 Blocked by high-volatility gate (need TA≥78 or ML≥92)")
                    return None
                elif market_regime == 'low_volatility':
                    # allow slightly lower TA score if agree
                    if ta_signal and (ta_score >= 68 or overall_score >= 68):
                        return ta_signal
                    # fallback to agreement default
                    return ta_signal
                else:
                    return ta_signal
            elif overall_score >= 75 and confidence_level in ['high', 'very_high']:
                # If TA score is very high, generate signal based on trend direction
                trend_direction = trend_analysis.get('trend_direction', 'neutral')
                if trend_direction == 'bullish':
                    ta_override_signal = 'call'
                elif trend_direction == 'bearish':
                    ta_override_signal = 'put'
                else:
                    ta_override_signal = None
                
                if ta_override_signal:
                    logger.info(f"✅ HIGH-CONFIDENCE TA SIGNAL for {asset} (overrides ML)")
                    logger.info(f"   📊 Score: {overall_score}/100, Confidence: {confidence_level}")
                    logger.info(f"   📈 TA Signal: {ta_override_signal.upper()} (trend: {trend_direction})")
                    # update decision source for transparency
                    last_signal_details['decision_source'] = 'TA'
                    last_signal_details['ta_signal'] = ta_override_signal
                    last_signal_details['ta_score'] = overall_score
                    # record TA override as teacher if in teacher mode
                    if TEACHER_MODE and overall_score >= int(max(TEACHER_CONF_MIN, 70)):
                        try:
                            adaptive_ensemble_learner.record_teacher_signal(
                                asset=asset,
                                label=ta_override_signal,
                                confidence=float(confidence_score or overall_score),
                                ta_context={'trend_direction': trend_direction, 'market_regime': market_regime},
                                ml_prediction=ml_prediction
                            )
                            last_signal_details['teacher_label'] = ta_override_signal
                            last_signal_details['teacher_confidence'] = int(confidence_score or overall_score)
                        except Exception as e:
                            logger.error(f"Teacher record (override) failed for {asset}: {e}")
                    # Regime-specific gates for TA override
                    if market_regime == 'high_volatility' and overall_score < 80:
                        logger.info("🛑 High-volatility TA override gate: need TA overall ≥80")
                        return None
                    return ta_override_signal
                else:
                    logger.info(f"⚠️ High TA score but neutral trend - falling back to ML signal")
            elif ml_confidence > 85 and overall_score >= 50:  # Relaxed thresholds
                # If ML is very confident and TA doesn't strongly disagree
                logger.info(f"✅ HIGH-CONFIDENCE ML SIGNAL for {asset} (TA score: {overall_score}/100)")
                logger.info(f"   ⚠️  Note: TA shows {trend_direction} trend")
                last_signal_details['decision_source'] = 'ML'
                # Regime-specific gates for ML-driven signal (relaxed)
                if market_regime == 'high_volatility' and ml_confidence < 90:
                    logger.info("🛑 High-volatility ML gate: need ML ≥90")
                    return None
                # Teacher-mode gating: if not empowered yet, require stronger ML to proceed
                if TEACHER_MODE:
                    try:
                        status = adaptive_ensemble_learner.get_empowerment_status(asset)
                        if not status.get('ready') and ml_confidence < 90:
                            logger.info("🛑 Teacher-mode gate: ML not yet empowered (requires ≥90% confidence)")
                            return None
                    except Exception:
                        pass
                return 'call' if ml_prediction == 'buy' else 'put'
            elif ta_signal and overall_score >= 60:  # Allow TA-only signals with PUT restrictions
                # STRICTER TA-only requirements for PUT signals
                if ta_signal == 'put':
                    if overall_score < 70:  # Higher TA score requirement for PUT
                        logger.info(f"❌ PUT TA-only signal rejected for {asset}: score {overall_score}/100 < 70 minimum")
                        return None
                    if rsi_value < 65:  # RSI requirement for TA-only PUT
                        logger.info(f"❌ PUT TA-only signal rejected for {asset}: RSI {rsi_value:.1f} < 65 minimum")
                        return None
                        
                logger.info(f"✅ TA-ONLY SIGNAL for {asset} (score: {overall_score}/100)")
                logger.info(f"   📈 TA Signal: {ta_signal.upper()} (trend: {trend_direction})")
                last_signal_details['decision_source'] = 'TA'
                # Teacher-mode: record TA-only as teacher if confident
                if TEACHER_MODE and int(confidence_score or 0) >= int(TEACHER_CONF_MIN or 0):
                    try:
                        adaptive_ensemble_learner.record_teacher_signal(
                            asset=asset,
                            label=ta_signal,
                            confidence=float(confidence_score),
                            ta_context={'trend_direction': trend_direction, 'market_regime': market_regime},
                            ml_prediction=ml_prediction
                        )
                        last_signal_details['teacher_label'] = ta_signal
                        last_signal_details['teacher_confidence'] = int(confidence_score)
                    except Exception as e:
                        logger.error(f"Teacher record (TA-only) failed for {asset}: {e}")
                return ta_signal
            elif ml_confidence > 75:  # Allow ML-only signals with lower threshold
                logger.info(f"✅ ML-ONLY SIGNAL for {asset} (confidence: {ml_confidence}%)")
                logger.info(f"   📊 ML Prediction: {ml_prediction.upper()}")
                last_signal_details['decision_source'] = 'ML'
                # Teacher-mode gating for ML-only low threshold path
                if TEACHER_MODE:
                    try:
                        status = adaptive_ensemble_learner.get_empowerment_status(asset)
                        if not status.get('ready') and ml_confidence < 85:
                            logger.info("🛑 Teacher-mode gate: ML-only signal blocked until empowerment or ≥85% confidence")
                            return None
                    except Exception:
                        pass
                return 'call' if ml_prediction == 'buy' else 'put'
            else:
                # Log detailed rejection reason
                logger.info(f"❌ SIGNAL REJECTED for {asset}: Insufficient confirmation")
                logger.info(f"   📊 ML Prediction: {ml_prediction.upper() if ml_prediction else 'NONE'} (confidence: {ml_confidence}%)") 
                logger.info(f"   📈 TA Analysis: {ta_signal.upper() if ta_signal and isinstance(ta_signal, str) else 'NONE'} (score: {overall_score:.1f}/100)")
                logger.info(f"   🔍 Trend: {trend_direction.upper()}, Regime: {market_regime}")
                return None
                
        except Exception as e:
            logger.error(f"TA analysis failed for {asset}: {e}")
            # Fallback to ML only if TA fails
            return "call" if ml_prediction == "buy" else "put"
            
    except Exception as e:
        logger.error(f"Signal generation failed for {asset}: {e}")
        return None
