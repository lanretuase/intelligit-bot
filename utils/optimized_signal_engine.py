# optimized_signal_engine.py - SIMPLIFIED HIGH-PERFORMANCE SIGNAL ENGINE FOR 3-MINUTE TRADES

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OptimizedSignalEngine:
    """
    Simplified, high-performance signal engine optimized for 3-minute binary options.
    Focuses on proven profitable patterns with minimal complexity.
    """
    
    def __init__(self):
        self.last_confidence_score = 0
        self.last_signal_details = {}
        
    def calculate_technical_indicators(self, candles: List[Dict]) -> Dict:
        """Calculate essential technical indicators for 3-minute trades"""
        try:
            closes = np.array([float(c.get('close', 0)) for c in candles])
            highs = np.array([float(c.get('max', c.get('high', c.get('close', 0)))) for c in candles])
            lows = np.array([float(c.get('min', c.get('low', c.get('close', 0)))) for c in candles])
            volumes = np.array([float(c.get('volume', 1)) for c in candles])
            
            if len(closes) < 20:
                return {}
                
            # RSI (14-period)
            rsi = self._calculate_rsi(closes, 14)
            
            # MACD (12,26,9)
            macd, macd_signal, macd_histogram = self._calculate_macd(closes)
            
            # EMAs
            ema8 = self._calculate_ema(closes, 8)
            ema21 = self._calculate_ema(closes, 21)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
            
            # Volume analysis
            volume_sma = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
            
            # Price position relative to EMAs
            current_price = closes[-1]
            price_vs_ema8 = ((current_price - ema8[-1]) / ema8[-1]) * 100 if ema8[-1] > 0 else 0
            price_vs_ema21 = ((current_price - ema21[-1]) / ema21[-1]) * 100 if ema21[-1] > 0 else 0
            
            # Trend strength
            ema8_slope = (ema8[-1] - ema8[-5]) / ema8[-5] * 100 if len(ema8) >= 5 and ema8[-5] > 0 else 0
            ema21_slope = (ema21[-1] - ema21[-5]) / ema21[-5] * 100 if len(ema21) >= 5 and ema21[-5] > 0 else 0
            
            return {
                'rsi': rsi[-1] if len(rsi) > 0 else 50,
                'macd': macd[-1] if len(macd) > 0 else 0,
                'macd_signal': macd_signal[-1] if len(macd_signal) > 0 else 0,
                'macd_histogram': macd_histogram[-1] if len(macd_histogram) > 0 else 0,
                'ema8': ema8[-1] if len(ema8) > 0 else current_price,
                'ema21': ema21[-1] if len(ema21) > 0 else current_price,
                'bb_upper': bb_upper[-1] if len(bb_upper) > 0 else current_price,
                'bb_lower': bb_lower[-1] if len(bb_lower) > 0 else current_price,
                'current_price': current_price,
                'price_vs_ema8': price_vs_ema8,
                'price_vs_ema21': price_vs_ema21,
                'ema8_slope': ema8_slope,
                'ema21_slope': ema21_slope,
                'volume_ratio': volume_ratio,
                'volatility': self._calculate_volatility(closes)
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_optimized_signal(self, asset: str, candles: List[Dict]) -> Tuple[Optional[str], float, Dict]:
        """
        Generate optimized signal for 3-minute binary options.
        Returns (signal, confidence, details)
        """
        try:
            if len(candles) < 30:  # Minimum requirement
                return None, 0, {}
                
            indicators = self.calculate_technical_indicators(candles)
            if not indicators:
                return None, 0, {}
                
            # Extract key indicators
            rsi = indicators['rsi']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_histogram = indicators['macd_histogram']
            current_price = indicators['current_price']
            ema8 = indicators['ema8']
            ema21 = indicators['ema21']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            price_vs_ema8 = indicators['price_vs_ema8']
            price_vs_ema21 = indicators['price_vs_ema21']
            ema8_slope = indicators['ema8_slope']
            ema21_slope = indicators['ema21_slope']
            volume_ratio = indicators['volume_ratio']
            volatility = indicators['volatility']
            
            # CRITICAL: Determine overall market trend first
            trend_direction = self._determine_market_trend(ema8_slope, ema21_slope, price_vs_ema8, price_vs_ema21)
            
            # STRATEGY 1: MEAN REVERSION with trend validation
            call_score, call_reasons = self._analyze_mean_reversion_call(
                rsi, current_price, bb_lower, ema21, macd_histogram, volume_ratio, trend_direction
            )
            
            put_score, put_reasons = self._analyze_mean_reversion_put(
                rsi, current_price, bb_upper, ema21, macd_histogram, volume_ratio, trend_direction
            )
            
            # STRATEGY 2: MOMENTUM CONTINUATION (Secondary strategy)
            if call_score < 70 and put_score < 70:
                momentum_call, momentum_put = self._analyze_momentum_continuation(
                    rsi, macd, macd_signal, ema8_slope, ema21_slope, price_vs_ema8, volume_ratio, trend_direction
                )
                call_score = max(call_score, momentum_call)
                put_score = max(put_score, momentum_put)
                
                if momentum_call > 0:
                    call_reasons.append("Momentum continuation")
                if momentum_put > 0:
                    put_reasons.append("Momentum continuation")
            
            # Final signal determination with practical thresholds
            signal = None
            confidence = 0
            reasons = []
            min_score = 40  # Reduced from 50 for more signals
            min_difference = 8   # Reduced from 10 for better balance
            
            # Debug logging for signal analysis
            logger.info(f"📊 {asset} Analysis: CALL={call_score:.1f}, PUT={put_score:.1f}, Trend={trend_direction}")
            logger.info(f"   RSI={rsi:.1f}, MACD={macd:.4f}, EMA8_slope={ema8_slope:.3f}, Vol={volume_ratio:.2f}")
            
            # Signal selection with trend validation
            if call_score >= min_score and call_score > put_score + min_difference:
                # Additional validation for CALL signals
                if trend_direction == 'bearish' and call_score < 70:
                    logger.info(f"🚫 CALL blocked in bearish trend: score={call_score:.1f} < 70 required")
                elif rsi > 65 and trend_direction == 'bearish':
                    logger.info(f"🚫 CALL blocked: RSI={rsi:.1f} too high in bearish trend")
                else:
                    signal = 'call'
                    confidence = min(95, call_score + 25)
                    reasons = call_reasons
                    logger.info(f"🎯 OPTIMIZED SIGNAL: {asset} CALL @ {confidence:.1f}%")
                    logger.info(f"   Strategy: {', '.join(reasons[:4])}")
                    logger.info(f"   Scores: CALL={call_score:.1f}, PUT={put_score:.1f}")
                    logger.info(f"   RSI={rsi:.1f}, Vol={volatility:.4f}, VolRatio={volume_ratio:.2f}")
            elif put_score >= min_score and put_score > call_score + min_difference:
                signal = 'put'
                confidence = min(95, put_score + 25)
                reasons = put_reasons
                logger.info(f"🎯 OPTIMIZED SIGNAL: {asset} PUT @ {confidence:.1f}%")
                logger.info(f"   Strategy: {', '.join(reasons[:4])}")
                logger.info(f"   Scores: CALL={call_score:.1f}, PUT={put_score:.1f}")
                logger.info(f"   RSI={rsi:.1f}, Vol={volatility:.4f}, VolRatio={volume_ratio:.2f}")
            else:
                logger.info(f"❌ No qualifying signal: CALL={call_score:.1f} PUT={put_score:.1f} (need {min_score}+ with {min_difference}+ difference)")
            
            # Volatility adjustment for 3-minute trades
            if volatility > 0.015:  # High volatility
                confidence = max(60, confidence - 15)  # Larger penalty
            elif volatility < 0.005:  # Low volatility
                confidence = min(90, confidence + 5)
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                confidence = min(90, confidence + 5)
            
            # Store signal details
            self.last_confidence_score = confidence
            self.last_signal_details = {
                'asset': asset,
                'signal_type': signal,
                'confidence': confidence,
                'call_score': call_score,
                'put_score': put_score,
                'reasons': reasons,
                'indicators': indicators,
                'strategy': 'mean_reversion' if any('reversion' in r for r in reasons) else 'momentum',
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'decision_source': 'Optimized_Engine'
            }
            
            if signal:
                logger.info(f"🎯 OPTIMIZED SIGNAL: {asset} {signal.upper()} @ {confidence:.1f}%")
                logger.info(f"   Strategy: {', '.join(reasons)}")
                logger.info(f"   Scores: CALL={call_score:.1f}, PUT={put_score:.1f}")
                logger.info(f"   RSI={rsi:.1f}, Vol={volatility:.4f}, VolRatio={volume_ratio:.2f}")
            
            return signal, confidence, self.last_signal_details
            
        except Exception as e:
            logger.error(f"Error generating optimized signal for {asset}: {e}")
            return None, 0, {}
    
    def _analyze_mean_reversion_call(self, rsi, price, bb_lower, ema21, macd_hist, vol_ratio, trend_direction):
        """Analyze CALL signals based on mean reversion (oversold bounce) with trend validation"""
        score = 0
        reasons = []
        
        # STRICT RSI oversold conditions - prevent CALL in bearish trends
        if rsi <= 25:  # Severely oversold
            score += 45
            reasons.append("Severely oversold RSI")
        elif rsi <= 35:  # Oversold
            score += 35
            reasons.append("Oversold RSI")
        elif rsi <= 40 and trend_direction == 'bullish':  # Only in bullish trend
            score += 20
            reasons.append("Approaching oversold in uptrend")
        else:
            # PENALTY for CALL signals when RSI is not oversold
            if rsi > 50:
                score -= 20  # Significant penalty
                reasons.append("RSI not oversold - risky CALL")
        
        # Bollinger Band support (stricter requirements)
        bb_distance = ((price - bb_lower) / bb_lower) * 100 if bb_lower > 0 else 0
        if bb_distance <= 0.05:  # Very close to lower BB
            score += 30
            reasons.append("Strong BB lower support")
        elif bb_distance <= 0.2:
            score += 20
            reasons.append("BB support zone")
        
        # EMA21 support with trend validation
        ema_distance = ((price - ema21) / ema21) * 100 if ema21 > 0 else 0
        if -0.1 <= ema_distance <= 0.1 and trend_direction != 'bearish':  # Near EMA21, not in bearish trend
            score += 20
            reasons.append("EMA21 support")
        elif ema_distance < -0.3:  # Well below EMA21 - good for bounce
            score += 15
            reasons.append("Below EMA21 - bounce potential")
        
        # MACD histogram turning positive (stricter)
        if macd_hist > 0.0001:  # More significant positive momentum
            score += 15
            reasons.append("Strong MACD bullish momentum")
        elif macd_hist > 0:
            score += 8
            reasons.append("MACD turning bullish")
        
        # Volume confirmation
        if vol_ratio > 1.5:  # Higher volume threshold
            score += 12
            reasons.append("High volume confirmation")
        elif vol_ratio > 1.2:
            score += 6
            reasons.append("Moderate volume")
        
        # TREND VALIDATION PENALTIES
        if trend_direction == 'bearish':
            score -= 25  # Heavy penalty for CALL in bearish trend
            reasons.append("⚠️ Bearish trend penalty")
        elif trend_direction == 'bullish':
            score += 10  # Bonus for CALL in bullish trend
            reasons.append("✅ Bullish trend confirmation")
        
        return score, reasons
    
    def _analyze_mean_reversion_put(self, rsi, price, bb_upper, ema21, macd_hist, vol_ratio, trend_direction):
        """Analyze PUT signals based on mean reversion (overbought reversal) with trend validation"""
        score = 0
        reasons = []
        
        # RSI overbought analysis (enhanced for PUT signals)
        if rsi >= 70:  # Overbought - reduced threshold
            score += 40  # Increased bonus
            reasons.append("Overbought RSI")
        elif rsi >= 65:  # Early overbought
            score += 25  # Increased bonus
            reasons.append("Early overbought")
        elif rsi >= 60:  # Moderate high - new condition
            score += 15
            reasons.append("Moderate high RSI")
        elif rsi <= 40:  # Reduced penalty threshold
            score -= 15  # Reduced penalty
            reasons.append("⚠️ RSI too low for PUT")
        else:
            # PENALTY for PUT signals when RSI is not overbought
            if rsi < 50:
                score -= 20  # Significant penalty
                reasons.append("RSI not overbought - risky PUT")
        
        # Bollinger Band upper resistance - Enhanced
        bb_distance = ((price - bb_upper) / bb_upper) * 100 if bb_upper > 0 else 0
        if bb_distance >= 0:  # Above upper band
            score += 30  # Increased bonus
            reasons.append("Above BB upper band")
        elif bb_distance >= -0.1:  # At upper band
            score += 25
            reasons.append("BB upper resistance")
        elif bb_distance >= -0.2:
            score += 15  # Reduced but still positive
            reasons.append("BB resistance zone")
        
        # EMA21 resistance with trend validation
        ema_distance = ((price - ema21) / ema21) * 100 if ema21 > 0 else 0
        if ema_distance >= 0.3 and trend_direction != 'bullish':  # Well above EMA21, not in bullish trend
            score += 20
            reasons.append("EMA21 resistance")
        elif ema_distance > 0.5:  # Very extended above EMA21
            score += 15
            reasons.append("Extended above EMA21 - reversal potential")
        
        # MACD histogram turning negative - Enhanced
        if macd_hist < -0.00005:  # Reduced threshold for easier trigger
            score += 20  # Increased bonus
            reasons.append("Strong MACD bearish momentum")
        elif macd_hist < 0:
            score += 12  # Increased bonus
            reasons.append("MACD turning bearish")
        
        # Volume confirmation
        if vol_ratio > 1.5:  # Higher volume threshold
            score += 12
            reasons.append("High volume confirmation")
        elif vol_ratio > 1.2:
            score += 6
            reasons.append("Moderate volume")
        
        # TREND VALIDATION PENALTIES - Reduced for better balance
        if trend_direction == 'bullish':
            score -= 15  # Reduced penalty from 25
            reasons.append("⚠️ Bullish trend penalty")
        elif trend_direction == 'bearish':
            score += 20  # Increased bonus from 10
            reasons.append("✅ Bearish trend confirmation")
        else:  # neutral trend
            score += 5   # Small bonus for neutral
            reasons.append("Neutral trend - reversal opportunity")
        
        return score, reasons
    
    def _analyze_momentum_continuation(self, rsi, macd, macd_signal, ema8_slope, ema21_slope, price_vs_ema8, vol_ratio, trend_direction):
        """Analyze momentum continuation signals with trend validation"""
        call_score = 0
        put_score = 0
        
        # Bullish momentum - ONLY in bullish or neutral trends
        if (macd > macd_signal and ema8_slope > 0.05 and ema21_slope > 0.02 and 
            price_vs_ema8 > 0.1 and 40 <= rsi <= 70 and trend_direction != 'bearish'):
            call_score = 60
            
        # Bearish momentum - ONLY in bearish or neutral trends
        if (macd < macd_signal and ema8_slope < -0.05 and ema21_slope < -0.02 and 
            price_vs_ema8 < -0.1 and 30 <= rsi <= 60 and trend_direction != 'bullish'):
            put_score = 60
            
        # Volume boost
        if vol_ratio > 1.5:
            call_score += 10
            put_score += 10
            
        return call_score, put_score
    
    def _determine_market_trend(self, ema8_slope, ema21_slope, price_vs_ema8, price_vs_ema21):
        """Determine overall market trend direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA slope analysis
        if ema8_slope > 0.03:  # Strong upward slope
            bullish_signals += 2
        elif ema8_slope > 0:
            bullish_signals += 1
        elif ema8_slope < -0.03:  # Strong downward slope
            bearish_signals += 2
        elif ema8_slope < 0:
            bearish_signals += 1
            
        if ema21_slope > 0.02:  # Strong upward slope
            bullish_signals += 2
        elif ema21_slope > 0:
            bullish_signals += 1
        elif ema21_slope < -0.02:  # Strong downward slope
            bearish_signals += 2
        elif ema21_slope < 0:
            bearish_signals += 1
        
        # Price position relative to EMAs
        if price_vs_ema8 > 0.2:  # Well above EMA8
            bullish_signals += 1
        elif price_vs_ema8 < -0.2:  # Well below EMA8
            bearish_signals += 1
            
        if price_vs_ema21 > 0.3:  # Well above EMA21
            bullish_signals += 1
        elif price_vs_ema21 < -0.3:  # Well below EMA21
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals >= bearish_signals + 2:
            return 'bullish'
        elif bearish_signals >= bullish_signals + 2:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_optimal_entry_timing(self, signal: str, indicators: Dict) -> str:
        """Get optimal entry timing for 3-minute trades"""
        now = datetime.now()
        
        # For 3-minute trades, we want to enter at specific seconds within the minute
        # to maximize the probability of closing in-the-money
        
        volatility = indicators.get('volatility', 0.01)
        rsi = indicators.get('rsi', 50)
        
        # Calculate optimal entry delay based on market conditions
        if volatility > 0.015:  # High volatility
            delay_seconds = 15  # Enter later to avoid noise
        elif volatility < 0.005:  # Low volatility
            delay_seconds = 5   # Enter early for better positioning
        else:
            delay_seconds = 10  # Standard timing
        
        # Adjust for RSI extremes (better entry timing for reversals)
        if signal == 'put' and rsi >= 75:
            delay_seconds = 5  # Enter quickly on extreme overbought
        elif signal == 'call' and rsi <= 25:
            delay_seconds = 5  # Enter quickly on extreme oversold
        
        # Calculate target entry time
        target_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=delay_seconds)
        
        # If we're already past the optimal time, use next minute
        if target_time <= now:
            target_time += timedelta(minutes=1)
        
        return target_time.strftime('%H:%M:%S')
    
    def get_confidence_info(self, asset: str = None) -> Dict:
        """Get confidence information for the last signal"""
        if asset and self.last_signal_details.get('asset') != asset:
            return {'confidence_score': 0, 'signal_details': {}}
            
        return {
            'confidence_score': self.last_confidence_score,
            'dynamic_threshold': self.last_confidence_score,
            'signal_details': self.last_signal_details,
            'asset_accuracy': 0.65,  # Default decent accuracy
            'timestamp': datetime.now().isoformat()
        }
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
            
            for i in range(period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
                avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period:]
    
    def _calculate_ema(self, prices, period):
        """Calculate EMA"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def _calculate_volatility(self, prices, period=14):
        """Calculate price volatility"""
        if len(prices) < period:
            return 0.01
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-period:])

# Global instance
optimized_engine = OptimizedSignalEngine()
