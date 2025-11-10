# 🤖 Intelligit Bot - AI-Powered Trading Bot

A streamlined, production-ready trading bot for IQ Option that uses machine learning, adaptive learning, and intelligent risk management.

## 📋 Features

- **🧠 Machine Learning**: Ensemble models with LSTM for signal prediction
- **📊 Signal Engine**: Advanced technical analysis and pattern recognition
- **🎯 Adaptive Learning**: Self-improving system based on trade results
- **⚠️ Risk Management**: Intelligent martingale and loss prevention
- **🔄 Auto Trading**: Fully automated trade execution
- **📈 Performance Tracking**: Comprehensive trade history and analytics

## 🏗️ Project Structure

```
INTELLIGIT-Bot/
├── main.py                          # Main orchestrator
├── config_user.py                   # Configuration management
├── core_iqconnect.py               # IQ Option API integration
├── model/
│   ├── ensemble/                   # Trained ML models (git-ignored)
│   ├── pytorch_lstm.py             # LSTM implementation
│   └── predict.py                  # Prediction engine
├── utils/
│   ├── signal_engine.py            # Signal generation
│   ├── adaptive_learning.py        # Self-learning system
│   ├── martingale_manager.py       # Risk management
│   └── optimized_signal_engine.py  # Enhanced signals
├── iq_connect/
│   ├── websocket_trader.py         # WebSocket trading
│   ├── candle_manager.py           # Data fetching
│   └── trade_executor.py           # Trade execution
└── data/
    └── performance_history.json    # Learning database
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- IQ Option account
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/intelligit-bot.git
   cd intelligit-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your credentials**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   # IQ Option Credentials
   IQ_OPTION_EMAIL=your_email@example.com
   IQ_OPTION_PASSWORD=your_password
   ACCOUNT_TYPE=PRACTICE  # Start with PRACTICE
   
   # Trading Settings
   INITIAL_STAKE=1.0
   MAX_STAKE=50.0
   SIGNAL_THRESHOLD=85.0
   
   # Risk Management
   MAX_DAILY_LOSS=100.0
   MAX_CONSECUTIVE_LOSSES=3
   ```

4. **Run the bot**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Environment Variables

All configuration is done via the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `IQ_OPTION_EMAIL` | Your IQ Option email | Required |
| `IQ_OPTION_PASSWORD` | Your IQ Option password | Required |
| `ACCOUNT_TYPE` | `PRACTICE` or `REAL` | `PRACTICE` |
| `INITIAL_STAKE` | Starting stake amount | `1.0` |
| `MAX_STAKE` | Maximum stake amount | `50.0` |
| `SIGNAL_THRESHOLD` | Minimum confidence % | `85.0` |
| `MAX_DAILY_LOSS` | Maximum daily loss limit | `100.0` |
| `MAX_CONSECUTIVE_LOSSES` | Max losses before stop | `3` |
| `SCAN_INTERVAL` | Seconds between scans | `60` |
| `TRADE_DURATION` | Minutes per trade | `1` |

### Trading Assets

Edit in `.env`:
```env
TRADING_ASSETS=EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURGBP,EURJPY,GBPJPY
```

## 📊 Usage

### Start Trading

```bash
python main.py
```

The bot will:
1. ✅ Connect to IQ Option
2. 🔄 Load ML models
3. 📊 Scan assets for signals
4. 💰 Execute trades automatically
5. 📈 Track performance and adapt

### Test Configuration

```bash
python config_user.py
```

### Test Connection

```bash
python core_iqconnect.py
```

## 🧠 How It Works

### 1. Signal Generation
The bot analyzes market data using:
- Machine learning predictions
- Technical analysis indicators
- Pattern recognition
- Market microstructure analysis

### 2. Trade Decision
Only executes trades when:
- Signal confidence ≥ threshold
- Risk limits not exceeded
- Asset available for trading
- No active trades on same asset

### 3. Trade Execution
- Calculates optimal stake (martingale)
- Places trade via IQ Option API
- Monitors trade result
- Records outcome for learning

### 4. Adaptive Learning
- Analyzes winning/losing patterns
- Adjusts strategy based on performance
- Improves signal quality over time
- Updates performance database

## 📈 Performance Tracking

All trades are logged in `data/performance_history.json`:
- Trade details (asset, direction, stake, result)
- Win/loss statistics
- Profit/loss tracking
- Pattern analysis
- Optimal asset/timeframe discovery

## ⚠️ Risk Management

The bot includes multiple safety features:

1. **Daily Loss Limit**: Stops trading if max loss reached
2. **Consecutive Loss Limit**: Pauses after N losses
3. **Martingale Management**: Controlled stake progression
4. **Signal Filtering**: Only high-confidence trades
5. **Asset Cooldown**: Prevents overtrading same asset

## 🔧 Advanced Configuration

### Enable/Disable Features

In `.env`:
```env
ENABLE_ADAPTIVE_LEARNING=True
ENABLE_MARTINGALE=True
USE_TECHNICAL_ANALYSIS=True
MARTINGALE_MULTIPLIER=2.0
```

### Telegram Notifications (Optional)

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📝 Logging

Logs are saved to `intelligit_bot.log`:
- Connection status
- Signal generation
- Trade execution
- Performance statistics
- Errors and warnings

## 🐛 Troubleshooting

### Connection Issues
```bash
# Test your connection
python core_iqconnect.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Invalid Configuration
```bash
# Validate your config
python config_user.py
```

## 🔒 Security

- ⚠️ Never commit your `.env` file
- 🔐 Keep credentials secure
- ✅ Use strong passwords
- 🧪 Always test with PRACTICE account first
- 💰 Start with small stakes

## 📦 Deployment

### Docker

```bash
docker build -t intelligit-bot .
docker run -d --env-file .env intelligit-bot
```

### Heroku

```bash
heroku create your-bot-name
git push heroku main
heroku config:set IQ_OPTION_EMAIL=your_email
heroku config:set IQ_OPTION_PASSWORD=your_password
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚡ Performance Tips

1. **Start with PRACTICE**: Test strategies without risk
2. **Monitor first 50 trades**: Assess win rate before going live
3. **Adjust threshold**: Higher threshold = fewer but better trades
4. **Regular model retraining**: Update models with fresh data
5. **Review performance logs**: Identify what works best

## 📚 Documentation

- `main.py` - Main bot orchestrator and trading logic
- `config_user.py` - Configuration management and validation
- `core_iqconnect.py` - IQ Option API wrapper
- `model/predict.py` - ML prediction engine
- `utils/signal_engine.py` - Signal generation system
- `utils/adaptive_learning.py` - Self-improvement system
- `utils/martingale_manager.py` - Risk management

## 🎯 Roadmap

- [ ] Web dashboard for monitoring
- [ ] Multi-broker support
- [ ] Advanced ML models
- [ ] Backtesting system
- [ ] Strategy marketplace
- [ ] Mobile app

## ⚠️ Disclaimer

**Trading involves risk.** This bot is for educational purposes. Always:
- Trade responsibly
- Never invest more than you can afford to lose
- Test thoroughly in PRACTICE mode
- Understand the risks of automated trading
- Comply with local regulations

## 📧 Support

- 📖 Read the documentation
- 🐛 Report issues on GitHub
- 💬 Join our community

## 📜 License

MIT License - See LICENSE file for details

---

**Made with ❤️ by the Intelligit Team**

*Start small, learn continuously, trade smart.*
