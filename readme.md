# 📈 Srini's Trading App

> A comprehensive Python-based trading platform supporting multiple brokers, automated strategies, and systematic investment plans.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-blue)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Features

### 🔥 Core Trading Features
- **Multi-Broker Support**: Upstox, Zerodha (Kite Connect)
- **Real-time Market Data**: Live quotes, OHLC, market depth
- **Order Management**: Place, modify, cancel orders with advanced types
- **Portfolio Tracking**: Real-time P&L, holdings, positions
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Strategy Automation**: Custom and pre-built trading strategies

### 📊 Investment Management
- **SIP (Systematic Investment Plan)**: Automated mutual fund investments
- **Portfolio Rebalancing**: Automated allocation adjustments
- **Performance Analytics**: Returns, drawdown, Sharpe ratio
- **Tax Optimization**: FIFO/LIFO, tax-loss harvesting

### 🔍 Analysis & Research
- **Technical Indicators**: 30+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Backtesting Engine**: Historical strategy validation
- **Market Screening**: Stock filtering and ranking
- **Chart Analysis**: Interactive candlestick charts

### 🖥️ User Interface
- **NiceGUI Interface**: Modern responsive web UI
- **Real-time Updates**: WebSocket-based live data
- **Mobile Responsive**: Works on all devices

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[NiceGUI Interface]
        B[Portfolio Management]
        C[Static Assets]
    end
    
    subgraph "Backend Layer"
        D[FastAPI REST API]
        E[Trading Services]
        F[Strategy Engine]
        G[WebSocket Server]
    end
    
    subgraph "Data Layer"
        H[PostgreSQL Database]
        I[Broker APIs]
        J[Market Data Sources]
    end
    
    A --> D
    B --> D
    D --> E
    E --> F
    E --> H
    E --> I
    F --> H
    G --> A
    G --> B
    I --> J
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL (or use Docker)
- Broker API credentials (Upstox/Zerodha)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/algo-trading-app.git
cd algo-trading-app
```

### 2. Environment Setup
Create `.env` file in project root:
```env
# Database
DATABASE_URL=postgresql://trading_user:password123@localhost:5432/trading_db

# Upstox API
UPSTOX_API_KEY=your_upstox_api_key
UPSTOX_API_SECRET=your_upstox_api_secret

# Zerodha API
ZERODHA_API_KEY=your_zerodha_api_key
ZERODHA_API_SECRET=your_zerodha_api_secret

# JWT Security
SECRET_KEY=your_super_secret_jwt_key
ALGORITHM=HS256

# Email Notifications (Optional)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_email_password
```

### 3. Docker Deployment (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Manual Setup (Development)

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Setup database
python app/create_tables.py

# Start backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
pip install -r requirements.txt

# Start NiceGUI interface
python app_ui.py
```

## 📚 API Documentation

### Authentication
```python
# Login
POST /auth/login
{
    "username": "your_email@example.com",
    "password": "your_password"
}

# Response
{
    "access_token": "jwt_token_here",
    "token_type": "bearer"
}
```

### Trading Operations
```python
# Place Order
POST /orders/place
{
    "broker": "Upstox",
    "trading_symbol": "RELIANCE",
    "transaction_type": "BUY",
    "quantity": 10,
    "order_type": "MARKET",
    "product_type": "CNC"
}

# Get Portfolio
GET /portfolio/{broker}

# Get Market Data
GET /quotes/{instrument_token}
```

### SIP Management
```python
# Create SIP Portfolio
POST /sip/portfolios
{
    "portfolio_name": "Large Cap SIP",
    "symbols": [
        {"symbol": "RELIANCE", "allocation_percentage": 30},
        {"symbol": "TCS", "allocation_percentage": 25},
        {"symbol": "HDFCBANK", "allocation_percentage": 45}
    ],
    "config": {
        "investment_amount": 10000,
        "frequency": "monthly",
        "auto_rebalance": true
    }
}
```

## 🛠️ Development

### Project Structure
```
algo-trading-app/
├── backend/          # FastAPI backend
├── frontend/         # NiceGUI frontend
├── common_utils/     # Shared utilities
├── docker-compose.yml
└── README.md
```

### Adding New Strategies

1. **Create Strategy Class**:
```python
# common_utils/strategies/my_strategy.py
from common_utils.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy")
    
    def generate_signals(self, data):
        # Your strategy logic here
        return signals
```

2. **Register Strategy**:
```python
# backend/app/services.py
from common_utils.strategies.my_strategy import MyCustomStrategy

# Add to strategy registry
strategy_registry['my_custom'] = MyCustomStrategy
```

### Database Schema
```sql
-- Key Tables
users              # User accounts and broker credentials
orders              # Order management
sip_portfolios      # SIP portfolio configurations
trade_history       # Historical trade records
strategies          # Strategy configurations
instruments         # Market instruments
```

## 📊 Trading Strategies

### Pre-built Strategies
- **RSI Mean Reversion**: Buy oversold, sell overbought
- **MACD Crossover**: Trend following based on MACD signals
- **Bollinger Bands**: Mean reversion with volatility bands
- **Moving Average Crossover**: Simple trend following
- **Momentum Strategy**: Price and volume momentum

### Strategy Configuration
```python
strategy_config = {
    "rsi_strategy": {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        "position_size": 0.02
    },
    "macd_strategy": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    }
}
```

## 🔧 Configuration

### Backend Configuration
```python
# backend/app/config.py
DATABASE_URL = "postgresql://user:pass@localhost/db"
SECRET_KEY = "your-secret-key"
TOKEN_EXPIRE_MINUTES = 30
```

### Frontend Configuration
```python
# frontend/config.py
BACKEND_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"
REFRESH_INTERVAL = 5  # seconds
```

## 📈 Performance Monitoring

### Key Metrics
- **Portfolio Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Monitoring Dashboard
Access real-time performance metrics at:
- NiceGUI Interface: `http://localhost:8080`
- API Documentation: `http://localhost:8000/docs`

## 🔒 Security Best Practices

### API Security
- JWT token authentication
- Rate limiting on sensitive endpoints
- Input validation and sanitization
- SQL injection prevention

### Broker API Security
- Encrypted credential storage
- Token rotation and refresh
- Secure environment variable handling
- API key access controls

### Database Security
- Connection pooling with SSL
- Parameterized queries
- Regular backups
- Access logging

## 🚨 Risk Management

### Position Sizing
```python
def calculate_position_size(account_value, risk_per_trade, stop_loss_pct):
    risk_amount = account_value * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return min(position_size, account_value * 0.1)  # Max 10% per position
```

### Stop-Loss Management
- **Fixed Percentage**: 2-5% of entry price
- **ATR-based**: Dynamic based on volatility
- **Technical Levels**: Support/resistance levels
- **Trailing Stops**: Lock in profits as price moves favorably

## 📋 Testing

### Unit Tests
```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Full system test
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Backtesting
```python
# Run backtest
python scripts/backtest.py --strategy rsi --symbol RELIANCE --start 2023-01-01 --end 2024-01-01
```

## 🔄 Deployment

### Production Deployment
```bash
# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With nginx reverse proxy
docker-compose -f docker-compose.prod.yml -f docker-compose.nginx.yml up -d
```

### Environment Variables (Production)
```env
# Production overrides
DEBUG=false
DATABASE_URL=postgresql://prod_user:secure_pass@prod_db:5432/trading_db
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints for all functions
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading in financial markets involves significant risk and can result in financial loss. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Always consult with a qualified financial advisor before making investment decisions.**

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/yourusername/algo-trading-app/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/algo-trading-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/algo-trading-app/discussions)
- **Email**: seenu.saurav@gmail.com

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- [Upstox API](https://upstox.com/developer/) - Trading API
- [Zerodha Kite Connect](https://kite.trade/) - Trading API
- [PostgreSQL](https://postgresql.org/) - Database system

---
**Happy Trading! 📈**