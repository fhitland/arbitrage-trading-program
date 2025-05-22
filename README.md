# Cryptocurrency Arbitrage Trading System

A real-time arbitrage detection and trading system for cryptocurrency markets. This system monitors price differences across multiple exchanges (Binance, KuCoin, Coinbase) and identifies profitable trading opportunities.

## Features

- **Multi-Exchange Support**: Monitors Bitcoin prices across Binance, KuCoin, and Coinbase
- **Real-Time Arbitrage Detection**: Continuously scans for price disparities between exchanges
- **Profit Calculation**: Accounts for trading fees and estimated slippage
- **Portfolio Tracking**: Monitors theoretical balance and profits over time
- **Configurable Parameters**: Easily adjust profit thresholds, trading pairs, and other settings
- **Comprehensive Logging**: Detailed logs for monitoring system performance

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cryptocurrency-arbitrage-system.git
cd cryptocurrency-arbitrage-system

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python arbitrage_system.py

# Run with custom parameters
python arbitrage_system.py --runtime 60 --balance 5000 --trade-amount 500 --auto-trade
```

### Command Line Options

- `--runtime`: Runtime in minutes (0 for indefinite)
- `--balance`: Initial portfolio balance (default: 10000)
- `--trade-amount`: Amount to use per trade (default: 1000)
- `--auto-trade`: Execute trades automatically (simulation)

## Configuration

Edit `config.py` to modify:

- Trading pairs to monitor
- Minimum profit threshold
- Exchange API settings
- Rate limiting parameters
- Transaction cost estimates

## Project Structure

- `arbitrage_system.py`: Main application file
- `config.py`: Configuration settings
- `utils.py`: Utility functions
- `logger_config.py`: Logging configuration

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.

## License

[MIT License](LICENSE) 