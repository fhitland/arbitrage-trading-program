"""
Configuration settings for the Real-time Arbitrage Trading System
"""

import os
from typing import Dict, List

class Config:
    """Configuration for the arbitrage trading system."""
    
    # Exchange API Configuration
    EXCHANGE_TIMEOUT = 10  # seconds for API requests
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds between retries
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 100
    REQUEST_DELAY = 0.6  # minimum seconds between requests
    
    # Arbitrage Detection Parameters
    MIN_PROFIT_PERCENTAGE = 0.16  # Just above trading fees + slippage (0.15% combined)
    MAX_PROFIT_PERCENTAGE = 10.0  # sanity check
    
    # Transaction Cost Estimates
    TRADING_FEE_PERCENTAGE = 0.001  # 0.1% typical exchange fee
    SLIPPAGE_PERCENTAGE = 0.0005    # 0.05% estimated slippage
    
    # Monitoring and Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "arbitrage_system.log"
    
    # Exchange-specific settings
    EXCHANGE_CONFIGS = {
        "Binance": {
            "base_url": "https://api.binance.com/api/v3",
            "rate_limit": 1200,
            "typical_fee": 0.001,
            "reliability_score": 0.95
        },
        "KuCoin": {
            "base_url": "https://api.kucoin.com/api/v1", 
            "rate_limit": 600,
            "typical_fee": 0.001,
            "reliability_score": 0.90
        },
        "Coinbase": {
            "base_url": "https://api.exchange.coinbase.com",
            "rate_limit": 300,
            "typical_fee": 0.005,
            "reliability_score": 0.98
        }
    }
    
    # Trading pairs to monitor
    PRIORITY_SYMBOLS = [
        "BTCUSDT", # For all exchanges
        # Remove BTCUSD 
        # Add other Bitcoin pairs if necessary in the future
        # "ETHUSDT",
        # "ADAUSDT",
        # "SOLUSDT",
        # "DOGEUSDT",
        # "MATICUSDT",
        # "LINKUSDT",
        # "DOTUSDT",
    ]
    
    @classmethod
    def get_exchange_config(cls, exchange_name: str) -> Dict:
        """Get configuration for a specific exchange."""
        return cls.EXCHANGE_CONFIGS.get(exchange_name, {})
    
    @classmethod
    def get_priority_symbols(cls) -> List[str]:
        """Get list of symbols to monitor."""
        return cls.PRIORITY_SYMBOLS

# Initialize configuration
current_config = Config()