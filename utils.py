"""
Utility functions for the Arbitrage Trading System

This module contains helper functions that are used throughout the system.
Centralizing these utilities makes the code more maintainable and reduces
duplication. Think of this as your Swiss Army knife for common operations.
"""

import time
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal, ROUND_HALF_UP
import re

from config import current_config
from logger_config import get_logger

# Create a logger for utility functions
utils_logger = get_logger("utils")

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class PriceFormatter:
    """
    Utility class for consistent price formatting and validation.
    
    Financial applications require precise decimal handling because
    floating-point arithmetic can introduce errors that compound
    over many calculations. This class ensures consistent precision.
    """
    
    @staticmethod
    def format_price(price: Union[float, str, Decimal], decimals: int = 8) -> Decimal:
        """
        Format a price to the specified number of decimal places.
        
        Using Decimal instead of float prevents floating-point precision
        errors that could lead to incorrect arbitrage calculations.
        """
        try:
            if isinstance(price, str):
                price = float(price)
            
            decimal_price = Decimal(str(price))
            quantized = decimal_price.quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP)
            return quantized
        except (ValueError, TypeError) as e:
            utils_logger.error(f"Error formatting price {price}: {e}")
            raise ValidationError(f"Invalid price format: {price}")
    
    @staticmethod
    def format_percentage(percentage: Union[float, Decimal], decimals: int = 3) -> str:
        """Format a percentage for consistent display."""
        try:
            if isinstance(percentage, Decimal):
                formatted = percentage.quantize(Decimal('0.' + '0' * decimals))
            else:
                formatted = round(float(percentage), decimals)
            return f"{formatted}%"
        except (ValueError, TypeError):
            return "N/A%"
    
    @staticmethod
    def format_currency(amount: Union[float, Decimal], currency: str = "USD") -> str:
        """Format currency amounts for consistent display."""
        try:
            if isinstance(amount, Decimal):
                formatted = float(amount)
            else:
                formatted = float(amount)
            
            if currency == "USD":
                return f"${formatted:,.2f}"
            else:
                return f"{formatted:,.6f} {currency}"
        except (ValueError, TypeError):
            return f"N/A {currency}"

class SymbolNormalizer:
    """
    Utility class for normalizing trading pair symbols across different exchanges.
    
    Different exchanges use different formats for the same trading pair:
    - Binance: BTCUSDT
    - KuCoin: BTC-USDT  
    - Coinbase: BTC-USD
    
    This class handles these differences transparently.
    """
    
    # Mapping of common symbol variations
    SYMBOL_MAPPINGS = {
        "BTC": ["BITCOIN", "XBT"],
        "ETH": ["ETHEREUM"],
        "USD": ["USDT", "USDC", "BUSD"],
        "ADA": ["CARDANO"],
        "SOL": ["SOLANA"],
        "DOT": ["POLKADOT"],
        "LINK": ["CHAINLINK"]
    }
    
    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """
        Convert a symbol to our standard format.
        
        Standard format: BASEQUOTE (e.g., BTCUSDT)
        This removes separators and standardizes asset names.
        """
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        
        # Remove common separators and convert to uppercase
        normalized = symbol.replace('-', '').replace('_', '').replace('/', '').upper()
        
        # Apply any specific mappings we've defined
        for standard, variations in cls.SYMBOL_MAPPINGS.items():
            for variation in variations:
                normalized = normalized.replace(variation, standard)
        
        return normalized
    
    @classmethod
    def parse_symbol(cls, symbol: str) -> Dict[str, str]:
        """
        Parse a trading symbol into base and quote assets.
        
        This is trickier than it looks because we need to identify where
        the base asset ends and the quote asset begins.
        """
        normalized = cls.normalize_symbol(symbol)
        
        # Common quote currencies in order of preference for matching
        quote_currencies = ["USDT", "USD", "USDC", "BUSD", "BTC", "ETH", "BNB"]
        
        for quote in quote_currencies:
            if normalized.endswith(quote):
                base = normalized[:-len(quote)]
                if base:  # Make sure we have a base asset
                    return {"base": base, "quote": quote}
        
        # If no standard quote currency found, assume last 3-4 characters are quote
        if len(normalized) >= 6:
            return {"base": normalized[:-4], "quote": normalized[-4:]}
        elif len(normalized) >= 5:
            return {"base": normalized[:-3], "quote": normalized[-3:]}
        
        raise ValidationError(f"Cannot parse symbol: {symbol}")
    
    @classmethod
    def is_valid_symbol(cls, symbol: str) -> bool:
        """Check if a symbol appears to be valid."""
        try:
            cls.parse_symbol(symbol)
            return True
        except ValidationError:
            return False

class RateLimiter:
    """
    Rate limiting utility to prevent API overuse.
    
    Exchange APIs have rate limits, and exceeding them can result in
    temporary bans. This class helps manage request timing to stay
    within limits while maximizing throughput.
    """
    
    def __init__(self, requests_per_minute: int = None):
        """Initialize rate limiter with specified limit."""
        self.limit = requests_per_minute or current_config.REQUESTS_PER_MINUTE
        self.requests = []  # Track request timestamps
        self.min_interval = 60.0 / self.limit  # Minimum seconds between requests
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limits.
        
        This method implements a sliding window rate limiter that
        tracks requests over the past minute and delays if needed.
        """
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Check if we're at the limit
        if len(self.requests) >= self.limit:
            # Calculate how long to wait
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request) + 0.1  # Add small buffer
            
            if wait_time > 0:
                utils_logger.debug(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
        
        # Also enforce minimum interval between requests
        if self.requests:
            time_since_last = now - self.requests[-1]
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
        
        # Record this request
        self.requests.append(time.time())

class DataValidator:
    """
    Validation utilities for market data and trading parameters.
    
    Financial data validation is critical because incorrect data can
    lead to bad trading decisions. This class provides comprehensive
    validation for different types of data we handle.
    """
    
    @staticmethod
    def validate_price_data(price_data: Dict) -> bool:
        """
        Validate that price data contains required fields and reasonable values.
        
        This catches obviously incorrect data before it enters our system,
        preventing bad data from causing incorrect arbitrage signals.
        """
        required_fields = ['exchange', 'symbol', 'bid', 'ask', 'timestamp']
        
        # Check required fields exist
        for field in required_fields:
            if field not in price_data:
                utils_logger.warning(f"Missing required field: {field}")
                return False
        
        try:
            bid = float(price_data['bid'])
            ask = float(price_data['ask'])
            
            # Sanity checks on price values
            if bid <= 0 or ask <= 0:
                utils_logger.warning(f"Invalid price values: bid={bid}, ask={ask}")
                return False
            
            if ask < bid:
                utils_logger.warning(f"Ask price ({ask}) less than bid price ({bid})")
                return False
            
            # Check for reasonable spread (not more than 10%)
            spread_percentage = ((ask - bid) / bid) * 100
            if spread_percentage > 10:
                utils_logger.warning(f"Unreasonably large spread: {spread_percentage:.2f}%")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            utils_logger.warning(f"Invalid price data format: {e}")
            return False
    
    @staticmethod
    def validate_arbitrage_opportunity(opportunity) -> bool:
        """
        Validate that an arbitrage opportunity is reasonable and actionable.
        
        This prevents false positives that could lead to failed trades
        or unrealistic profit expectations.
        """
        try:
            # Check profit percentage is within reasonable bounds
            if opportunity.profit_percentage < 0:
                return False
            
            if opportunity.profit_percentage > current_config.MAX_PROFIT_PERCENTAGE:
                utils_logger.warning(f"Unreasonably high profit: {opportunity.profit_percentage}%")
                return False
            
            # Check that exchanges are different
            if opportunity.buy_exchange == opportunity.sell_exchange:
                return False
            
            # Check that prices make sense
            if opportunity.sell_price <= opportunity.buy_price:
                return False
            
            return True
            
        except AttributeError:
            utils_logger.error("Invalid arbitrage opportunity object")
            return False

class PerformanceTracker:
    """
    Utility for tracking system performance metrics.
    """
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'successful_api_calls': 0,
            'opportunities_detected': 0,
            'trades_simulated': 0,
            'total_profit_simulated': 0.0,
            'start_time': time.time()
        }
    
    def record_api_call(self, success: bool = True):
        """Record an API call for performance tracking."""
        self.metrics['api_calls'] += 1
        if success:
            self.metrics['successful_api_calls'] += 1
    
    def record_opportunity(self, profit_amount: float = 0.0):
        """Record a detected arbitrage opportunity."""
        self.metrics['opportunities_detected'] += 1
        if profit_amount > 0:
            self.metrics['total_profit_simulated'] += profit_amount
    
    def record_trade(self):
        """Record a simulated trade execution."""
        self.metrics['trades_simulated'] += 1

def create_signature(secret: str, params: str) -> str:
    """
    Create HMAC signature for authenticated API requests.
    
    Many exchange APIs require signed requests for trading operations.
    This utility handles the cryptographic signing process.
    """
    return hmac.new(
        secret.encode('utf-8'),
        params.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float with error handling.
    
    API responses can sometimes contain unexpected data types
    or null values. This function handles these cases gracefully.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        utils_logger.debug(f"Could not convert {value} to float, using default {default}")
        return default

def safe_datetime_parse(timestamp_str: str, default: Optional[datetime] = None) -> datetime:
    """
    Parse timestamp strings with multiple format support.
    
    Different APIs return timestamps in different formats.
    This function tries multiple common formats to parse them reliably.
    """
    if default is None:
        default = datetime.now()
    
    if not timestamp_str:
        return default
    
    # Common timestamp formats to try
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # If all parsing attempts fail, try Unix timestamp
    try:
        timestamp = float(timestamp_str)
        if timestamp > 1e10:  # Milliseconds
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp)
    except (ValueError, TypeError):
        pass
    
    utils_logger.warning(f"Could not parse timestamp: {timestamp_str}")
    return default

def calculate_portfolio_stats(trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate portfolio performance statistics from trade history.
    
    This function computes key metrics that are important for
    evaluating trading strategy performance in interviews.
    """
    if not trades:
        return {}
    
    profits = [trade.get('net_profit', 0) for trade in trades]
    returns = [trade.get('profit_percentage', 0) for trade in trades]
    
    total_profit = sum(profits)
    total_trades = len(trades)
    winning_trades = len([p for p in profits if p > 0])
    losing_trades = len([p for p in profits if p < 0])
    
    stats = {
        'total_profit': total_profit,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
        'average_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0
    }
    
    if returns:
        stats['average_return'] = sum(returns) / len(returns)
        stats['best_trade_return'] = max(returns)
        stats['worst_trade_return'] = min(returns)
    
    return stats

# Global instances for common utilities
price_formatter = PriceFormatter()
symbol_normalizer = SymbolNormalizer()
data_validator = DataValidator()
performance_tracker = PerformanceTracker()

# Global rate limiter instance
default_rate_limiter = RateLimiter()