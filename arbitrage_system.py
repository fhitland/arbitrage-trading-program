#!/usr/bin/env python3
"""Real-time Arbitrage Trading System"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import our custom modules
from config import current_config
from logger_config import get_logger

class PriceData:
    """Represents price information for a trading pair on an exchange."""
    def __init__(self, exchange: str, symbol: str, bid: float, ask: float, timestamp: datetime, volume: float = 0.0):
        self.exchange = exchange
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.timestamp = timestamp
        self.volume = volume
        self._validate_data()

    def _validate_data(self):
        """Validate the price data after initialization."""
        if not isinstance(self.bid, (int, float)) or self.bid <= 0:
            raise ValueError(f"Invalid bid price: {self.bid}")
        if not isinstance(self.ask, (int, float)) or self.ask <= 0:
            raise ValueError(f"Invalid ask price: {self.ask}")
        if not isinstance(self.volume, (int, float)) or self.volume < 0:
            raise ValueError(f"Invalid volume: {self.volume}")
        if not isinstance(self.exchange, str) or not self.exchange:
            raise ValueError(f"Invalid exchange: {self.exchange}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError(f"Invalid symbol: {self.symbol}")
        if not isinstance(self.timestamp, datetime):
            raise ValueError(f"Invalid timestamp: {self.timestamp}")

class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity between two exchanges."""
    def __init__(self, buy_exchange: str, sell_exchange: str, symbol: str, 
                 buy_price: float, sell_price: float, profit_percentage: float, 
                 profit_absolute: float, timestamp: datetime, confidence: float = 1.0):
        self.buy_exchange = buy_exchange
        self.sell_exchange = sell_exchange
        self.symbol = symbol
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.profit_percentage = profit_percentage
        self.profit_absolute = profit_absolute
        self.timestamp = timestamp
        self.confidence = confidence
        # Basic validation
        if not isinstance(self.buy_price, (int, float)) or self.buy_price <= 0:
            raise ValueError(f"Invalid buy_price: {self.buy_price}")
        if not isinstance(self.sell_price, (int, float)) or self.sell_price <= 0:
            raise ValueError(f"Invalid sell_price: {self.sell_price}")
        if not isinstance(self.profit_percentage, (int, float)): # profit can be negative
             raise ValueError(f"Invalid profit_percentage: {self.profit_percentage}")
        if not isinstance(self.profit_absolute, (int, float)): # profit can be negative
             raise ValueError(f"Invalid profit_absolute: {self.profit_absolute}")

class PortfolioTracker:
    """Tracks portfolio balance and profits over time."""
    def __init__(self, initial_balance=10000.0, currency="USD"):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.currency = currency
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.start_time = datetime.now()
        self.profit_history = []
        self.logger = get_logger("portfolio_tracker")
        self.logger.info(f"Starting portfolio tracking with initial balance of {initial_balance} {currency}")

    def execute_opportunity(self, opportunity: ArbitrageOpportunity, trade_amount: float = 1000.0):
        """Simulates executing an arbitrage opportunity and updates the balance."""
        # Don't trade more than we have
        trade_amount = min(trade_amount, self.current_balance)
        
        # Calculate number of coins bought
        coins_bought = trade_amount / opportunity.buy_price
        
        # Calculate amount received when selling those coins
        amount_received = coins_bought * opportunity.sell_price
        
        # Update balance
        profit = amount_received - trade_amount
        self.current_balance += profit
        
        # Record the profit
        self.profit_history.append({
            'timestamp': datetime.now(),
            'profit': profit,
            'profit_percentage': opportunity.profit_percentage,
            'balance': self.current_balance
        })
        
        self.opportunities_executed += 1
        
        self.logger.info(f"Executed trade: {opportunity.symbol} - Profit: {profit:.2f} {self.currency} ({opportunity.profit_percentage:.2f}%) - New balance: {self.current_balance:.2f} {self.currency}")
        
        return profit

    def record_opportunity(self, opportunity: ArbitrageOpportunity):
        """Records an arbitrage opportunity without executing it."""
        self.opportunities_found += 1
        
    def get_summary(self):
        """Returns a summary of portfolio performance."""
        runtime = datetime.now() - self.start_time
        profit = self.current_balance - self.initial_balance
        profit_percentage = (profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'profit': profit,
            'profit_percentage': profit_percentage,
            'runtime': runtime,
            'opportunities_found': self.opportunities_found,
            'opportunities_executed': self.opportunities_executed
        }
        
    def print_summary(self):
        """Prints a summary of portfolio performance."""
        summary = self.get_summary()
        runtime_str = str(summary['runtime']).split('.')[0]  # Remove microseconds
        
        print("\n===== PORTFOLIO SUMMARY =====")
        print(f"Runtime: {runtime_str}")
        print(f"Initial Balance: {summary['initial_balance']:.2f} {self.currency}")
        print(f"Current Balance: {summary['current_balance']:.2f} {self.currency}")
        print(f"Total Profit: {summary['profit']:.2f} {self.currency} ({summary['profit_percentage']:.2f}%)")
        print(f"Opportunities Found: {summary['opportunities_found']}")
        print(f"Opportunities Executed: {summary['opportunities_executed']}")
        print("=============================\n")

class ExchangeAPI:
    """Base class for exchange API interactions."""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = get_logger(f"exchange_api.{name.lower()}")
        
        # Configure session
        self.session.timeout = current_config.EXCHANGE_TIMEOUT
        self.session.headers.update({
            'User-Agent': 'ArbitrageSystem/1.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make an HTTP request to the exchange API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
            
        except Exception as e:
            self.logger.error(f"API request to {url} failed: {e}")
            return None
    
    def get_ticker(self, symbol: str) -> Optional[PriceData]:
        """Fetch current price data for a specific symbol."""
        raise NotImplementedError("Subclasses must implement get_ticker")
    
    def get_all_tickers(self) -> List[PriceData]:
        """Fetch price data for all available symbols."""
        raise NotImplementedError("Subclasses must implement get_all_tickers")

class BinanceAPI(ExchangeAPI):
    """Binance API implementation."""
    
    def __init__(self):
        config = current_config.get_exchange_config("Binance")
        super().__init__("Binance", config.get("base_url", "https://api.binance.com/api/v3"))
    
    def get_ticker(self, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data for a specific symbol from Binance."""
        try:
            data = self._make_request("ticker/bookTicker", {"symbol": symbol})
            if not data:
                return None
                
            bid = float(data['bidPrice'])
            ask = float(data['askPrice'])
            
            # Validate prices
            if bid <= 0 or ask <= 0:
                self.logger.warning(f"Invalid prices received from Binance for {symbol}: bid={bid}, ask={ask}")
                return None
            
            return PriceData(
                exchange=self.name,
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
                volume=float(data.get('bidQty', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from Binance: {e}")
            return None
    
    def get_all_tickers(self) -> List[PriceData]:
        """Fetch all ticker data from Binance."""
        try:
            data = self._make_request("ticker/bookTicker")
            if not data:
                return []
            
            tickers = []
            for item in data:
                try:
                    symbol = item['symbol']
                    if symbol in current_config.PRIORITY_SYMBOLS:
                        bid = float(item['bidPrice'])
                        ask = float(item['askPrice'])
                        
                        # Validate prices
                        if bid <= 0 or ask <= 0:
                            self.logger.warning(f"Invalid prices received from Binance for {symbol}: bid={bid}, ask={ask}")
                            continue
                            
                        ticker = PriceData(
                            exchange=self.name,
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            timestamp=datetime.now(),
                            volume=float(item.get('bidQty', 0))
                        )
                        tickers.append(ticker)
                except Exception:
                    continue
            
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error fetching all tickers from Binance: {e}")
            return []

class KucoinAPI(ExchangeAPI):
    """KuCoin API implementation."""
    
    def __init__(self):
        config = current_config.get_exchange_config("KuCoin")
        super().__init__("KuCoin", config.get("base_url", "https://api.kucoin.com/api/v1"))
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format to KuCoin's format."""
        # Convert BTCUSDT to BTC-USDT
        if len(symbol) < 6:
            return symbol
        base = symbol[:-4]
        quote = symbol[-4:]
        return f"{base}-{quote}"
    
    def get_ticker(self, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data for a specific symbol from KuCoin."""
        try:
            kucoin_symbol = self._convert_symbol_format(symbol)
            data = self._make_request("market/orderbook/level1", {"symbol": kucoin_symbol})
            if not data or 'data' not in data:
                return None
            
            ticker_data = data['data']
            bid = float(ticker_data['bestBid'])
            ask = float(ticker_data['bestAsk'])
            
            # Validate prices
            if bid <= 0 or ask <= 0:
                self.logger.warning(f"Invalid prices received from KuCoin for {symbol}: bid={bid}, ask={ask}")
                return None
            
            return PriceData(
                exchange=self.name,
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
                volume=float(ticker_data.get('size', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from KuCoin: {e}")
            return None
    
    def get_all_tickers(self) -> List[PriceData]:
        """Fetch all ticker data from KuCoin."""
        try:
            data = self._make_request("market/allTickers")
            if not data or 'data' not in data or 'ticker' not in data['data']:
                return []
            
            tickers = []
            for item in data['data']['ticker']:
                try:
                    symbol = item['symbol'].replace('-', '')  # Convert BTC-USDT to BTCUSDT
                    if symbol in current_config.PRIORITY_SYMBOLS:
                        bid = float(item['buy'])
                        ask = float(item['sell'])
                        
                        # Validate prices
                        if bid <= 0 or ask <= 0:
                            self.logger.warning(f"Invalid prices received from KuCoin for {symbol}: bid={bid}, ask={ask}")
                            continue
                            
                        ticker = PriceData(
                            exchange=self.name,
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            timestamp=datetime.now(),
                            volume=float(item.get('vol', 0))
                        )
                        tickers.append(ticker)
                except Exception:
                    continue
            
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error fetching all tickers from KuCoin: {e}")
            return []

class CoinbaseAPI(ExchangeAPI):
    """Coinbase API implementation."""
    
    def __init__(self):
        config = current_config.get_exchange_config("Coinbase")
        super().__init__("Coinbase", config.get("base_url", "https://api.exchange.coinbase.com"))
    
    def _normalize_coinbase_symbol_to_standard(self, coinbase_symbol: str) -> str:
        """Converts Coinbase symbol (e.g., BTC-USD) to standard (e.g., BTCUSD)."""
        return coinbase_symbol.replace("-", "")

    def _convert_standard_symbol_to_coinbase(self, standard_symbol: str) -> Optional[str]:
        """Converts standard symbol (e.g., BTCUSDT) to Coinbase format (e.g., BTC-USD)."""
        if len(standard_symbol) < 6:
            return None
        
        # Extract base and quote currencies
        if standard_symbol.endswith("USDT"):
            base = standard_symbol[:-4]
            quote = "USD"  # Convert USDT to USD for Coinbase
        else:
            # For other formats, try a simple conversion
            base = standard_symbol[:-3]
            quote = standard_symbol[-3:]
        
        return f"{base}-{quote}"
    
    def get_ticker(self, symbol: str) -> Optional[PriceData]:
        """Fetch ticker data for a specific symbol from Coinbase."""
        try:
            coinbase_symbol = self._convert_standard_symbol_to_coinbase(symbol)
            if not coinbase_symbol:
                self.logger.error(f"Could not convert {symbol} to Coinbase format")
                return None
                
            # Fetch product ticker
            data = self._make_request(f"products/{coinbase_symbol}/ticker")
            if not data:
                return None
            
            # Get the order book for bid/ask
            order_book = self._make_request(f"products/{coinbase_symbol}/book", {"level": 1})
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return None
                
            bid = float(order_book['bids'][0][0]) if order_book['bids'] else 0
            ask = float(order_book['asks'][0][0]) if order_book['asks'] else 0
            
            # Validate prices
            if bid <= 0 or ask <= 0:
                self.logger.warning(f"Invalid prices received from Coinbase for {symbol}: bid={bid}, ask={ask}")
                return None
            
            # Convert back to original symbol format
            return PriceData(
                exchange=self.name,
                symbol=symbol,  # Use original symbol for consistency
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
                volume=float(data.get('volume', 0))
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from Coinbase: {e}")
            return None
    
    def get_all_tickers(self) -> List[PriceData]:
        """Fetch ticker data for the configured symbols from Coinbase."""
        try:
            # Fetch all products first to get available markets
            products = self._make_request("products")
            if not products:
                return []
                
            # Create a mapping of standard symbols to Coinbase symbols
            symbol_map = {}
            for product in products:
                if 'id' in product:
                    # Keep only the active products with base and quote currencies
                    if product.get('status') == 'online' and '-' in product['id']:
                        coinbase_symbol = product['id']
                        standard_symbol = self._normalize_coinbase_symbol_to_standard(coinbase_symbol)
                        # Check if this matches our priority symbols
                        for priority in current_config.PRIORITY_SYMBOLS:
                            # Handle USDT->USD conversion for matching
                            if priority.replace('USDT', 'USD') == standard_symbol:
                                symbol_map[priority] = coinbase_symbol
            
            # Fetch tickers for the matched symbols
            tickers = []
            for standard_symbol, coinbase_symbol in symbol_map.items():
                try:
                    # Get the order book for bid/ask
                    order_book = self._make_request(f"products/{coinbase_symbol}/book", {"level": 1})
                    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                        continue
                        
                    bid = float(order_book['bids'][0][0]) if order_book['bids'] else 0
                    ask = float(order_book['asks'][0][0]) if order_book['asks'] else 0
                    
                    # Validate prices
                    if bid <= 0 or ask <= 0:
                        self.logger.warning(f"Invalid prices received from Coinbase for {coinbase_symbol}: bid={bid}, ask={ask}")
                        continue
                        
                    ticker = PriceData(
                        exchange=self.name,
                        symbol=standard_symbol,  # Use standard symbol for consistency
                        bid=bid,
                        ask=ask,
                        timestamp=datetime.now(),
                        volume=0.0  # We don't get volume from the book endpoint
                    )
                    tickers.append(ticker)
                except Exception as e:
                    self.logger.error(f"Error fetching {coinbase_symbol} from Coinbase: {e}")
                    continue
            
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error fetching all tickers from Coinbase: {e}")
            return []

class ArbitrageDetector:
    """Real-time arbitrage detection engine."""
    
    def __init__(self, exchanges: List[ExchangeAPI]):
        self.exchanges = exchanges
        self.min_profit_percentage = current_config.MIN_PROFIT_PERCENTAGE
        self.max_profit_percentage = current_config.MAX_PROFIT_PERCENTAGE
        self.logger = get_logger("arbitrage_detector")
    
    def fetch_all_prices(self) -> Dict[str, List[PriceData]]:
        """Fetch prices from all exchanges concurrently."""
        prices_by_exchange = {}
        
        def fetch_from_exchange(exchange):
            exchange_name = exchange.name
            try:
                self.logger.debug(f"Fetching prices from {exchange_name}")
                prices = exchange.get_all_tickers()
                if prices:
                    self.logger.debug(f"Received {len(prices)} price entries from {exchange_name}")
                    return exchange_name, prices
                else:
                    self.logger.warning(f"No price data received from {exchange_name}")
                    return None
            except Exception as e:
                self.logger.error(f"Error fetching prices from {exchange_name}: {e}")
                return None
        
        # Use thread pool to fetch data concurrently
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = [executor.submit(fetch_from_exchange, exchange) for exchange in self.exchanges]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    exchange_name, prices = result
                    if prices:
                        prices_by_exchange[exchange_name] = prices
        
        return prices_by_exchange

    def detect_arbitrage(self, prices_by_exchange: Dict[str, List[PriceData]]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities between different exchanges."""
        try:
            opportunities = []
            
            # Log the input data
            self.logger.debug(f"Detecting arbitrage with data from {len(prices_by_exchange)} exchanges")
            
            # Organize prices by symbol
            symbol_prices = {}
            for exchange_name, prices in prices_by_exchange.items():
                for price_data in prices:
                    try:
                        symbol = price_data.symbol
                        if symbol not in symbol_prices:
                            symbol_prices[symbol] = {}
                        symbol_prices[symbol][exchange_name] = price_data
                    except Exception as e:
                        self.logger.error(f"Error processing price data from {exchange_name}: {e}")
                        continue
            
            # Check each symbol across all exchange pairs
            for symbol, exchange_prices in symbol_prices.items():
                if len(exchange_prices) < 2:
                    self.logger.debug(f"Skipping {symbol} - only {len(exchange_prices)} exchanges available")
                    continue
                
                exchange_names = list(exchange_prices.keys())
                
                # Compare every pair of exchanges
                for i in range(len(exchange_names)):
                    for j in range(i + 1, len(exchange_names)):
                        try:
                            exchange_a = exchange_names[i]
                            exchange_b = exchange_names[j]
                            
                            price_a = exchange_prices[exchange_a]
                            price_b = exchange_prices[exchange_b]
                            
                            # Check both directions
                            self._check_opportunity(price_a, price_b, opportunities)
                            self._check_opportunity(price_b, price_a, opportunities)
                            
                        except Exception as e:
                            self.logger.error(f"Error comparing exchanges for {symbol}: {e}")
                            continue
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in detect_arbitrage: {e}")
            return []

    def _check_opportunity(self, buy_price: PriceData, sell_price: PriceData, opportunities: List[ArbitrageOpportunity]):
        """Check if there's an arbitrage opportunity between two prices."""
        try:
            # Validate prices are not zero
            if buy_price.ask <= 0 or sell_price.bid <= 0:
                return

            if sell_price.bid <= buy_price.ask:
                return

            # Calculate profit
            gross_profit = sell_price.bid - buy_price.ask
            fees = (buy_price.ask + sell_price.bid) * current_config.TRADING_FEE_PERCENTAGE
            slippage = (buy_price.ask + sell_price.bid) * current_config.SLIPPAGE_PERCENTAGE
            net_profit = gross_profit - fees - slippage

            if buy_price.ask == 0:
                raise ValueError(f"buy_price.ask is zero, cannot calculate percentage")

            profit_percentage = (net_profit / buy_price.ask) * 100

            if profit_percentage >= self.min_profit_percentage and profit_percentage <= self.max_profit_percentage:
                opportunity = ArbitrageOpportunity(
                    buy_exchange=buy_price.exchange,
                    sell_exchange=sell_price.exchange,
                    symbol=buy_price.symbol,
                    buy_price=buy_price.ask,
                    sell_price=sell_price.bid,
                    profit_percentage=profit_percentage,
                    profit_absolute=net_profit,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)

        except Exception as e:
            self.logger.error(f"Error in _check_opportunity: {e}")

    def run_detection_cycle(self) -> List[ArbitrageOpportunity]:
        """Execute one complete cycle of arbitrage detection."""
        try:
            # Fetch current prices
            prices_by_exchange = self.fetch_all_prices()

            if not prices_by_exchange:
                self.logger.info("No price data received from any exchange in this cycle.")
                return []
                
            # Detect opportunities
            opportunities = self.detect_arbitrage(prices_by_exchange)
            
            if opportunities:
                self.logger.info(f"Found {len(opportunities)} arbitrage opportunities this cycle")
                
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error during detection cycle: {e}")
            return []

def main():
    """Main execution function."""
    logger = get_logger("main")
    logger.info("Starting Real-time Arbitrage Detection System")

    # Parse command-line arguments (if provided)
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Arbitrage Trading System")
    parser.add_argument("--runtime", type=int, default=0, help="Runtime in minutes (0 for indefinite)")
    parser.add_argument("--balance", type=float, default=10000, help="Initial portfolio balance")
    parser.add_argument("--trade-amount", type=float, default=1000, help="Amount to use per trade")
    parser.add_argument("--auto-trade", action="store_true", help="Execute trades automatically")
    args = parser.parse_args()

    try:
        # Initialize exchanges
        exchanges = [
            BinanceAPI(),
            KucoinAPI(), 
            CoinbaseAPI()
        ]
        logger.info(f"Initialized with {len(exchanges)} exchange(s).")

        # Create detector and portfolio tracker
        detector = ArbitrageDetector(exchanges)
        portfolio = PortfolioTracker(initial_balance=args.balance)
        
        logger.info(f"Monitoring {len(exchanges)} exchanges")
        logger.info(f"Minimum profit threshold: {detector.min_profit_percentage}%")
        print(f"Starting with initial balance: {args.balance:.2f} USD")
        print(f"Set to run for: {'indefinitely' if args.runtime == 0 else f'{args.runtime} minutes'}")
        
        # Calculate end time if specified
        end_time = None
        if args.runtime > 0:
            end_time = datetime.now() + timedelta(minutes=args.runtime)
            logger.info(f"System will stop at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        # Performance tracking
        cycle_count = 0
        last_summary_time = datetime.now()
        summary_interval = timedelta(minutes=5)  # Print summary every 5 minutes
        
        # Main loop
        while True:
            try:
                # Check if runtime limit reached
                if end_time and datetime.now() >= end_time:
                    logger.info(f"Runtime limit of {args.runtime} minutes reached")
                    portfolio.print_summary()
                    break
                    
                # Run detection cycle
                opportunities = detector.run_detection_cycle()
                cycle_count += 1
                
                # Process opportunities
                if opportunities:
                    print(f"\nFound {len(opportunities)} opportunities!")
                    for opp in sorted(opportunities, key=lambda x: x.profit_percentage, reverse=True)[:3]:
                        print(f"  {opp.symbol}: {opp.profit_percentage:.2f}% profit")
                        print(f"    Buy on {opp.buy_exchange} @ {opp.buy_price:.6f}")
                        print(f"    Sell on {opp.sell_exchange} @ {opp.sell_price:.6f}")
                        
                        # Record all opportunities
                        portfolio.record_opportunity(opp)
                        
                        # Execute trade if auto-trade enabled
                        if args.auto_trade:
                            profit = portfolio.execute_opportunity(opp, args.trade_amount)
                            print(f"    Executed trade! Profit: {profit:.2f} USD")
                else:
                    print(".", end="", flush=True)
                
                # Print summary periodically
                if datetime.now() - last_summary_time >= summary_interval:
                    portfolio.print_summary()
                    last_summary_time = datetime.now()
                
                time.sleep(1)  # Wait 1 second between cycles
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                portfolio.print_summary()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()