"""
Yahoo Finance provider implementation for warspite_financial library.

This module implements the YFinanceProvider class that wraps the yfinance library
to provide standardized access to Yahoo Finance market data.
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import logging

from .base import BaseProvider

# Set up logging
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance library not available. Install with: pip install yfinance")


class YFinanceProvider(BaseProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider wraps the yfinance library to access Yahoo Finance market data
    and returns standardized pandas DataFrames with OHLCV data.
    
    Supports stocks, ETFs, indices, currencies, and cryptocurrencies available
    through Yahoo Finance.
    """
    
    def __init__(self):
        """
        Initialize the Yahoo Finance provider.
        
        Raises:
            ImportError: If yfinance library is not installed
        """
        super().__init__()
        
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance library is required for YFinanceProvider. "
                "Install with: pip install yfinance"
            )
        
        # Yahoo Finance specific interval mapping
        self._yf_interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',  # Not directly supported, will use 1h
            '1d': '1d',
            '1w': '1wk',
            '1mo': '1mo'
        }
        
        # Cache for symbol validation to avoid repeated API calls
        self._symbol_cache = {}
        
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        """
        Retrieve financial data from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo')
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
            
        Raises:
            ValueError: If symbol is invalid or dates are malformed
            ConnectionError: If Yahoo Finance is unreachable
        """
        try:
            # Validate interval
            self.validate_interval(interval)
            
            # Map to yfinance interval
            yf_interval = self._yf_interval_map.get(interval, interval)
            
            # Handle 4h interval by using 1h and resampling
            if interval == '4h':
                yf_interval = '1h'
            
            # Create yfinance Ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date} with interval {interval}")
            
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=False,  # Keep original prices
                prepost=False,      # No pre/post market data
                actions=False       # No dividends/splits
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data available for symbol {symbol} in the specified date range")
            
            # Handle 4h resampling if needed
            if interval == '4h' and not data.empty:
                data = self._resample_to_4h(data)
            
            # Standardize column names and format
            return self.standardize_dataframe(data, symbol)
            
        except Exception as e:
            if "No data found" in str(e) or "404" in str(e):
                raise ValueError(f"Invalid symbol or no data available: {symbol}")
            elif "Connection" in str(e) or "timeout" in str(e).lower():
                raise ConnectionError(f"Unable to connect to Yahoo Finance: {e}")
            else:
                # Re-raise as ValueError for other yfinance errors
                raise ValueError(f"Error retrieving data for {symbol}: {e}")
    
    def _resample_to_4h(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-hour data to 4-hour intervals.
        
        Args:
            data: DataFrame with 1-hour OHLCV data
            
        Returns:
            DataFrame resampled to 4-hour intervals
        """
        if data.empty:
            return data
            
        # Resample to 4-hour intervals
        resampled = data.resample('4H').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from Yahoo Finance.
        
        Note: Yahoo Finance doesn't provide a comprehensive symbol list API.
        This method returns a sample of commonly traded symbols for demonstration.
        
        Returns:
            List of sample symbol strings
            
        Raises:
            ConnectionError: If Yahoo Finance is unreachable
        """
        # Yahoo Finance doesn't provide a symbol list API
        # Return a representative sample of commonly available symbols
        sample_symbols = [
            # Major US stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL',
            
            # Major indices
            '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
            
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND',
            
            # Currencies (vs USD)
            'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'CADUSD=X',
            
            # Cryptocurrencies
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD',
            
            # Commodities
            'GC=F', 'SI=F', 'CL=F', 'NG=F'
        ]
        
        logger.info(f"Returning sample of {len(sample_symbols)} available symbols")
        return sample_symbols
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by Yahoo Finance.
        
        This method attempts to fetch basic info for the symbol to determine
        if it's valid. Results are cached to avoid repeated API calls.
        
        Args:
            symbol: The symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol = symbol.strip().upper()
        
        # Check cache first
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        try:
            # Try to get basic info for the symbol
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info back
            # Yahoo Finance returns empty dict or dict with 'trailingPegRatio' for invalid symbols
            is_valid = (
                info and 
                len(info) > 1 and 
                'symbol' in info and
                info.get('symbol') == symbol
            ) or (
                # Alternative check: try to get recent data
                info and len(info) > 5  # Valid symbols typically have more info fields
            )
            
            # If info check is inconclusive, try fetching recent data
            if not is_valid:
                try:
                    recent_data = ticker.history(period='5d', interval='1d')
                    is_valid = not recent_data.empty
                except:
                    is_valid = False
            
            # Cache the result
            self._symbol_cache[symbol] = is_valid
            
            logger.debug(f"Symbol validation for {symbol}: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.debug(f"Symbol validation failed for {symbol}: {e}")
            # Cache negative result
            self._symbol_cache[symbol] = False
            return False
    
    def clear_symbol_cache(self) -> None:
        """Clear the symbol validation cache."""
        self._symbol_cache.clear()
        logger.debug("Symbol validation cache cleared")
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: The symbol to get info for
            
        Returns:
            Dictionary containing symbol information
            
        Raises:
            ValueError: If symbol is invalid
            ConnectionError: If Yahoo Finance is unreachable
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            symbol_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
            }
            
            return symbol_info
            
        except Exception as e:
            if "Connection" in str(e) or "timeout" in str(e).lower():
                raise ConnectionError(f"Unable to connect to Yahoo Finance: {e}")
            else:
                raise ValueError(f"Error getting info for {symbol}: {e}")