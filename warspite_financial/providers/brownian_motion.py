"""
Brownian Motion provider implementation for warspite_financial library.

This module implements the BrownianMotionProvider class that generates synthetic
financial data using Brownian motion (random walk) for testing purposes.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging

from .base import BaseProvider

# Set up logging
logger = logging.getLogger(__name__)


class BrownianMotionProvider(BaseProvider):
    """
    Brownian Motion data provider for testing purposes.
    
    This provider generates synthetic OHLCV data using Brownian motion (random walk)
    based on a configurable seed for reproducible results. It accepts any valid
    symbol string and provides data for standard Monday-Friday trading days.
    
    The generated data follows realistic financial market patterns with:
    - Configurable volatility and drift
    - Proper OHLC relationships (High >= Open,Close >= Low)
    - Realistic volume patterns
    - Weekend gaps (no Saturday/Sunday data)
    """
    
    def __init__(self, seed: Optional[int] = None, base_price: float = 100.0, 
                 volatility: float = 0.02, drift: float = 0.0001):
        """
        Initialize the Brownian Motion provider.
        
        Args:
            seed: Random seed for reproducible results (default: None for random)
            base_price: Starting price for generated data (default: 100.0)
            volatility: Daily volatility as a fraction (default: 0.02 = 2%)
            drift: Daily drift as a fraction (default: 0.0001 = 0.01%)
        """
        super().__init__()
        
        self.seed = seed
        self.base_price = base_price
        self.volatility = volatility
        self.drift = drift
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Cache for generated data to ensure consistency
        self._data_cache = {}
        
        logger.info(f"BrownianMotionProvider initialized with seed={seed}, "
                   f"base_price={base_price}, volatility={volatility}, drift={drift}")
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        """
        Generate synthetic financial data using Brownian motion.
        
        Args:
            symbol: Symbol to generate data for (e.g., 'AAPL', 'TEST', 'STOCK1')
            start_date: Start date for data generation
            end_date: End date for data generation
            interval: Data interval (only '1d' supported for now)
            
        Returns:
            DataFrame with synthetic OHLCV data indexed by datetime
            
        Raises:
            ValueError: If symbol is invalid or interval not supported
        """
        # Validate symbol
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}. Symbol must be a non-empty string")
        
        # Validate interval - support both daily and hourly
        if interval not in ['1d', '1h']:
            raise ValueError(f"Unsupported interval: {interval}. Only '1d' and '1h' are supported")
        
        # Generate cache key
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        # Check cache first
        if cache_key in self._data_cache:
            logger.debug(f"Returning cached data for {cache_key}")
            return self._data_cache[cache_key].copy()
        
        # Generate trading days/hours based on interval
        if interval == '1d':
            trading_periods = self._generate_trading_days(start_date, end_date)
        else:  # '1h'
            trading_periods = self._generate_trading_hours(start_date, end_date)
        
        if not trading_periods:
            raise ValueError(f"No trading periods found between {start_date} and {end_date}")
        
        # Generate price data using Brownian motion
        prices = self._generate_brownian_motion_prices(symbol, len(trading_periods))
        
        # Generate OHLCV data
        ohlcv_data = self._generate_ohlcv_from_prices(prices, trading_periods)
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_data, index=pd.DatetimeIndex(trading_periods))
        df.index.name = 'Date'
        
        # Cache the result
        self._data_cache[cache_key] = df.copy()
        
        logger.info(f"Generated {len(df)} days of data for {symbol}")
        return df
    
    def _generate_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Generate list of trading days (Monday-Friday) between start and end dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of datetime objects for trading days
        """
        trading_days = []
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            # Monday=0, Sunday=6, so weekdays are 0-4
            if current_date.weekday() < 5:  # Monday to Friday
                trading_days.append(datetime.combine(current_date, datetime.min.time()))
            current_date += timedelta(days=1)
        
        return trading_days
    
    def _generate_trading_hours(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Generate list of trading hours (9 AM to 4 PM, Monday-Friday) between start and end dates.
        
        Args:
            start_date: Start date for trading hour generation
            end_date: End date for trading hour generation
            
        Returns:
            List of datetime objects representing trading hours
        """
        trading_hours = []
        current_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        while current_date <= end_date:
            # Only include weekdays (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                # Generate hours from 9 AM to 4 PM (market hours)
                for hour in range(9, 17):  # 9 AM to 4 PM
                    hour_time = current_date.replace(hour=hour)
                    if hour_time <= end_date:
                        trading_hours.append(hour_time)
            
            # Move to next day
            current_date += timedelta(days=1)
            current_date = current_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        return trading_hours
    
    def _generate_brownian_motion_prices(self, symbol: str, num_days: int) -> np.ndarray:
        """
        Generate price series using Brownian motion.
        
        Args:
            symbol: Symbol for seeding (ensures different symbols have different patterns)
            num_days: Number of trading days to generate
            
        Returns:
            Array of closing prices
        """
        # Use symbol hash for additional randomness while maintaining reproducibility
        symbol_seed = hash(symbol) % 1000000
        if self.seed is not None:
            np.random.seed(self.seed + symbol_seed)
        
        # Generate random returns using normal distribution
        returns = np.random.normal(
            loc=self.drift,  # Mean return (drift)
            scale=self.volatility,  # Standard deviation (volatility)
            size=num_days
        )
        
        # Convert returns to prices using cumulative product
        # Price[t] = Price[0] * exp(sum(returns[0:t]))
        log_prices = np.log(self.base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        return prices
    
    def _generate_ohlcv_from_prices(self, close_prices: np.ndarray, 
                                   trading_days: List[datetime]) -> Dict[str, List[float]]:
        """
        Generate realistic OHLCV data from closing prices.
        
        Args:
            close_prices: Array of closing prices
            trading_days: List of trading days
            
        Returns:
            Dictionary with OHLCV data
        """
        num_days = len(close_prices)
        
        # Initialize arrays
        opens = np.zeros(num_days)
        highs = np.zeros(num_days)
        lows = np.zeros(num_days)
        closes = close_prices.copy()
        volumes = np.zeros(num_days)
        
        # First day: open equals close (no gap)
        opens[0] = closes[0]
        
        # Subsequent days: open with small gap from previous close
        for i in range(1, num_days):
            # Small overnight gap (usually smaller than daily volatility)
            gap = np.random.normal(0, self.volatility * 0.3)
            opens[i] = closes[i-1] * (1 + gap)
        
        # Generate intraday high/low for each day
        for i in range(num_days):
            open_price = opens[i]
            close_price = closes[i]
            
            # Generate intraday volatility (fraction of daily volatility)
            intraday_vol = self.volatility * np.random.uniform(0.5, 1.5)
            
            # High and low should encompass open and close
            min_price = min(open_price, close_price)
            max_price = max(open_price, close_price)
            
            # Extend range based on intraday volatility
            price_range = max_price - min_price
            if price_range == 0:
                price_range = open_price * intraday_vol
            
            # Generate high (above max of open/close)
            high_extension = np.random.exponential(price_range * 0.5)
            highs[i] = max_price + high_extension
            
            # Generate low (below min of open/close)
            low_extension = np.random.exponential(price_range * 0.5)
            lows[i] = max(0.01, min_price - low_extension)  # Ensure positive prices
            
            # Generate volume (log-normal distribution for realistic volume patterns)
            base_volume = 1000000  # 1M shares base
            volume_multiplier = np.random.lognormal(0, 0.5)  # Log-normal for realistic distribution
            volumes[i] = int(base_volume * volume_multiplier)
        
        return {
            'Open': opens.tolist(),
            'High': highs.tolist(),
            'Low': lows.tolist(),
            'Close': closes.tolist(),
            'Volume': volumes.tolist()
        }
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols (sample symbols for testing).
        
        Returns:
            List of sample symbols for testing
        """
        sample_symbols = [
            'TEST', 'STOCK1', 'STOCK2', 'STOCK3', 'STOCK4',
            'TECH', 'FINANCE', 'ENERGY', 'HEALTHCARE', 'RETAIL',
            'VOLATILE', 'STABLE', 'GROWTH', 'VALUE', 'DIVIDEND',
            'CRYPTO', 'FOREX', 'COMMODITY', 'INDEX', 'ETF'
        ]
        
        logger.info(f"Returning {len(sample_symbols)} available symbols")
        return sample_symbols
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported (must be a non-empty string).
        
        Args:
            symbol: The symbol to validate
            
        Returns:
            True if symbol is a non-empty string, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        return len(symbol.strip()) > 0
    
    def set_parameters(self, base_price: Optional[float] = None, 
                      volatility: Optional[float] = None, 
                      drift: Optional[float] = None,
                      seed: Optional[int] = None) -> None:
        """
        Update provider parameters and clear cache.
        
        Args:
            base_price: New base price (optional)
            volatility: New volatility (optional)
            drift: New drift (optional)
            seed: New random seed (optional)
        """
        if base_price is not None:
            self.base_price = base_price
        if volatility is not None:
            self.volatility = volatility
        if drift is not None:
            self.drift = drift
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        
        # Clear cache since parameters changed
        self._data_cache.clear()
        
        logger.info(f"Parameters updated: base_price={self.base_price}, "
                   f"volatility={self.volatility}, drift={self.drift}, seed={self.seed}")
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        logger.debug("Data cache cleared")
    
    def get_symbol_statistics(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> Dict[str, float]:
        """
        Get statistical information about generated data for a symbol.
        
        Args:
            symbol: Symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with statistical measures
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Get data
        df = self.get_data(symbol, start_date, end_date)
        
        if df.empty:
            return {}
        
        # Calculate statistics
        returns = df['Close'].pct_change().dropna()
        
        stats = {
            'start_price': float(df['Close'].iloc[0]),
            'end_price': float(df['Close'].iloc[-1]),
            'total_return': float((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1),
            'volatility': float(returns.std()),
            'mean_return': float(returns.mean()),
            'max_price': float(df['High'].max()),
            'min_price': float(df['Low'].min()),
            'avg_volume': float(df['Volume'].mean()),
            'num_trading_days': len(df)
        }
        
        return stats