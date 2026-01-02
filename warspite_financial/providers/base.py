"""
Base provider interfaces for warspite_financial library.

This module defines the abstract base classes for data providers and trading providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np


class BaseProvider(ABC):
    """
    Abstract base class for financial data providers.
    
    This class defines the interface that all data providers must implement
    to supply financial market data to the warspite_financial library.
    """
    
    # Standard OHLCV column names for consistent data formatting
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    VALID_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']
    
    def __init__(self):
        """Initialize the base provider."""
        pass
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        """
        Retrieve financial data for a given symbol and date range.
        
        Args:
            symbol: The financial instrument symbol (e.g., 'AAPL', 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval ('1m', '5m', '1h', '1d', '1w', '1mo')
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
            
        Raises:
            ValueError: If symbol is invalid or dates are malformed
            ConnectionError: If provider is unreachable
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from this provider.
        
        Returns:
            List of available symbol strings
            
        Raises:
            ConnectionError: If provider is unreachable
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by this provider.
        
        Args:
            symbol: The symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        pass
    
    # Common functionality for date handling and data formatting
    
    def normalize_date(self, date_input: Union[datetime, date, str]) -> datetime:
        """
        Normalize various date input formats to datetime objects.
        
        Args:
            date_input: Date in various formats (datetime, date, or ISO string)
            
        Returns:
            Normalized datetime object
            
        Raises:
            ValueError: If date format is invalid or unparseable
        """
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, date):
            return datetime.combine(date_input, datetime.min.time())
        elif isinstance(date_input, str):
            try:
                # Try parsing ISO format first
                return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try parsing common date formats
                    return pd.to_datetime(date_input).to_pydatetime()
                except Exception:
                    raise ValueError(f"Unable to parse date: {date_input}")
        else:
            raise ValueError(f"Unsupported date type: {type(date_input)}")
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """
        Validate that the date range is logical and reasonable.
        
        Args:
            start_date: Start date for validation
            end_date: End date for validation
            
        Raises:
            ValueError: If date range is invalid
        """
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Check if dates are too far in the future
        now = datetime.now()
        if start_date > now:
            raise ValueError("Start date cannot be in the future")
        
        # Check for reasonable date range (not more than 50 years)
        max_range_days = 50 * 365
        if (end_date - start_date).days > max_range_days:
            raise ValueError(f"Date range too large. Maximum {max_range_days} days allowed")
    
    def validate_interval(self, interval: str) -> None:
        """
        Validate that the interval is supported.
        
        Args:
            interval: Time interval string to validate
            
        Raises:
            ValueError: If interval is not supported
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}. "
                           f"Supported intervals: {', '.join(self.VALID_INTERVALS)}")
    
    def standardize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardize DataFrame format to ensure consistent OHLCV structure.
        
        Args:
            df: Raw DataFrame from provider
            symbol: Symbol for which data was retrieved
            
        Returns:
            Standardized DataFrame with proper OHLCV columns and datetime index
            
        Raises:
            ValueError: If DataFrame cannot be standardized
        """
        if df is None or df.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        # Ensure DataFrame has datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'Datetime' in df.columns:
                df = df.set_index('Datetime')
            else:
                raise ValueError("DataFrame must have datetime index or Date/Datetime column")
        
        # Standardize column names (case-insensitive matching)
        column_mapping = {}
        df_columns_lower = [col.lower() for col in df.columns]
        
        for required_col in self.REQUIRED_COLUMNS:
            required_lower = required_col.lower()
            if required_lower in df_columns_lower:
                original_col = df.columns[df_columns_lower.index(required_lower)]
                column_mapping[original_col] = required_col
            else:
                # Try common variations
                variations = {
                    'open': ['o', 'open_price'],
                    'high': ['h', 'high_price'],
                    'low': ['l', 'low_price'],
                    'close': ['c', 'close_price', 'adj close', 'adj_close'],
                    'volume': ['vol', 'v', 'volume_traded']
                }
                
                found = False
                for variation in variations.get(required_lower, []):
                    if variation in df_columns_lower:
                        original_col = df.columns[df_columns_lower.index(variation)]
                        column_mapping[original_col] = required_col
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Required column '{required_col}' not found in data for {symbol}")
        
        # Rename columns to standard format
        df = df.rename(columns=column_mapping)
        
        # Ensure we have all required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select only the required columns in the correct order
        df = df[self.REQUIRED_COLUMNS]
        
        # Ensure numeric data types
        for col in self.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if df.empty:
            raise ValueError(f"No valid data remaining after cleaning for symbol: {symbol}")
        
        # Sort by datetime index
        df = df.sort_index()
        
        return df
    
    def validate_data_integrity(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validate the integrity of financial data.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol for error reporting
            
        Raises:
            ValueError: If data integrity checks fail
        """
        if df.empty:
            raise ValueError(f"Empty dataset for symbol: {symbol}")
        
        # Check for negative prices (except for some instruments that can go negative)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] < 0).any():
                raise ValueError(f"Negative prices found in {col} for symbol: {symbol}")
        
        # Check that High >= Low for each row
        if (df['High'] < df['Low']).any():
            raise ValueError(f"High price less than Low price found for symbol: {symbol}")
        
        # Check that High >= Open, Close and Low <= Open, Close
        if ((df['High'] < df['Open']) | (df['High'] < df['Close'])).any():
            raise ValueError(f"High price less than Open/Close price found for symbol: {symbol}")
        
        if ((df['Low'] > df['Open']) | (df['Low'] > df['Close'])).any():
            raise ValueError(f"Low price greater than Open/Close price found for symbol: {symbol}")
        
        # Check for reasonable volume (non-negative)
        if (df['Volume'] < 0).any():
            raise ValueError(f"Negative volume found for symbol: {symbol}")
    
    def get_data_with_validation(self, symbol: str, start_date: Union[datetime, date, str], 
                               end_date: Union[datetime, date, str], interval: str = '1d') -> pd.DataFrame:
        """
        Get data with full validation and standardization.
        
        This method provides a complete wrapper around get_data with all common
        validation and formatting applied.
        
        Args:
            symbol: The financial instrument symbol
            start_date: Start date (various formats accepted)
            end_date: End date (various formats accepted)
            interval: Data interval
            
        Returns:
            Standardized and validated DataFrame
            
        Raises:
            ValueError: If any validation fails
            ConnectionError: If provider is unreachable
        """
        # Normalize and validate inputs
        start_dt = self.normalize_date(start_date)
        end_dt = self.normalize_date(end_date)
        
        self.validate_date_range(start_dt, end_dt)
        self.validate_interval(interval)
        
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Get raw data from provider
        raw_data = self.get_data(symbol, start_dt, end_dt, interval)
        
        # Standardize and validate the data
        standardized_data = self.standardize_dataframe(raw_data, symbol)
        self.validate_data_integrity(standardized_data, symbol)
        
        return standardized_data


class OrderResult:
    """Represents the result of a trading order."""
    
    def __init__(self, order_id: str, success: bool, message: str = ""):
        self.order_id = order_id
        self.success = success
        self.message = message


class AccountInfo:
    """Represents trading account information."""
    
    def __init__(self, account_id: str, balance: float, currency: str):
        self.account_id = account_id
        self.balance = balance
        self.currency = currency


class Position:
    """Represents a trading position."""
    
    def __init__(self, position_id: str, symbol: str, quantity: float, 
                 current_price: float, unrealized_pnl: float):
        self.position_id = position_id
        self.symbol = symbol
        self.quantity = quantity
        self.current_price = current_price
        self.unrealized_pnl = unrealized_pnl


class TradingProvider(BaseProvider):
    """
    Abstract base class for trading providers that support both data retrieval
    and trade execution.
    
    This class extends BaseProvider with trading capabilities for live trading
    through supported brokers and exchanges.
    """
    
    @abstractmethod
    def place_order(self, symbol: str, quantity: float, order_type: str) -> OrderResult:
        """
        Place a trading order.
        
        Args:
            symbol: The financial instrument symbol
            quantity: Order quantity (positive for buy, negative for sell)
            order_type: Type of order ('market', 'limit', etc.)
            
        Returns:
            OrderResult containing order details and status
            
        Raises:
            ValueError: If order parameters are invalid
            ConnectionError: If trading provider is unreachable
            PermissionError: If trading permissions are insufficient
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns:
            AccountInfo with balance and account details
            
        Raises:
            ConnectionError: If trading provider is unreachable
            PermissionError: If account access is denied
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current open positions.
        
        Returns:
            List of Position objects representing open positions
            
        Raises:
            ConnectionError: If trading provider is unreachable
        """
        pass
    
    @abstractmethod
    def close_position(self, position_id: str) -> bool:
        """
        Close a specific position.
        
        Args:
            position_id: Unique identifier for the position to close
            
        Returns:
            True if position was successfully closed, False otherwise
            
        Raises:
            ValueError: If position_id is invalid
            ConnectionError: If trading provider is unreachable
        """
        pass
    
    @abstractmethod
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            True if all positions were successfully closed, False otherwise
            
        Raises:
            ConnectionError: If trading provider is unreachable
        """
        pass