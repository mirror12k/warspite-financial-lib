"""
Unit tests for base interfaces.
"""

import pytest
from abc import ABC
from datetime import datetime, date
import pandas as pd
import numpy as np
from warspite_financial.providers.base import BaseProvider, TradingProvider
from warspite_financial.strategies.base import BaseStrategy


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        # Return sample OHLCV data for testing
        dates = pd.date_range(start_date, end_date, freq='D')
        data = {
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        # Ensure High >= Low and other constraints
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        return df
    
    def get_available_symbols(self) -> list:
        return ['AAPL', 'GOOGL', 'MSFT', 'TEST']
    
    def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.get_available_symbols()


class TestBaseProvider:
    """Test BaseProvider interface."""
    
    def test_base_provider_is_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()
    
    def test_base_provider_inheritance(self):
        """Test that BaseProvider is an ABC."""
        assert issubclass(BaseProvider, ABC)
        
        # Check required abstract methods
        abstract_methods = BaseProvider.__abstractmethods__
        expected_methods = {'get_data', 'get_available_symbols', 'validate_symbol'}
        assert abstract_methods == expected_methods
    
    def test_required_columns_constant(self):
        """Test that REQUIRED_COLUMNS is properly defined."""
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert BaseProvider.REQUIRED_COLUMNS == expected_columns
    
    def test_valid_intervals_constant(self):
        """Test that VALID_INTERVALS is properly defined."""
        expected_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']
        assert BaseProvider.VALID_INTERVALS == expected_intervals


class TestBaseProviderCommonFunctionality:
    """Test BaseProvider common functionality methods."""
    
    @pytest.fixture
    def provider(self):
        """Create a concrete provider instance for testing."""
        return ConcreteProvider()
    
    def test_normalize_date_datetime_input(self, provider):
        """Test normalize_date with datetime input."""
        dt = datetime(2023, 1, 15, 10, 30, 0)
        result = provider.normalize_date(dt)
        assert result == dt
        assert isinstance(result, datetime)
    
    def test_normalize_date_date_input(self, provider):
        """Test normalize_date with date input."""
        d = date(2023, 1, 15)
        result = provider.normalize_date(d)
        expected = datetime(2023, 1, 15, 0, 0, 0)
        assert result == expected
        assert isinstance(result, datetime)
    
    def test_normalize_date_string_input(self, provider):
        """Test normalize_date with string input."""
        # ISO format
        result = provider.normalize_date('2023-01-15')
        expected = datetime(2023, 1, 15, 0, 0, 0)
        assert result == expected
        
        # ISO format with time
        result = provider.normalize_date('2023-01-15T10:30:00')
        expected = datetime(2023, 1, 15, 10, 30, 0)
        assert result == expected
    
    def test_normalize_date_invalid_input(self, provider):
        """Test normalize_date with invalid input."""
        with pytest.raises(ValueError, match="Unable to parse date"):
            provider.normalize_date('invalid-date')
        
        with pytest.raises(ValueError, match="Unsupported date type"):
            provider.normalize_date(123)
    
    def test_validate_date_range_valid(self, provider):
        """Test validate_date_range with valid range."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        # Should not raise any exception
        provider.validate_date_range(start, end)
    
    def test_validate_date_range_invalid_order(self, provider):
        """Test validate_date_range with invalid date order."""
        start = datetime(2023, 1, 31)
        end = datetime(2023, 1, 1)
        with pytest.raises(ValueError, match="Start date must be before end date"):
            provider.validate_date_range(start, end)
    
    def test_validate_date_range_future_start(self, provider):
        """Test validate_date_range with future start date."""
        start = datetime(2030, 1, 1)
        end = datetime(2030, 1, 31)
        with pytest.raises(ValueError, match="Start date cannot be in the future"):
            provider.validate_date_range(start, end)
    
    def test_validate_date_range_too_large(self, provider):
        """Test validate_date_range with too large range."""
        start = datetime(1970, 1, 1)
        end = datetime(2030, 1, 1)
        with pytest.raises(ValueError, match="Date range too large"):
            provider.validate_date_range(start, end)
    
    def test_validate_interval_valid(self, provider):
        """Test validate_interval with valid intervals."""
        for interval in BaseProvider.VALID_INTERVALS:
            # Should not raise any exception
            provider.validate_interval(interval)
    
    def test_validate_interval_invalid(self, provider):
        """Test validate_interval with invalid interval."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.validate_interval('invalid')
        
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.validate_interval('1wk')  # Should be '1w'
    
    def test_standardize_dataframe_valid(self, provider):
        """Test standardize_dataframe with valid DataFrame."""
        # Create test DataFrame with proper OHLCV data
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        data = {
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        df = pd.DataFrame(data, index=dates)
        
        result = provider.standardize_dataframe(df, 'TEST')
        
        assert list(result.columns) == BaseProvider.REQUIRED_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5
        assert result.index.is_monotonic_increasing
    
    def test_standardize_dataframe_case_insensitive_columns(self, provider):
        """Test standardize_dataframe with different case column names."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'open': [100, 101, 102],
            'HIGH': [105, 106, 107],
            'Low': [95, 96, 97],
            'close': [102, 103, 104],
            'VOLUME': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        
        result = provider.standardize_dataframe(df, 'TEST')
        
        assert list(result.columns) == BaseProvider.REQUIRED_COLUMNS
        assert len(result) == 3
    
    def test_standardize_dataframe_column_variations(self, provider):
        """Test standardize_dataframe with column name variations."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'o': [100, 101, 102],
            'h': [105, 106, 107],
            'l': [95, 96, 97],
            'c': [102, 103, 104],
            'vol': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        
        result = provider.standardize_dataframe(df, 'TEST')
        
        assert list(result.columns) == BaseProvider.REQUIRED_COLUMNS
        assert len(result) == 3
    
    def test_standardize_dataframe_missing_columns(self, provider):
        """Test standardize_dataframe with missing required columns."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97]
            # Missing Close and Volume
        }
        df = pd.DataFrame(data, index=dates)
        
        with pytest.raises(ValueError, match="Required column.*not found"):
            provider.standardize_dataframe(df, 'TEST')
    
    def test_standardize_dataframe_empty(self, provider):
        """Test standardize_dataframe with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No data available for symbol"):
            provider.standardize_dataframe(df, 'TEST')
    
    def test_validate_data_integrity_valid(self, provider):
        """Test validate_data_integrity with valid data."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        
        # Should not raise any exception
        provider.validate_data_integrity(df, 'TEST')
    
    def test_validate_data_integrity_negative_prices(self, provider):
        """Test validate_data_integrity with negative prices."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'Open': [-100, 101, 102],  # Negative price
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        
        with pytest.raises(ValueError, match="Negative prices found"):
            provider.validate_data_integrity(df, 'TEST')
    
    def test_validate_data_integrity_high_low_violation(self, provider):
        """Test validate_data_integrity with High < Low violation."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'Open': [100, 101, 102],
            'High': [95, 106, 107],   # High < Low for first row
            'Low': [105, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        
        with pytest.raises(ValueError, match="High price less than Low price"):
            provider.validate_data_integrity(df, 'TEST')
    
    def test_validate_data_integrity_negative_volume(self, provider):
        """Test validate_data_integrity with negative volume."""
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        data = {
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [-1000, 1100, 1200]  # Negative volume
        }
        df = pd.DataFrame(data, index=dates)
        
        with pytest.raises(ValueError, match="Negative volume found"):
            provider.validate_data_integrity(df, 'TEST')
    
    def test_get_data_with_validation_success(self, provider):
        """Test get_data_with_validation with valid inputs."""
        result = provider.get_data_with_validation(
            'TEST', 
            '2023-01-01', 
            '2023-01-05', 
            '1d'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == BaseProvider.REQUIRED_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) > 0
    
    def test_get_data_with_validation_invalid_symbol(self, provider):
        """Test get_data_with_validation with invalid symbol."""
        with pytest.raises(ValueError, match="Invalid symbol"):
            provider.get_data_with_validation(
                'INVALID', 
                '2023-01-01', 
                '2023-01-05', 
                '1d'
            )
    
    def test_get_data_with_validation_invalid_date_range(self, provider):
        """Test get_data_with_validation with invalid date range."""
        with pytest.raises(ValueError, match="Start date must be before end date"):
            provider.get_data_with_validation(
                'TEST', 
                '2023-01-05', 
                '2023-01-01',  # End before start
                '1d'
            )
    
    def test_get_data_with_validation_invalid_interval(self, provider):
        """Test get_data_with_validation with invalid interval."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.get_data_with_validation(
                'TEST', 
                '2023-01-01', 
                '2023-01-05', 
                'invalid'
            )


class TestTradingProvider:
    """Test TradingProvider interface."""
    
    def test_trading_provider_is_abstract(self):
        """Test that TradingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TradingProvider()
    
    def test_trading_provider_inheritance(self):
        """Test that TradingProvider inherits from BaseProvider."""
        assert issubclass(TradingProvider, BaseProvider)
        
        # Check all abstract methods (inherited + new)
        abstract_methods = TradingProvider.__abstractmethods__
        expected_methods = {
            'get_data', 'get_available_symbols', 'validate_symbol',  # from BaseProvider
            'place_order', 'get_account_info', 'get_positions',     # from TradingProvider
            'close_position', 'close_all_positions'
        }
        assert abstract_methods == expected_methods


class TestBaseStrategy:
    """Test BaseStrategy interface."""
    
    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_base_strategy_inheritance(self):
        """Test that BaseStrategy is an ABC."""
        assert issubclass(BaseStrategy, ABC)
        
        # Check required abstract methods
        abstract_methods = BaseStrategy.__abstractmethods__
        expected_methods = {'generate_positions'}
        assert abstract_methods == expected_methods