"""
Unit tests for BrownianMotionProvider.

Tests synthetic data generation, symbol validation, and statistical properties.
Requirements: Testing provider for synthetic data generation
"""

import pytest
from unittest.mock import patch
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from warspite_financial.providers.brownian_motion import BrownianMotionProvider


class TestBrownianMotionProviderInitialization:
    """Test BrownianMotionProvider initialization and setup."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        provider = BrownianMotionProvider()
        
        assert provider.seed is None
        assert provider.base_price == 100.0
        assert provider.volatility == 0.02
        assert provider.drift == 0.0001
        assert provider._data_cache == {}
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        provider = BrownianMotionProvider(
            seed=42,
            base_price=150.0,
            volatility=0.03,
            drift=0.0005
        )
        
        assert provider.seed == 42
        assert provider.base_price == 150.0
        assert provider.volatility == 0.03
        assert provider.drift == 0.0005
        assert provider._data_cache == {}
    
    @patch('numpy.random.seed')
    def test_init_sets_random_seed(self, mock_seed):
        """Test that initialization sets numpy random seed when provided."""
        BrownianMotionProvider(seed=42)
        mock_seed.assert_called_once_with(42)
    
    @patch('numpy.random.seed')
    def test_init_no_seed_no_call(self, mock_seed):
        """Test that initialization doesn't set seed when None."""
        BrownianMotionProvider(seed=None)
        mock_seed.assert_not_called()


class TestBrownianMotionProviderSymbolValidation:
    """Test BrownianMotionProvider symbol validation functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42)
    
    def test_validate_symbol_valid_symbols(self, provider):
        """Test validate_symbol with valid symbols."""
        valid_symbols = [
            'TEST',
            'AAPL',
            'STOCK1',
            'FINANCE',
            'test',  # Case doesn't matter for validation
            'MixedCase',
            'SYMBOL_WITH_UNDERSCORE',
            '123',
            'A',
            'BM-TEST',  # BM- prefix is still valid, just not required
            'XYZ-ABC'   # Any format is valid
        ]
        
        for symbol in valid_symbols:
            assert provider.validate_symbol(symbol) is True, f"Symbol {symbol} should be valid"
    
    def test_validate_symbol_invalid_symbols(self, provider):
        """Test validate_symbol with invalid symbols."""
        invalid_symbols = [
            '',               # Empty string
            '   ',            # Whitespace only
            '\t\n',          # Only whitespace characters
        ]
        
        for symbol in invalid_symbols:
            assert provider.validate_symbol(symbol) is False, f"Symbol {symbol} should be invalid"
    
    def test_validate_symbol_none_and_non_string(self, provider):
        """Test validate_symbol with None and non-string inputs."""
        invalid_inputs = [None, 123, ['TEST'], {'symbol': 'TEST'}]
        
        for invalid_input in invalid_inputs:
            assert provider.validate_symbol(invalid_input) is False
    
    def test_validate_symbol_whitespace_handling(self, provider):
        """Test validate_symbol handles whitespace correctly."""
        # Should handle leading/trailing whitespace
        assert provider.validate_symbol('  TEST  ') is True
        assert provider.validate_symbol('\tTEST\n') is True
    
    def test_validate_symbol_edge_cases(self, provider):
        """Test validate_symbol with edge cases."""
        # Very short valid symbols
        assert provider.validate_symbol('A') is True
        assert provider.validate_symbol('1') is True
        assert provider.validate_symbol('X') is True


class TestBrownianMotionProviderAvailableSymbols:
    """Test BrownianMotionProvider available symbols functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42)
    
    def test_get_available_symbols_returns_list(self, provider):
        """Test that get_available_symbols returns a list of symbols."""
        symbols = provider.get_available_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(symbol, str) for symbol in symbols)
    
    def test_get_available_symbols_contains_expected_symbols(self, provider):
        """Test that get_available_symbols contains expected symbols."""
        symbols = provider.get_available_symbols()
        
        expected_symbols = ['TEST', 'STOCK1', 'TECH', 'FINANCE']
        for symbol in expected_symbols:
            assert symbol in symbols
    
    def test_get_available_symbols_all_valid(self, provider):
        """Test that all returned symbols pass validation."""
        symbols = provider.get_available_symbols()
        
        for symbol in symbols:
            assert provider.validate_symbol(symbol) is True


class TestBrownianMotionProviderTradingDays:
    """Test BrownianMotionProvider trading days generation."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42)
    
    def test_generate_trading_days_weekdays_only(self, provider):
        """Test that trading days generation excludes weekends."""
        # Test a week that includes a weekend
        start_date = datetime(2024, 1, 1)  # Monday
        end_date = datetime(2024, 1, 7)    # Sunday
        
        trading_days = provider._generate_trading_days(start_date, end_date)
        
        # Should have 5 trading days (Mon-Fri)
        assert len(trading_days) == 5
        
        # Check that all days are weekdays (0-4, Monday-Friday)
        for day in trading_days:
            assert day.weekday() < 5, f"Day {day} is not a weekday"
    
    def test_generate_trading_days_date_range(self, provider):
        """Test trading days generation for various date ranges."""
        test_cases = [
            (datetime(2024, 1, 1), datetime(2024, 1, 31), 23),  # January 2024
            (datetime(2024, 2, 1), datetime(2024, 2, 29), 21),  # February 2024 (leap year)
            (datetime(2024, 12, 23), datetime(2024, 12, 29), 5), # Week with holidays
        ]
        
        for start_date, end_date, expected_days in test_cases:
            trading_days = provider._generate_trading_days(start_date, end_date)
            assert len(trading_days) == expected_days
    
    def test_generate_trading_days_single_day(self, provider):
        """Test trading days generation for single day ranges."""
        # Monday
        monday = datetime(2024, 1, 1)
        trading_days = provider._generate_trading_days(monday, monday)
        assert len(trading_days) == 1
        
        # Saturday (weekend)
        saturday = datetime(2024, 1, 6)
        trading_days = provider._generate_trading_days(saturday, saturday)
        assert len(trading_days) == 0
    
    def test_generate_trading_days_empty_range(self, provider):
        """Test trading days generation for weekend-only ranges."""
        # Saturday to Sunday
        start_date = datetime(2024, 1, 6)  # Saturday
        end_date = datetime(2024, 1, 7)    # Sunday
        
        trading_days = provider._generate_trading_days(start_date, end_date)
        assert len(trading_days) == 0


class TestBrownianMotionProviderDataGeneration:
    """Test BrownianMotionProvider data generation functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42, base_price=100.0, volatility=0.02)
    
    def test_get_data_valid_symbol_and_dates(self, provider):
        """Test get_data with valid symbol and date range."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        result = provider.get_data('TEST', start_date, end_date, '1d')
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) > 0
        
        # Should only have weekdays
        for date in result.index:
            assert date.weekday() < 5
    
    def test_get_data_invalid_symbol(self, provider):
        """Test get_data with invalid symbol."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        with pytest.raises(ValueError, match="Invalid symbol.*Symbol must be a non-empty string"):
            provider.get_data('', start_date, end_date, '1d')
    
    def test_get_data_unsupported_interval(self, provider):
        """Test get_data with unsupported interval."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        with pytest.raises(ValueError, match="Unsupported interval.*Only '1d' and '1h' are supported"):
            provider.get_data('TEST', start_date, end_date, '5m')
    
    def test_get_data_no_trading_days(self, provider):
        """Test get_data when no trading days exist in range."""
        # Weekend only
        start_date = datetime(2024, 1, 6)  # Saturday
        end_date = datetime(2024, 1, 7)    # Sunday
        
        with pytest.raises(ValueError, match="No trading periods found"):
            provider.get_data('TEST', start_date, end_date, '1d')
    
    def test_get_data_caching(self, provider):
        """Test that data generation results are cached."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        # First call
        result1 = provider.get_data('TEST', start_date, end_date, '1d')
        
        # Second call should return identical data (from cache)
        result2 = provider.get_data('TEST', start_date, end_date, '1d')
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_get_data_different_symbols_different_data(self, provider):
        """Test that different symbols generate different data."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        result1 = provider.get_data('TEST1', start_date, end_date, '1d')
        result2 = provider.get_data('TEST2', start_date, end_date, '1d')
        
        # Should have same structure but different values
        assert result1.shape == result2.shape
        assert list(result1.columns) == list(result2.columns)
        
        # Values should be different (very unlikely to be identical with different seeds)
        assert not result1['Close'].equals(result2['Close'])
    
    def test_get_data_reproducible_with_seed(self):
        """Test that same seed produces identical results."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        provider1 = BrownianMotionProvider(seed=42)
        provider2 = BrownianMotionProvider(seed=42)
        
        result1 = provider1.get_data('TEST', start_date, end_date, '1d')
        result2 = provider2.get_data('TEST', start_date, end_date, '1d')
        
        pd.testing.assert_frame_equal(result1, result2)


class TestBrownianMotionProviderDataIntegrity:
    """Test BrownianMotionProvider data integrity and financial constraints."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42, base_price=100.0, volatility=0.02)
    
    @pytest.fixture
    def sample_data(self, provider):
        """Generate sample data for testing."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        return provider.get_data('TEST', start_date, end_date, '1d')
    
    def test_ohlc_relationships(self, sample_data):
        """Test that OHLC data maintains proper relationships."""
        df = sample_data
        
        # High should be >= Open, Close, Low
        assert (df['High'] >= df['Open']).all(), "High should be >= Open"
        assert (df['High'] >= df['Close']).all(), "High should be >= Close"
        assert (df['High'] >= df['Low']).all(), "High should be >= Low"
        
        # Low should be <= Open, Close, High
        assert (df['Low'] <= df['Open']).all(), "Low should be <= Open"
        assert (df['Low'] <= df['Close']).all(), "Low should be <= Close"
        assert (df['Low'] <= df['High']).all(), "Low should be <= High"
    
    def test_positive_prices(self, sample_data):
        """Test that all prices are positive."""
        df = sample_data
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            assert (df[col] > 0).all(), f"All {col} prices should be positive"
    
    def test_non_negative_volume(self, sample_data):
        """Test that volume is non-negative."""
        df = sample_data
        assert (df['Volume'] >= 0).all(), "Volume should be non-negative"
    
    def test_reasonable_price_ranges(self, sample_data):
        """Test that generated prices are within reasonable ranges."""
        df = sample_data
        
        # Prices shouldn't deviate too extremely from base price
        # With 2% daily volatility over a month, prices should generally stay within reasonable bounds
        min_expected = 50.0   # Very conservative lower bound
        max_expected = 200.0  # Very conservative upper bound
        
        assert df['Low'].min() > min_expected, "Prices shouldn't go too low"
        assert df['High'].max() < max_expected, "Prices shouldn't go too high"
    
    def test_volume_distribution(self, sample_data):
        """Test that volume follows reasonable distribution."""
        df = sample_data
        
        # Volume should vary (not all the same)
        assert df['Volume'].std() > 0, "Volume should vary across days"
        
        # Volume should be reasonable numbers (can be float due to pandas operations)
        assert df['Volume'].dtype in [np.int64, np.int32, np.float64], "Volume should be numeric"
        
        # All volume values should be positive integers (even if stored as float)
        assert (df['Volume'] > 0).all(), "All volumes should be positive"
        assert (df['Volume'] == df['Volume'].astype(int)).all(), "Volume should be whole numbers"


class TestBrownianMotionProviderStatistics:
    """Test BrownianMotionProvider statistical functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42, base_price=100.0, volatility=0.02)
    
    def test_get_symbol_statistics_valid_symbol(self, provider):
        """Test get_symbol_statistics with valid symbol."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        stats = provider.get_symbol_statistics('TEST', start_date, end_date)
        
        expected_keys = [
            'start_price', 'end_price', 'total_return', 'volatility',
            'mean_return', 'max_price', 'min_price', 'avg_volume', 'num_trading_days'
        ]
        
        for key in expected_keys:
            assert key in stats, f"Statistics should include {key}"
            assert isinstance(stats[key], (int, float)), f"{key} should be numeric"
    
    def test_get_symbol_statistics_invalid_symbol(self, provider):
        """Test get_symbol_statistics with invalid symbol."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            provider.get_symbol_statistics('', start_date, end_date)
    
    def test_get_symbol_statistics_calculations(self, provider):
        """Test that statistics calculations are reasonable."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        stats = provider.get_symbol_statistics('TEST', start_date, end_date)
        
        # Total return should be (end_price / start_price) - 1
        expected_return = (stats['end_price'] / stats['start_price']) - 1
        assert abs(stats['total_return'] - expected_return) < 1e-10
        
        # Max/min prices should be reasonable
        assert stats['max_price'] >= stats['end_price']
        assert stats['max_price'] >= stats['start_price']
        assert stats['min_price'] <= stats['end_price']
        assert stats['min_price'] <= stats['start_price']
        
        # Number of trading days should be positive
        assert stats['num_trading_days'] > 0


class TestBrownianMotionProviderParameterManagement:
    """Test BrownianMotionProvider parameter management functionality."""
    
    def test_set_parameters_updates_values(self):
        """Test that set_parameters updates provider parameters."""
        provider = BrownianMotionProvider(seed=42, base_price=100.0, volatility=0.02)
        
        provider.set_parameters(
            base_price=150.0,
            volatility=0.03,
            drift=0.001,
            seed=123
        )
        
        assert provider.base_price == 150.0
        assert provider.volatility == 0.03
        assert provider.drift == 0.001
        assert provider.seed == 123
    
    def test_set_parameters_partial_update(self):
        """Test that set_parameters can update individual parameters."""
        provider = BrownianMotionProvider(seed=42, base_price=100.0, volatility=0.02, drift=0.0001)
        
        # Update only volatility
        provider.set_parameters(volatility=0.05)
        
        assert provider.base_price == 100.0  # Unchanged
        assert provider.volatility == 0.05   # Changed
        assert provider.drift == 0.0001      # Unchanged
        assert provider.seed == 42           # Unchanged
    
    @patch('numpy.random.seed')
    def test_set_parameters_sets_seed(self, mock_seed):
        """Test that set_parameters sets numpy random seed when seed is updated."""
        provider = BrownianMotionProvider()
        
        provider.set_parameters(seed=123)
        
        mock_seed.assert_called_with(123)
    
    def test_set_parameters_clears_cache(self):
        """Test that set_parameters clears the data cache."""
        provider = BrownianMotionProvider(seed=42)
        
        # Generate some data to populate cache
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        provider.get_data('TEST', start_date, end_date, '1d')
        
        # Cache should have data
        assert len(provider._data_cache) > 0
        
        # Update parameters
        provider.set_parameters(volatility=0.05)
        
        # Cache should be cleared
        assert len(provider._data_cache) == 0
    
    def test_clear_cache(self):
        """Test clear_cache functionality."""
        provider = BrownianMotionProvider(seed=42)
        
        # Generate some data to populate cache
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        provider.get_data('TEST', start_date, end_date, '1d')
        
        # Cache should have data
        assert len(provider._data_cache) > 0
        
        # Clear cache
        provider.clear_cache()
        
        # Cache should be empty
        assert len(provider._data_cache) == 0


class TestBrownianMotionProviderEdgeCases:
    """Test BrownianMotionProvider edge cases and error conditions."""
    
    @pytest.fixture
    def provider(self):
        """Create BrownianMotionProvider instance for testing."""
        return BrownianMotionProvider(seed=42)
    
    def test_very_short_date_range(self, provider):
        """Test data generation for very short date ranges."""
        # Single trading day
        start_date = datetime(2024, 1, 1)  # Monday
        end_date = datetime(2024, 1, 1)    # Same Monday
        
        result = provider.get_data('TEST', start_date, end_date, '1d')
        
        assert len(result) == 1
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def test_very_long_date_range(self, provider):
        """Test data generation for long date ranges."""
        # One year
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        result = provider.get_data('TEST', start_date, end_date, '1d')
        
        # Should have approximately 260 trading days (52 weeks * 5 days)
        assert 250 <= len(result) <= 270  # Allow some flexibility
        assert isinstance(result, pd.DataFrame)
    
    def test_extreme_volatility_parameters(self):
        """Test provider with extreme volatility parameters."""
        # Very high volatility
        provider_high_vol = BrownianMotionProvider(seed=42, volatility=0.5)  # 50% daily volatility
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        result = provider_high_vol.get_data('TEST', start_date, end_date, '1d')
        
        # Should still maintain data integrity despite high volatility
        assert (result['High'] >= result['Low']).all()
        assert (result['High'] >= result['Open']).all()
        assert (result['High'] >= result['Close']).all()
        assert (result['Low'] <= result['Open']).all()
        assert (result['Low'] <= result['Close']).all()
    
    def test_zero_volatility(self):
        """Test provider with zero volatility."""
        provider = BrownianMotionProvider(seed=42, volatility=0.0, drift=0.0)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        result = provider.get_data('TEST', start_date, end_date, '1d')
        
        # With zero volatility and drift, prices should be relatively stable
        # (though some variation may occur due to intraday generation)
        price_range = result['Close'].max() - result['Close'].min()
        base_price_range = provider.base_price * 0.1  # 10% of base price
        
        assert price_range < base_price_range, "Prices should be relatively stable with zero volatility"