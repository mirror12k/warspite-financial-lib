"""
Property-based tests for provider interfaces.

These tests verify universal properties that should hold across all valid inputs
for the provider system in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns

from warspite_financial.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider implementation for property testing."""
    
    def __init__(self, symbols=None, fail_on_invalid=True):
        super().__init__()
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TEST', 'SPY']
        self.fail_on_invalid = fail_on_invalid
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Generate date range
        if interval == '1d':
            freq = 'D'
        elif interval == '1h':
            freq = 'h'  # Updated from 'H'
        elif interval == '1w':
            freq = 'W'
        elif interval == '1mo':
            freq = 'ME'  # Updated from 'M'
        else:
            freq = 'D'  # Default fallback
        
        dates = pd.date_range(start_date, end_date, freq=freq)
        if len(dates) == 0:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS, 
                              index=pd.DatetimeIndex([]))
        
        # Generate realistic OHLCV data
        np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol
        base_price = 100.0
        
        data = {}
        for i, date in enumerate(dates):
            # Generate prices with some volatility
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            if i == 0:
                open_price = base_price
            else:
                open_price = data['Close'][i-1] * (1 + np.random.normal(0, 0.005))
            
            close_price = open_price * (1 + price_change)
            
            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            
            # Ensure prices are positive
            high_price = max(high_price, 0.01)
            low_price = max(low_price, 0.01)
            open_price = max(open_price, 0.01)
            close_price = max(close_price, 0.01)
            
            # Generate volume
            volume = max(1, int(np.random.lognormal(8, 1)))  # Log-normal distribution
            
            if i == 0:
                data['Open'] = [open_price]
                data['High'] = [high_price]
                data['Low'] = [low_price]
                data['Close'] = [close_price]
                data['Volume'] = [volume]
            else:
                data['Open'].append(open_price)
                data['High'].append(high_price)
                data['Low'].append(low_price)
                data['Close'].append(close_price)
                data['Volume'].append(volume)
        
        return pd.DataFrame(data, index=dates)
    
    def get_available_symbols(self) -> list:
        return self.symbols.copy()
    
    def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols


# Hypothesis strategies for generating test data
@st.composite
def valid_symbols(draw):
    """Generate valid symbols for testing."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TEST', 'SPY']  # Only use symbols that are in MockProvider
    return draw(st.sampled_from(symbols))


@st.composite
def valid_date_ranges(draw):
    """Generate valid date ranges for testing."""
    # Generate dates within a reasonable range (last 5 years to yesterday)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=draw(st.integers(1, 1825)))  # 1 day to 5 years ago
    
    return start_date, end_date


@st.composite
def valid_intervals(draw):
    """Generate valid intervals for testing."""
    return draw(st.sampled_from(BaseProvider.VALID_INTERVALS))


class TestProviderDataStructureConsistency:
    """
    Property-based tests for Provider Data Structure Consistency.
    
    **Feature: warspite-financial-library, Property 1: Provider Data Structure Consistency**
    **Validates: Requirements 2.3**
    """
    
    @given(
        symbol=valid_symbols(),
        date_range=valid_date_ranges(),
        interval=valid_intervals()
    )
    @settings(max_examples=100, deadline=None)
    def test_provider_data_structure_consistency(self, symbol, date_range, interval):
        """
        Property 1: Provider Data Structure Consistency
        
        For any valid provider and symbol query, the returned data should have 
        consistent structure with required OHLCV columns and proper data types.
        
        **Feature: warspite-financial-library, Property 1: Provider Data Structure Consistency**
        **Validates: Requirements 2.3**
        """
        # Create provider instance for this test
        provider = MockProvider()
        
        start_date, end_date = date_range
        
        # Skip if date range is invalid for the provider's validation
        assume(start_date < end_date)
        assume(start_date < datetime.now())
        assume((end_date - start_date).days <= 365 * 10)  # Max 10 years
        
        # Get data from provider
        result = provider.get_data(symbol, start_date, end_date, interval)
        
        # Property assertions: Data structure consistency
        
        # 1. Result must be a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Provider must return pandas DataFrame"
        
        # 2. DataFrame must have datetime index
        assert isinstance(result.index, pd.DatetimeIndex), "DataFrame must have DatetimeIndex"
        
        # 3. If data is not empty, it must have all required OHLCV columns
        if not result.empty:
            assert all(col in result.columns for col in BaseProvider.REQUIRED_COLUMNS), \
                f"DataFrame must contain all required columns: {BaseProvider.REQUIRED_COLUMNS}"
            
            # 4. All OHLCV columns must be numeric
            for col in BaseProvider.REQUIRED_COLUMNS:
                assert pd.api.types.is_numeric_dtype(result[col]), \
                    f"Column {col} must be numeric"
            
            # 5. No NaN values in required columns
            for col in BaseProvider.REQUIRED_COLUMNS:
                assert not result[col].isna().any(), \
                    f"Column {col} must not contain NaN values"
            
            # 6. All prices must be non-negative
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                assert (result[col] >= 0).all(), \
                    f"All prices in {col} must be non-negative"
            
            # 7. Volume must be non-negative
            assert (result['Volume'] >= 0).all(), \
                "Volume must be non-negative"
            
            # 8. High >= Low for all rows
            assert (result['High'] >= result['Low']).all(), \
                "High price must be >= Low price for all rows"
            
            # 9. High >= Open and High >= Close for all rows
            assert (result['High'] >= result['Open']).all(), \
                "High price must be >= Open price for all rows"
            assert (result['High'] >= result['Close']).all(), \
                "High price must be >= Close price for all rows"
            
            # 10. Low <= Open and Low <= Close for all rows
            assert (result['Low'] <= result['Open']).all(), \
                "Low price must be <= Open price for all rows"
            assert (result['Low'] <= result['Close']).all(), \
                "Low price must be <= Close price for all rows"
            
            # 11. Index must be sorted (monotonic increasing)
            assert result.index.is_monotonic_increasing, \
                "DateTime index must be sorted in ascending order"
            
            # 12. Index must be within the requested date range
            assert result.index.min() >= start_date, \
                "All dates must be >= start_date"
            assert result.index.max() <= end_date, \
                "All dates must be <= end_date"
        
        # 13. Empty DataFrame should still have correct structure
        else:
            # Even empty DataFrames should have the right columns if they're structured
            if len(result.columns) > 0:
                assert all(col in BaseProvider.REQUIRED_COLUMNS for col in result.columns), \
                    "Even empty DataFrame should only contain valid OHLCV columns"
    
    @given(
        symbol=valid_symbols(),
        date_range=valid_date_ranges(),
        interval=valid_intervals()
    )
    @settings(max_examples=50, deadline=None)
    def test_provider_data_validation_wrapper_consistency(self, symbol, date_range, interval):
        """
        Test that get_data_with_validation produces consistent results with get_data.
        
        This ensures the validation wrapper maintains the same data structure properties.
        """
        # Create provider instance for this test
        provider = MockProvider()
        
        start_date, end_date = date_range
        
        # Skip invalid ranges
        assume(start_date < end_date)
        assume(start_date < datetime.now())
        assume(end_date < datetime.now())  # Also ensure end_date is in the past
        assume((end_date - start_date).days <= 365 * 10)
        
        # Get data using both methods
        raw_result = provider.get_data(symbol, start_date, end_date, interval)
        validated_result = provider.get_data_with_validation(symbol, start_date, end_date, interval)
        
        # Both should have the same structure properties
        assert type(raw_result) == type(validated_result), \
            "Both methods should return the same type"
        
        if not raw_result.empty and not validated_result.empty:
            # Should have same columns
            assert list(validated_result.columns) == BaseProvider.REQUIRED_COLUMNS, \
                "Validated result should have standardized columns"
            
            # Should have same basic structure
            assert isinstance(validated_result.index, pd.DatetimeIndex), \
                "Validated result should have DatetimeIndex"
            
            # Should maintain data integrity
            assert validated_result.index.is_monotonic_increasing, \
                "Validated result should be sorted"
    
    @given(st.text(min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_invalid_symbol_handling(self, invalid_symbol):
        """
        Test that invalid symbols are handled consistently.
        
        For any invalid symbol, the provider should either return empty data
        or raise a ValueError, but never return malformed data.
        """
        # Create provider instance for this test
        provider = MockProvider()
        
        # Use a symbol that's definitely not in our test set
        assume(invalid_symbol not in provider.get_available_symbols())
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        try:
            result = provider.get_data(invalid_symbol, start_date, end_date, '1d')
            # If no exception, result should be empty or properly structured
            assert isinstance(result, pd.DataFrame), \
                "Invalid symbol should return DataFrame or raise exception"
            
            if not result.empty:
                # If data is returned, it should still be properly structured
                assert isinstance(result.index, pd.DatetimeIndex), \
                    "Even invalid symbol data should have proper index"
        except ValueError:
            # This is acceptable behavior for invalid symbols
            pass
        except Exception as e:
            # Other exceptions are not acceptable
            pytest.fail(f"Unexpected exception type for invalid symbol: {type(e).__name__}: {e}")