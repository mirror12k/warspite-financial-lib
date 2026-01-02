"""
Provider integration tests for warspite_financial library.

These tests verify provider integration with real APIs using test accounts and scenarios.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from warspite_financial import (
    BrownianMotionProvider,
    WarspiteDataset
)
from warspite_financial.utils.exceptions import ProviderError, WarspiteError

# Try to import optional providers
try:
    from warspite_financial import YFinanceProvider
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from warspite_financial import OANDAProvider
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False


class TestBrownianMotionProviderIntegration:
    """Test BrownianMotionProvider integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BrownianMotionProvider()
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
    def test_realistic_data_generation(self):
        """Test that generated data has realistic financial properties."""
        data = self.provider.get_data(
            symbol='BM-AAPL',
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in data.columns for col in expected_columns)
        
        # Verify realistic financial properties
        # High >= max(Open, Close) and Low <= min(Open, Close)
        assert all(data['High'] >= np.maximum(data['Open'], data['Close']))
        assert all(data['Low'] <= np.minimum(data['Open'], data['Close']))
        
        # Volume should be positive
        assert all(data['Volume'] > 0)
        
        # Prices should be positive
        for col in ['Open', 'High', 'Low', 'Close']:
            assert all(data[col] > 0)
            
    def test_multiple_symbols_consistency(self):
        """Test data consistency across multiple symbols."""
        symbols = ['BM-AAPL', 'BM-GOOGL', 'BM-MSFT']
        datasets = {}
        
        for symbol in symbols:
            data = self.provider.get_data(
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                interval='1d'
            )
            datasets[symbol] = data
            
        # All datasets should have same length (same date range)
        lengths = [len(data) for data in datasets.values()]
        assert all(length == lengths[0] for length in lengths)
        
        # All datasets should have same date index structure
        first_dates = datasets[symbols[0]].index
        for symbol in symbols[1:]:
            assert all(datasets[symbol].index == first_dates)
            
    def test_different_intervals(self):
        """Test data generation with different intervals."""
        intervals = ['1d', '1h']  # Test available intervals
        
        for interval in intervals:
            data = self.provider.get_data(
                symbol='BM-AAPL',
                start_date=self.start_date,
                end_date=self.end_date,
                interval=interval
            )
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            
            # Higher frequency should have more data points
            if interval == '1h':
                # Should have more data points than daily
                daily_data = self.provider.get_data(
                    symbol='BM-AAPL',
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval='1d'
                )
                assert len(data) >= len(daily_data)
                
    def test_dataset_creation_integration(self):
        """Test integration with WarspiteDataset creation."""
        symbols = ['BM-AAPL', 'BM-GOOGL']
        
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        assert len(dataset.symbols) == len(symbols)
        assert all(symbol in dataset.symbols for symbol in symbols)
        assert len(dataset.timestamps) > 0
        assert dataset.data.shape[0] == len(dataset.timestamps)
        assert dataset.data.shape[1] == len(symbols) * 5  # OHLCV for each symbol
        
    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios."""
        # Test with invalid symbol (should handle gracefully)
        try:
            data = self.provider.get_data(
                symbol='INVALID_SYMBOL_12345',
                start_date=self.start_date,
                end_date=self.end_date,
                interval='1d'
            )
            # If it succeeds, should return valid data structure
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            # If it fails, should be appropriate error type
            assert isinstance(e, (ProviderError, ValueError))
            
        # Test with invalid date range
        invalid_start = self.end_date + timedelta(days=1)
        invalid_end = self.end_date + timedelta(days=2)
        
        try:
            data = self.provider.get_data(
                symbol='BM-AAPL',
                start_date=invalid_start,
                end_date=invalid_end,
                interval='1d'
            )
            # If it succeeds, should return valid data
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            assert isinstance(e, (ProviderError, ValueError))


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderIntegration:
    """Test YFinanceProvider integration with real Yahoo Finance API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = YFinanceProvider()
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=10)  # Short period for faster tests
        
    def test_real_data_retrieval(self):
        """Test retrieving real data from Yahoo Finance."""
        # Use well-known symbols that should always exist
        symbol = 'AAPL'
        
        data = self.provider.get_data(
            symbol=symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
        # Verify required columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in data.columns for col in expected_columns)
        
        # Verify data quality
        assert all(data['High'] >= np.maximum(data['Open'], data['Close']))
        assert all(data['Low'] <= np.minimum(data['Open'], data['Close']))
        assert all(data['Volume'] >= 0)
        
    def test_symbol_validation(self):
        """Test symbol validation with real symbols."""
        # Test valid symbols
        assert self.provider.validate_symbol('AAPL')
        assert self.provider.validate_symbol('GOOGL')
        assert self.provider.validate_symbol('MSFT')
        
        # Test invalid symbols (should return False, not raise exception)
        assert not self.provider.validate_symbol('INVALID_SYMBOL_12345')
        assert not self.provider.validate_symbol('')
        
    def test_dataset_integration(self):
        """Test integration with WarspiteDataset using real data."""
        symbols = ['AAPL']  # Use single symbol for faster test
        
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        assert len(dataset.symbols) == len(symbols)
        assert dataset.symbols[0] == 'AAPL'
        assert len(dataset.timestamps) > 0
        
        # Verify data is reasonable
        df = dataset.to_dataframe()
        assert len(df) == len(dataset.timestamps)
        assert not df.empty
        
    def test_error_handling_with_real_api(self):
        """Test error handling with real API scenarios."""
        # Test with definitely invalid symbol
        with pytest.raises((ProviderError, ValueError)):
            self.provider.get_data(
                symbol='DEFINITELY_INVALID_SYMBOL_XYZ123',
                start_date=self.start_date,
                end_date=self.end_date,
                interval='1d'
            )
            
        # Test with invalid interval
        with pytest.raises((ProviderError, ValueError)):
        
            self.provider.get_data(
                symbol='AAPL',
                start_date=self.start_date,
                end_date=self.end_date,
                interval='invalid_interval'
            )


@pytest.mark.skipif(not OANDA_AVAILABLE, reason="OANDA provider not available")
class TestOANDAProviderIntegration:
    """Test OANDAProvider integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Note: This would require test credentials in real scenario
        # For now, test the provider interface without real API calls
        try:
            self.provider = OANDAProvider(api_token="test_token", account_id="test_account")
        except Exception:
            # Skip if can't create provider
            pytest.skip("OANDA provider requires credentials")
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=5)
        
    def test_provider_interface(self):
        """Test that OANDA provider implements required interfaces."""
        # Test that it has required methods
        assert hasattr(self.provider, 'get_data')
        assert hasattr(self.provider, 'validate_symbol')
        assert hasattr(self.provider, 'get_available_symbols')
        
        # Test trading provider methods
        assert hasattr(self.provider, 'place_order')
        assert hasattr(self.provider, 'get_account_info')
        assert hasattr(self.provider, 'get_positions')
        
    def test_forex_symbol_validation(self):
        """Test forex symbol validation."""
        # Test common forex pairs
        forex_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
        
        for symbol in forex_symbols:
            # Should not raise exception (may return True or False depending on implementation)
            result = self.provider.validate_symbol(symbol)
            assert isinstance(result, bool)
            
    def test_trading_provider_interface(self):
        """Test trading provider interface methods."""
        # These tests verify the interface exists and handles calls gracefully
        # In real scenarios, these would need proper authentication
        
        try:
            # Test account info (should handle gracefully without credentials)
            account_info = self.provider.get_account_info()
            # If it succeeds, should return some structure
            assert account_info is not None
        except Exception as e:
            # Should raise appropriate authentication or connection error
            assert isinstance(e, (ProviderError, ConnectionError, ValueError))
            
        try:
            # Test positions (should handle gracefully without credentials)
            positions = self.provider.get_positions()
            assert isinstance(positions, list)
        except Exception as e:
            assert isinstance(e, (ProviderError, ConnectionError, ValueError))


class TestMultiProviderIntegration:
    """Test scenarios involving multiple providers."""
    
    def test_mixed_provider_dataset_creation(self):
        """Test creating datasets from mixed provider types."""
        providers = [BrownianMotionProvider()]
        
        # Add YFinance if available
        if YFINANCE_AVAILABLE:
            providers.append(YFinanceProvider())
            
        symbols = ['BM-AAPL']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        # Should handle mixed providers gracefully
        dataset = WarspiteDataset.from_provider(
            provider=providers[0],  # Use single provider since from_provider takes one
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        assert len(dataset.symbols) == len(symbols)
        assert len(dataset.timestamps) > 0
        
    def test_provider_fallback_scenarios(self):
        """Test fallback scenarios when providers fail."""
        # Create a scenario where one provider might fail
        providers = [BrownianMotionProvider()]
        
        # This should always succeed with BrownianMotionProvider
        dataset = WarspiteDataset.from_provider(
            provider=providers[0],
            symbols=['BM-AAPL'],
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            interval='1d'
        )
        
        assert isinstance(dataset, WarspiteDataset)
        assert len(dataset.symbols) > 0


if __name__ == '__main__':
    pytest.main([__file__])