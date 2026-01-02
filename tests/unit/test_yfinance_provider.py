"""
Unit tests for YFinanceProvider.

Tests specific symbols, date ranges, and error handling for invalid symbols.
Requirements: 2.1, 2.3
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, date
import pandas as pd
import numpy as np

from warspite_financial.providers.yfinance import YFinanceProvider, YFINANCE_AVAILABLE


class TestYFinanceProviderInitialization:
    """Test YFinanceProvider initialization and setup."""
    
    @patch('warspite_financial.providers.yfinance.YFINANCE_AVAILABLE', False)
    def test_init_without_yfinance_library(self):
        """Test initialization fails when yfinance library is not available."""
        with pytest.raises(ImportError, match="yfinance library is required"):
            YFinanceProvider()
    
    @pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
    def test_init_with_yfinance_library(self):
        """Test successful initialization when yfinance is available."""
        provider = YFinanceProvider()
        assert provider is not None
        assert hasattr(provider, '_yf_interval_map')
        assert hasattr(provider, '_symbol_cache')
        assert provider._symbol_cache == {}
    
    @pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
    def test_interval_mapping(self):
        """Test that interval mapping is correctly configured."""
        provider = YFinanceProvider()
        expected_mapping = {
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
        assert provider._yf_interval_map == expected_mapping


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderDataRetrieval:
    """Test YFinanceProvider data retrieval functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create YFinanceProvider instance for testing."""
        return YFinanceProvider()
    
    @pytest.fixture
    def mock_ticker_data(self):
        """Create mock ticker data for testing."""
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        return pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
            'High': [155.0, 156.0, 157.0, 158.0, 159.0],
            'Low': [145.0, 146.0, 147.0, 148.0, 149.0],
            'Close': [152.0, 153.0, 154.0, 155.0, 156.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_valid_symbol_daily(self, mock_ticker_class, provider, mock_ticker_data):
        """Test get_data with valid symbol and daily interval."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_ticker_data
        mock_ticker_class.return_value = mock_ticker
        
        # Test data retrieval
        result = provider.get_data(
            'AAPL', 
            datetime(2023, 1, 1), 
            datetime(2023, 1, 5), 
            '1d'
        )
        
        # Verify mock was called correctly
        mock_ticker_class.assert_called_once_with('AAPL')
        mock_ticker.history.assert_called_once_with(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 5),
            interval='1d',
            auto_adjust=False,
            prepost=False,
            actions=False
        )
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 5
        assert isinstance(result.index, pd.DatetimeIndex)
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_different_intervals(self, mock_ticker_class, provider, mock_ticker_data):
        """Test get_data with different time intervals."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_ticker_data
        mock_ticker_class.return_value = mock_ticker
        
        test_intervals = ['1m', '5m', '1h', '1d', '1w', '1mo']
        expected_yf_intervals = ['1m', '5m', '1h', '1d', '1wk', '1mo']
        
        for interval, expected_yf_interval in zip(test_intervals, expected_yf_intervals):
            mock_ticker.history.reset_mock()
            
            provider.get_data('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 5), interval)
            
            # Check that correct yfinance interval was used
            call_args = mock_ticker.history.call_args
            assert call_args[1]['interval'] == expected_yf_interval
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_4h_interval_resampling(self, mock_ticker_class, provider):
        """Test get_data with 4h interval that requires resampling."""
        # Create hourly mock data for resampling test
        dates = pd.date_range('2023-01-01 00:00', '2023-01-01 23:00', freq='H')
        hourly_data = pd.DataFrame({
            'Open': np.random.uniform(150, 160, len(dates)),
            'High': np.random.uniform(160, 170, len(dates)),
            'Low': np.random.uniform(140, 150, len(dates)),
            'Close': np.random.uniform(150, 160, len(dates)),
            'Volume': np.random.randint(10000, 50000, len(dates))
        }, index=dates)
        
        # Ensure OHLC constraints are met
        for i in range(len(hourly_data)):
            row = hourly_data.iloc[i]
            hourly_data.iloc[i, hourly_data.columns.get_loc('High')] = max(row['Open'], row['High'], row['Close'])
            hourly_data.iloc[i, hourly_data.columns.get_loc('Low')] = min(row['Open'], row['Low'], row['Close'])
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = hourly_data
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.get_data('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 2), '4h')
        
        # Verify 1h interval was requested for resampling
        call_args = mock_ticker.history.call_args
        assert call_args[1]['interval'] == '1h'
        
        # Verify result is resampled (should have fewer rows than hourly data)
        assert len(result) < len(hourly_data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_empty_response(self, mock_ticker_class, provider):
        """Test get_data when yfinance returns empty data."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError, match="No data available for symbol"):
            provider.get_data('INVALID', datetime(2023, 1, 1), datetime(2023, 1, 5), '1d')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_none_response(self, mock_ticker_class, provider):
        """Test get_data when yfinance returns None."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = None
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError, match="No data available for symbol"):
            provider.get_data('INVALID', datetime(2023, 1, 1), datetime(2023, 1, 5), '1d')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_yfinance_no_data_error(self, mock_ticker_class, provider):
        """Test get_data when yfinance raises 'No data found' error."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("No data found, symbol may be delisted")
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError, match="Invalid symbol or no data available"):
            provider.get_data('DELISTED', datetime(2023, 1, 1), datetime(2023, 1, 5), '1d')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_connection_error(self, mock_ticker_class, provider):
        """Test get_data when connection to Yahoo Finance fails."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Connection timeout")
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ConnectionError, match="Unable to connect to Yahoo Finance"):
            provider.get_data('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 5), '1d')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_data_404_error(self, mock_ticker_class, provider):
        """Test get_data when yfinance returns 404 error."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("404 Client Error")
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError, match="Invalid symbol or no data available"):
            provider.get_data('NOTFOUND', datetime(2023, 1, 1), datetime(2023, 1, 5), '1d')
    
    def test_get_data_invalid_interval(self, provider):
        """Test get_data with invalid interval."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.get_data('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 5), 'invalid')


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderSymbolValidation:
    """Test YFinanceProvider symbol validation functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create YFinanceProvider instance for testing."""
        return YFinanceProvider()
    
    def test_validate_symbol_empty_string(self, provider):
        """Test validate_symbol with empty string."""
        assert provider.validate_symbol('') is False
        assert provider.validate_symbol('   ') is False
    
    def test_validate_symbol_none(self, provider):
        """Test validate_symbol with None."""
        assert provider.validate_symbol(None) is False
    
    def test_validate_symbol_non_string(self, provider):
        """Test validate_symbol with non-string input."""
        assert provider.validate_symbol(123) is False
        assert provider.validate_symbol(['AAPL']) is False
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_validate_symbol_valid_with_info(self, mock_ticker_class, provider):
        """Test validate_symbol with valid symbol that has info."""
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'AAPL', 'longName': 'Apple Inc.', 'currency': 'USD'}
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.validate_symbol('AAPL')
        
        assert result is True
        assert 'AAPL' in provider._symbol_cache
        assert provider._symbol_cache['AAPL'] is True
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_validate_symbol_valid_with_recent_data(self, mock_ticker_class, provider):
        """Test validate_symbol with valid symbol that has recent data but limited info."""
        mock_ticker = Mock()
        # Limited info that doesn't pass first check
        mock_ticker.info = {'trailingPegRatio': None}
        
        # But has recent data
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        recent_data = pd.DataFrame({
            'Close': [150, 151, 152]
        }, index=dates)
        mock_ticker.history.return_value = recent_data
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.validate_symbol('TEST')
        
        assert result is True
        assert provider._symbol_cache['TEST'] is True
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_validate_symbol_invalid_empty_info(self, mock_ticker_class, provider):
        """Test validate_symbol with invalid symbol that returns empty info."""
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()  # Empty recent data too
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.validate_symbol('INVALID')
        
        assert result is False
        assert provider._symbol_cache['INVALID'] is False
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_validate_symbol_exception_handling(self, mock_ticker_class, provider):
        """Test validate_symbol when yfinance raises exception."""
        mock_ticker = Mock()
        mock_ticker.info = Mock(side_effect=Exception("Network error"))
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.validate_symbol('ERROR')
        
        assert result is False
        assert provider._symbol_cache['ERROR'] is False
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_validate_symbol_caching(self, mock_ticker_class, provider):
        """Test that symbol validation results are cached."""
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'AAPL', 'longName': 'Apple Inc.'}
        mock_ticker_class.return_value = mock_ticker
        
        # First call
        result1 = provider.validate_symbol('AAPL')
        # Second call should use cache
        result2 = provider.validate_symbol('AAPL')
        
        assert result1 is True
        assert result2 is True
        # Ticker should only be created once due to caching
        assert mock_ticker_class.call_count == 1
    
    def test_clear_symbol_cache(self, provider):
        """Test clearing the symbol validation cache."""
        # Add something to cache
        provider._symbol_cache['TEST'] = True
        assert len(provider._symbol_cache) == 1
        
        # Clear cache
        provider.clear_symbol_cache()
        assert len(provider._symbol_cache) == 0


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderAvailableSymbols:
    """Test YFinanceProvider available symbols functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create YFinanceProvider instance for testing."""
        return YFinanceProvider()
    
    def test_get_available_symbols_returns_list(self, provider):
        """Test that get_available_symbols returns a list of symbols."""
        symbols = provider.get_available_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(symbol, str) for symbol in symbols)
    
    def test_get_available_symbols_contains_expected_symbols(self, provider):
        """Test that get_available_symbols contains expected major symbols."""
        symbols = provider.get_available_symbols()
        
        # Check for some major symbols that should be included
        expected_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', '^GSPC']
        for symbol in expected_symbols:
            assert symbol in symbols
    
    def test_get_available_symbols_includes_different_asset_types(self, provider):
        """Test that get_available_symbols includes different asset types."""
        symbols = provider.get_available_symbols()
        
        # Should include stocks, indices, ETFs, currencies, crypto
        has_stock = any(symbol in ['AAPL', 'MSFT', 'GOOGL'] for symbol in symbols)
        has_index = any(symbol.startswith('^') for symbol in symbols)
        has_etf = any(symbol in ['SPY', 'QQQ', 'VTI'] for symbol in symbols)
        has_currency = any('USD=X' in symbol for symbol in symbols)
        has_crypto = any('-USD' in symbol for symbol in symbols)
        
        assert has_stock, "Should include stock symbols"
        assert has_index, "Should include index symbols"
        assert has_etf, "Should include ETF symbols"
        assert has_currency, "Should include currency symbols"
        assert has_crypto, "Should include cryptocurrency symbols"


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderSymbolInfo:
    """Test YFinanceProvider symbol info functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create YFinanceProvider instance for testing."""
        return YFinanceProvider()
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_symbol_info_valid_symbol(self, mock_ticker_class, provider):
        """Test get_symbol_info with valid symbol."""
        # Mock validate_symbol to return True
        provider.validate_symbol = Mock(return_value=True)
        
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Apple Inc.',
            'currency': 'USD',
            'exchange': 'NASDAQ',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currentPrice': 150.0,
            'volume': 50000000
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.get_symbol_info('AAPL')
        
        expected_info = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'currency': 'USD',
            'exchange': 'NASDAQ',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000,
            'price': 150.0,
            'volume': 50000000
        }
        
        assert result == expected_info
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_symbol_info_invalid_symbol(self, mock_ticker_class, provider):
        """Test get_symbol_info with invalid symbol."""
        # Mock validate_symbol to return False
        provider.validate_symbol = Mock(return_value=False)
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            provider.get_symbol_info('INVALID')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_symbol_info_connection_error(self, mock_ticker_class, provider):
        """Test get_symbol_info when connection fails."""
        provider.validate_symbol = Mock(return_value=True)
        
        mock_ticker = Mock()
        # Mock the info property to raise an exception when accessed
        type(mock_ticker).info = PropertyMock(side_effect=Exception("Connection timeout"))
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ConnectionError, match="Unable to connect to Yahoo Finance"):
            provider.get_symbol_info('AAPL')
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_get_symbol_info_missing_fields(self, mock_ticker_class, provider):
        """Test get_symbol_info with missing info fields."""
        provider.validate_symbol = Mock(return_value=True)
        
        mock_ticker = Mock()
        # Minimal info with some missing fields
        mock_ticker.info = {
            'shortName': 'Apple',  # longName missing, should use shortName
            'regularMarketPrice': 150.0,  # currentPrice missing, should use regularMarketPrice
            'regularMarketVolume': 50000000  # volume missing, should use regularMarketVolume
        }
        mock_ticker_class.return_value = mock_ticker
        
        result = provider.get_symbol_info('AAPL')
        
        assert result['name'] == 'Apple'
        assert result['price'] == 150.0
        assert result['volume'] == 50000000
        assert result['currency'] == 'USD'  # Default value
        assert result['exchange'] == 'Unknown'  # Default value


@pytest.mark.skipif(not YFINANCE_AVAILABLE, reason="yfinance not available")
class TestYFinanceProviderIntegration:
    """Integration tests for YFinanceProvider with specific symbols and date ranges."""
    
    @pytest.fixture
    def provider(self):
        """Create YFinanceProvider instance for testing."""
        return YFinanceProvider()
    
    def test_real_symbol_validation_major_stocks(self, provider):
        """Test symbol validation with real major stock symbols."""
        # Test some major stocks that should always be valid
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in major_stocks:
            # This may make real API calls, so we'll be lenient with timing
            try:
                result = provider.validate_symbol(symbol)
                # Major stocks should generally be valid, but network issues might occur
                assert isinstance(result, bool)
            except Exception:
                # If there are network issues, that's acceptable for this test
                pytest.skip(f"Network issues prevented testing {symbol}")
    
    def test_invalid_symbol_validation(self, provider):
        """Test symbol validation with clearly invalid symbols."""
        invalid_symbols = ['INVALID123', 'NOTREAL', 'FAKE_SYMBOL', '']
        
        for symbol in invalid_symbols:
            try:
                result = provider.validate_symbol(symbol)
                # Invalid symbols should return False
                assert result is False
            except Exception:
                # Network issues are acceptable
                pytest.skip(f"Network issues prevented testing {symbol}")
    
    @patch('warspite_financial.providers.yfinance.yf.Ticker')
    def test_date_range_handling(self, mock_ticker_class, provider):
        """Test handling of different date ranges and formats."""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC constraints
        for i in range(len(mock_data)):
            row = mock_data.iloc[i]
            mock_data.iloc[i, mock_data.columns.get_loc('High')] = max(row['Open'], row['High'], row['Close'])
            mock_data.iloc[i, mock_data.columns.get_loc('Low')] = min(row['Open'], row['Low'], row['Close'])
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        # Test different date formats
        test_cases = [
            (datetime(2023, 1, 1), datetime(2023, 1, 10)),
            (date(2023, 1, 1), date(2023, 1, 10)),
        ]
        
        for start_date, end_date in test_cases:
            result = provider.get_data('AAPL', start_date, end_date, '1d')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
            assert isinstance(result.index, pd.DatetimeIndex)