"""
Unit tests for OANDAProvider.

Tests data retrieval functionality and trading interface methods with mock responses.
Requirements: 2.2, 2.5, 6.3
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date
import pandas as pd
import numpy as np
import json
import requests

from warspite_financial.providers.oanda import OANDAProvider
from warspite_financial.providers.base import OrderResult, AccountInfo, Position, Order


class TestOANDAProviderInitialization:
    """Test OANDAProvider initialization and setup."""
    
    def test_init_with_valid_credentials(self):
        """Test successful initialization with valid credentials."""
        provider = OANDAProvider(
            api_token="test_token",
            account_id="test_account",
            practice=True
        )
        
        assert provider.api_token == "test_token"
        assert provider.account_id == "test_account"
        assert provider.practice is True
        assert provider.base_url == OANDAProvider.PRACTICE_API_URL
        assert 'Authorization' in provider.session.headers
        assert provider.session.headers['Authorization'] == 'Bearer test_token'
    
    def test_init_with_live_environment(self):
        """Test initialization with live environment."""
        provider = OANDAProvider(
            api_token="live_token",
            account_id="live_account",
            practice=False
        )
        
        assert provider.practice is False
        assert provider.base_url == OANDAProvider.LIVE_API_URL
    
    def test_init_missing_api_token(self):
        """Test initialization fails with missing API token."""
        with pytest.raises(ValueError, match="API token is required"):
            OANDAProvider(api_token="", account_id="test_account")
    
    def test_init_missing_account_id(self):
        """Test initialization fails with missing account ID."""
        with pytest.raises(ValueError, match="Account ID is required"):
            OANDAProvider(api_token="test_token", account_id="")
    
    def test_session_configuration(self):
        """Test that session is properly configured with retry strategy."""
        provider = OANDAProvider("test_token", "test_account")
        
        # Check headers
        expected_headers = {
            'Authorization': 'Bearer test_token',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }
        
        for key, value in expected_headers.items():
            assert provider.session.headers[key] == value
    
    def test_interval_mapping(self):
        """Test OANDA interval mapping is correctly configured."""
        provider = OANDAProvider("test_token", "test_account")
        
        expected_mapping = {
            '1m': 'M1',
            '5m': 'M5',
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D',
            '1w': 'W',
            '1mo': 'M'
        }
        
        assert provider.OANDA_INTERVALS == expected_mapping


class TestOANDAProviderRequestHandling:
    """Test OANDAProvider HTTP request handling."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    @patch('requests.Session.get')
    def test_make_request_get_success(self, mock_get, provider):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response
        
        result = provider._make_request('GET', '/test/endpoint', params={'param': 'value'})
        
        assert result == {"test": "data"}
        mock_get.assert_called_once_with(
            f"{provider.base_url}/test/endpoint",
            params={'param': 'value'},
            timeout=30
        )
    
    @patch('requests.Session.post')
    def test_make_request_post_success(self, mock_post, provider):
        """Test successful POST request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"created": "success"}
        mock_post.return_value = mock_response
        
        data = {"order": {"type": "MARKET"}}
        result = provider._make_request('POST', '/orders', data=data)
        
        assert result == {"created": "success"}
        mock_post.assert_called_once_with(
            f"{provider.base_url}/orders",
            params=None,
            json=data,
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_make_request_401_unauthorized(self, mock_get, provider):
        """Test 401 unauthorized error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        with pytest.raises(PermissionError, match="Invalid API token or insufficient permissions"):
            provider._make_request('GET', '/test')
    
    @patch('requests.Session.get')
    def test_make_request_404_not_found(self, mock_get, provider):
        """Test 404 not found error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Resource not found"):
            provider._make_request('GET', '/test')
    
    @patch('requests.Session.get')
    def test_make_request_429_rate_limit(self, mock_get, provider):
        """Test 429 rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        with pytest.raises(ConnectionError, match="Rate limit exceeded"):
            provider._make_request('GET', '/test')
    
    @patch('requests.Session.get')
    def test_make_request_500_server_error(self, mock_get, provider):
        """Test 500 server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"errorMessage": "Internal server error"}
        mock_response.text = "Server Error"
        mock_get.return_value = mock_response
        
        with pytest.raises(ConnectionError, match="API request failed with status 500: Internal server error"):
            provider._make_request('GET', '/test')
    
    @patch('requests.Session.get')
    def test_make_request_timeout(self, mock_get, provider):
        """Test request timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(ConnectionError, match="Request timeout"):
            provider._make_request('GET', '/test')
    
    @patch('requests.Session.get')
    def test_make_request_connection_error(self, mock_get, provider):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(ConnectionError, match="Connection error"):
            provider._make_request('GET', '/test')
    
    def test_make_request_unsupported_method(self, provider):
        """Test unsupported HTTP method."""
        with pytest.raises(ValueError, match="Unsupported HTTP method: PATCH"):
            provider._make_request('PATCH', '/test')


class TestOANDAProviderDataRetrieval:
    """Test OANDAProvider data retrieval functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    @pytest.fixture
    def mock_candle_data(self):
        """Create mock candle data for testing."""
        return {
            "candles": [
                {
                    "complete": True,
                    "time": "2023-01-01T00:00:00.000000000Z",
                    "volume": 1000,
                    "mid": {
                        "o": "1.1000",
                        "h": "1.1050",
                        "l": "1.0950",
                        "c": "1.1025"
                    }
                },
                {
                    "complete": True,
                    "time": "2023-01-02T00:00:00.000000000Z",
                    "volume": 1100,
                    "mid": {
                        "o": "1.1025",
                        "h": "1.1075",
                        "l": "1.0975",
                        "c": "1.1050"
                    }
                }
            ]
        }
    
    def test_get_data_valid_symbol_daily(self, provider, mock_candle_data):
        """Test get_data with valid symbol and daily interval."""
        with patch.object(provider, '_make_request', return_value=mock_candle_data):
            with patch.object(provider, 'validate_symbol', return_value=True):
                result = provider.get_data(
                    'EUR_USD',
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 3),
                    '1d'
                )
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(result) == 2
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # Check data values
        assert result.iloc[0]['Open'] == 1.1000
        assert result.iloc[0]['High'] == 1.1050
        assert result.iloc[0]['Low'] == 1.0950
        assert result.iloc[0]['Close'] == 1.1025
        assert result.iloc[0]['Volume'] == 1000
    
    def test_get_data_different_intervals(self, provider, mock_candle_data):
        """Test get_data with different time intervals."""
        test_intervals = ['1m', '5m', '1h', '1d', '1w']
        expected_oanda_intervals = ['M1', 'M5', 'H1', 'D', 'W']
        
        for interval, expected_oanda_interval in zip(test_intervals, expected_oanda_intervals):
            with patch.object(provider, '_make_request', return_value=mock_candle_data) as mock_request:
                with patch.object(provider, 'validate_symbol', return_value=True):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), interval)
                    
                    # Check that correct OANDA interval was used
                    call_args = mock_request.call_args
                    assert call_args[1]['params']['granularity'] == expected_oanda_interval
    
    def test_get_data_invalid_symbol(self, provider):
        """Test get_data with invalid symbol."""
        with patch.object(provider, 'validate_symbol', return_value=False):
            with pytest.raises(ValueError, match="Invalid OANDA instrument: INVALID"):
                provider.get_data('INVALID', datetime(2023, 1, 1), datetime(2023, 1, 3), '1d')
    
    def test_get_data_unsupported_interval(self, provider):
        """Test get_data with unsupported interval."""
        with pytest.raises(ValueError, match="Unsupported interval: invalid"):
            provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), 'invalid')
    
    def test_get_data_no_candles_in_response(self, provider):
        """Test get_data when API returns no candles."""
        with patch.object(provider, '_make_request', return_value={}):
            with patch.object(provider, 'validate_symbol', return_value=True):
                with pytest.raises(ValueError, match="No candle data returned for EUR_USD"):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), '1d')
    
    def test_get_data_empty_candles_list(self, provider):
        """Test get_data when API returns empty candles list."""
        with patch.object(provider, '_make_request', return_value={"candles": []}):
            with patch.object(provider, 'validate_symbol', return_value=True):
                with pytest.raises(ValueError, match="No data available for EUR_USD"):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), '1d')
    
    def test_get_data_incomplete_candles_only(self, provider):
        """Test get_data when API returns only incomplete candles."""
        incomplete_data = {
            "candles": [
                {
                    "complete": False,
                    "time": "2023-01-01T00:00:00.000000000Z",
                    "volume": 1000,
                    "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"}
                }
            ]
        }
        
        with patch.object(provider, '_make_request', return_value=incomplete_data):
            with patch.object(provider, 'validate_symbol', return_value=True):
                with pytest.raises(ValueError, match="No complete candle data available"):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), '1d')
    
    def test_get_data_api_error(self, provider):
        """Test get_data when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with patch.object(provider, 'validate_symbol', return_value=True):
                with pytest.raises(ConnectionError, match="API Error"):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 3), '1d')
    
    def test_get_data_date_formatting(self, provider, mock_candle_data):
        """Test that dates are properly formatted for OANDA API."""
        with patch.object(provider, '_make_request', return_value=mock_candle_data) as mock_request:
            with patch.object(provider, 'validate_symbol', return_value=True):
                provider.get_data(
                    'EUR_USD',
                    datetime(2023, 1, 1, 10, 30, 45),
                    datetime(2023, 1, 3, 15, 45, 30),
                    '1d'
                )
                
                call_args = mock_request.call_args
                params = call_args[1]['params']
                
                # Check RFC3339 format
                assert params['from'] == '2023-01-01T10:30:45.000000000Z'
                assert params['to'] == '2023-01-03T15:45:30.000000000Z'


class TestOANDAProviderSymbolValidation:
    """Test OANDAProvider symbol validation functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    @pytest.fixture
    def mock_instruments_response(self):
        """Create mock instruments response."""
        return {
            "instruments": [
                {"name": "EUR_USD", "type": "CURRENCY"},
                {"name": "GBP_JPY", "type": "CURRENCY"},
                {"name": "US30_USD", "type": "CFD"}
            ]
        }
    
    def test_get_available_symbols_success(self, provider, mock_instruments_response):
        """Test successful retrieval of available symbols."""
        with patch.object(provider, '_make_request', return_value=mock_instruments_response):
            symbols = provider.get_available_symbols()
            
            assert symbols == ["EUR_USD", "GBP_JPY", "US30_USD"]
            assert provider._instruments_cache == symbols
            assert provider._cache_timestamp is not None
    
    def test_get_available_symbols_caching(self, provider, mock_instruments_response):
        """Test that symbols are cached to avoid repeated API calls."""
        with patch.object(provider, '_make_request', return_value=mock_instruments_response) as mock_request:
            # First call
            symbols1 = provider.get_available_symbols()
            # Second call should use cache
            symbols2 = provider.get_available_symbols()
            
            assert symbols1 == symbols2
            # API should only be called once due to caching
            assert mock_request.call_count == 1
    
    def test_get_available_symbols_cache_expiry(self, provider, mock_instruments_response):
        """Test that cache expires after the configured duration."""
        with patch.object(provider, '_make_request', return_value=mock_instruments_response) as mock_request:
            with patch('time.time', side_effect=[1000, 1000 + provider._cache_duration + 1]):
                # First call
                provider.get_available_symbols()
                # Second call after cache expiry
                provider.get_available_symbols()
                
                # API should be called twice due to cache expiry
                assert mock_request.call_count == 2
    
    def test_get_available_symbols_no_instruments(self, provider):
        """Test when API returns no instruments data."""
        with patch.object(provider, '_make_request', return_value={}):
            with pytest.raises(ValueError, match="No instruments data returned"):
                provider.get_available_symbols()
    
    def test_get_available_symbols_api_error(self, provider):
        """Test when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="API Error"):
                provider.get_available_symbols()
    
    def test_validate_symbol_valid(self, provider):
        """Test validate_symbol with valid symbol."""
        with patch.object(provider, 'get_available_symbols', return_value=["EUR_USD", "GBP_JPY"]):
            assert provider.validate_symbol("EUR_USD") is True
            assert provider.validate_symbol("GBP_JPY") is True
    
    def test_validate_symbol_invalid(self, provider):
        """Test validate_symbol with invalid symbol."""
        with patch.object(provider, 'get_available_symbols', return_value=["EUR_USD", "GBP_JPY"]):
            assert provider.validate_symbol("INVALID") is False
    
    def test_validate_symbol_fallback_format_check(self, provider):
        """Test validate_symbol fallback when API fails."""
        with patch.object(provider, 'get_available_symbols', side_effect=Exception("API Error")):
            # Should fall back to format check
            assert provider.validate_symbol("EUR_USD") is True  # Valid format
            assert provider.validate_symbol("INVALID") is False  # Invalid format


class TestOANDAProviderTradingInterface:
    """Test OANDAProvider trading interface methods."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    def test_place_order_success(self, provider):
        """Test successful order placement."""
        mock_response = {
            "orderCreateTransaction": {
                "id": "12345",
                "type": "MARKET_ORDER",
                "instrument": "EUR_USD",
                "units": "1000"
            }
        }
        
        with patch.object(provider, 'validate_symbol', return_value=True):
            with patch.object(provider, '_make_request', return_value=mock_response):
                result = provider.place_order("EUR_USD", 1000, "market")
                
                assert isinstance(result, OrderResult)
                assert result.success is True
                assert result.order_id == "12345"
                assert "Order placed successfully" in result.message
    
    def test_place_order_invalid_symbol(self, provider):
        """Test order placement with invalid symbol."""
        with patch.object(provider, 'validate_symbol', return_value=False):
            result = provider.place_order("INVALID", 1000, "market")
            
            assert isinstance(result, OrderResult)
            assert result.success is False
            assert "Invalid instrument" in result.message
    
    def test_place_order_zero_quantity(self, provider):
        """Test order placement with zero quantity."""
        result = provider.place_order("EUR_USD", 0, "market")
        
        assert isinstance(result, OrderResult)
        assert result.success is False
        assert "Order quantity cannot be zero" in result.message
    
    def test_place_order_api_error(self, provider):
        """Test order placement when API request fails."""
        with patch.object(provider, 'validate_symbol', return_value=True):
            with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
                result = provider.place_order("EUR_USD", 1000, "market")
                
                assert isinstance(result, OrderResult)
                assert result.success is False
                assert "Failed to place order" in result.message
    
    def test_place_order_no_transaction_id(self, provider):
        """Test order placement when API doesn't return transaction ID."""
        mock_response = {"someOtherField": "value"}
        
        with patch.object(provider, 'validate_symbol', return_value=True):
            with patch.object(provider, '_make_request', return_value=mock_response):
                result = provider.place_order("EUR_USD", 1000, "market")
                
                assert isinstance(result, OrderResult)
                assert result.success is False
                assert "Order creation failed" in result.message
    
    def test_get_account_info_success(self, provider):
        """Test successful account info retrieval."""
        mock_response = {
            "account": {
                "id": "test_account",
                "balance": "10000.0000",
                "currency": "USD"
            }
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.get_account_info()
            
            assert isinstance(result, AccountInfo)
            assert result.account_id == "test_account"
            assert result.balance == 10000.0
            assert result.currency == "USD"
    
    def test_get_account_info_no_account_data(self, provider):
        """Test account info when API returns no account data."""
        with patch.object(provider, '_make_request', return_value={}):
            with pytest.raises(ValueError, match="No account data returned"):
                provider.get_account_info()
    
    def test_get_account_info_api_error(self, provider):
        """Test account info when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="API Error"):
                provider.get_account_info()
    
    def test_get_positions_success(self, provider):
        """Test successful positions retrieval."""
        mock_positions_response = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "unrealizedPL": "50.0"},
                    "short": {"units": "0", "unrealizedPL": "0.0"}
                },
                {
                    "instrument": "GBP_JPY",
                    "long": {"units": "0", "unrealizedPL": "0.0"},
                    "short": {"units": "-500", "unrealizedPL": "-25.0"}
                }
            ]
        }
        
        mock_pricing_response = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.1000"}],
                    "asks": [{"price": "1.1002"}]
                }
            ]
        }
        
        with patch.object(provider, '_make_request') as mock_request:
            # First call returns positions, second call returns pricing
            mock_request.side_effect = [mock_positions_response, mock_pricing_response, mock_pricing_response]
            
            positions = provider.get_positions()
            
            assert len(positions) == 2
            
            # Check first position (long EUR_USD)
            pos1 = positions[0]
            assert isinstance(pos1, Position)
            assert pos1.symbol == "EUR_USD"
            assert pos1.quantity == 1000.0
            assert pos1.unrealized_pnl == 50.0
            assert pos1.position_id == "EUR_USD_test_account"
            
            # Check second position (short GBP_JPY)
            pos2 = positions[1]
            assert pos2.symbol == "GBP_JPY"
            assert pos2.quantity == -500.0
            assert pos2.unrealized_pnl == -25.0
    
    def test_get_positions_no_positions(self, provider):
        """Test positions retrieval when no positions exist."""
        mock_response = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "0", "unrealizedPL": "0.0"},
                    "short": {"units": "0", "unrealizedPL": "0.0"}
                }
            ]
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            positions = provider.get_positions()
            
            assert len(positions) == 0
    
    def test_get_positions_no_positions_data(self, provider):
        """Test positions retrieval when API returns no positions data."""
        with patch.object(provider, '_make_request', return_value={}):
            with pytest.raises(ValueError, match="No positions data returned"):
                provider.get_positions()
    
    def test_get_positions_pricing_error(self, provider):
        """Test positions retrieval when pricing request fails."""
        mock_positions_response = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "unrealizedPL": "50.0"},
                    "short": {"units": "0", "unrealizedPL": "0.0"}
                }
            ]
        }
        
        with patch.object(provider, '_make_request') as mock_request:
            # First call succeeds, second call (pricing) fails
            mock_request.side_effect = [mock_positions_response, Exception("Pricing error")]
            
            positions = provider.get_positions()
            
            # Should still return position but with 0 current price
            assert len(positions) == 1
            assert positions[0].current_price == 0.0
    
    def test_close_position_success(self, provider):
        """Test successful position closure."""
        mock_response = {
            "longOrderCreateTransaction": {"id": "123"},
            "shortOrderCreateTransaction": {"id": "124"}
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.close_position("EUR_USD_test_account")
            
            assert result is True
    
    def test_close_position_invalid_id_format(self, provider):
        """Test position closure with invalid position ID format."""
        result = provider.close_position("INVALID_FORMAT")
        
        assert result is False
    
    def test_close_position_api_error(self, provider):
        """Test position closure when API request fails."""
        with patch.object(provider, '_make_request', side_effect=Exception("API Error")):
            result = provider.close_position("EUR_USD_test_account")
            
            assert result is False
    
    def test_close_position_no_transactions(self, provider):
        """Test position closure when no transactions are created."""
        mock_response = {}  # No transaction IDs
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.close_position("EUR_USD_test_account")
            
            assert result is False
    
    def test_close_all_positions_success(self, provider):
        """Test successful closure of all positions."""
        mock_positions = [
            Position("EUR_USD_test_account", "EUR_USD", 1000, 1.1000, 50.0),
            Position("GBP_JPY_test_account", "GBP_JPY", -500, 150.0, -25.0)
        ]
        
        with patch.object(provider, 'get_positions', return_value=mock_positions):
            with patch.object(provider, 'close_position', return_value=True):
                result = provider.close_all_positions()
                
                assert result is True
    
    def test_close_all_positions_no_positions(self, provider):
        """Test closure of all positions when no positions exist."""
        with patch.object(provider, 'get_positions', return_value=[]):
            result = provider.close_all_positions()
            
            assert result is True  # No positions to close is considered success
    
    def test_close_all_positions_partial_failure(self, provider):
        """Test closure of all positions when some closures fail."""
        mock_positions = [
            Position("EUR_USD_test_account", "EUR_USD", 1000, 1.1000, 50.0),
            Position("GBP_JPY_test_account", "GBP_JPY", -500, 150.0, -25.0)
        ]
        
        with patch.object(provider, 'get_positions', return_value=mock_positions):
            with patch.object(provider, 'close_position', side_effect=[True, False]):
                result = provider.close_all_positions()
                
                assert result is False  # Not all positions closed successfully
    
    def test_close_all_positions_get_positions_error(self, provider):
        """Test closure of all positions when getting positions fails."""
        with patch.object(provider, 'get_positions', side_effect=Exception("API Error")):
            result = provider.close_all_positions()
            
            assert result is False


class TestOANDAProviderIntegration:
    """Integration tests for OANDAProvider with realistic scenarios."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    def test_complete_trading_workflow(self, provider):
        """Test a complete trading workflow from data retrieval to position management."""
        # Mock data retrieval
        mock_candle_data = {
            "candles": [
                {
                    "complete": True,
                    "time": "2023-01-01T00:00:00.000000000Z",
                    "volume": 1000,
                    "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0950", "c": "1.1025"}
                }
            ]
        }
        
        # Mock account info
        mock_account = {
            "account": {"id": "test_account", "balance": "10000.0000", "currency": "USD"}
        }
        
        # Mock order response
        mock_order = {
            "orderCreateTransaction": {"id": "12345", "type": "MARKET_ORDER"}
        }
        
        # Mock positions
        mock_positions = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "unrealizedPL": "50.0"},
                    "short": {"units": "0", "unrealizedPL": "0.0"}
                }
            ]
        }
        
        # Mock pricing response for get_positions
        mock_pricing = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.1000"}],
                    "asks": [{"price": "1.1002"}]
                }
            ]
        }
        
        with patch.object(provider, 'validate_symbol', return_value=True):
            with patch.object(provider, '_make_request') as mock_request:
                # Set up the sequence of API calls
                mock_request.side_effect = [
                    mock_candle_data,  # get_data
                    mock_account,      # get_account_info
                    mock_order,        # place_order
                    mock_positions,    # get_positions
                    mock_pricing,      # get_positions pricing call
                    {"longOrderCreateTransaction": {"id": "123"}}  # close_position
                ]
                
                # 1. Get market data
                data = provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 2), '1d')
                assert len(data) == 1
                
                # 2. Check account info
                account = provider.get_account_info()
                assert account.balance == 10000.0
                
                # 3. Place order
                order_result = provider.place_order('EUR_USD', 1000, 'market')
                assert order_result.success is True
                
                # 4. Check positions
                positions = provider.get_positions()
                assert len(positions) == 1
                assert positions[0].quantity == 1000.0
                
                # 5. Close position
                close_result = provider.close_position('EUR_USD_test_account')
                assert close_result is True
    
    def test_error_handling_across_methods(self, provider):
        """Test consistent error handling across different methods."""
        # Test that all methods handle connection errors consistently
        with patch.object(provider, '_make_request', side_effect=ConnectionError("Network error")):
            with patch.object(provider, 'validate_symbol', return_value=True):
                # Data retrieval should raise ConnectionError
                with pytest.raises(ConnectionError):
                    provider.get_data('EUR_USD', datetime(2023, 1, 1), datetime(2023, 1, 2), '1d')
                
                # Account info should raise ConnectionError
                with pytest.raises(ConnectionError):
                    provider.get_account_info()
                
                # Get positions should raise ConnectionError
                with pytest.raises(ConnectionError):
                    provider.get_positions()
                
                # Place order should return failed OrderResult
                order_result = provider.place_order('EUR_USD', 1000, 'market')
                assert order_result.success is False
                
                # Close position should return False
                close_result = provider.close_position('EUR_USD_test_account')
                assert close_result is False
    
    def test_date_and_interval_validation_integration(self, provider):
        """Test that date and interval validation works with base class methods."""
        # Test invalid date range
        with pytest.raises(ValueError, match="Start date must be before end date"):
            provider.validate_date_range(datetime(2023, 1, 2), datetime(2023, 1, 1))
        
        # Test invalid interval
        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.validate_interval('invalid')
        
        # Test future date (use different end date to avoid same date issue)
        future_date = datetime.now().replace(year=datetime.now().year + 1)
        future_end_date = future_date.replace(day=future_date.day + 1)
        with pytest.raises(ValueError, match="Start date cannot be in the future"):
            provider.validate_date_range(future_date, future_end_date)


class TestOANDAProviderOrderManagement:
    """Test OANDAProvider order management functionality."""
    
    @pytest.fixture
    def provider(self):
        """Create OANDAProvider instance for testing."""
        return OANDAProvider("test_token", "test_account", practice=True)
    
    @pytest.fixture
    def mock_orders_response(self):
        """Create mock orders response."""
        return {
            "orders": [
                {
                    "id": "12345",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "type": "LIMIT",
                    "state": "PENDING",
                    "createTime": "2023-01-01T10:00:00.000000000Z",
                    "price": "1.1000",
                    "timeInForce": "GTC",
                    "clientExtensions": {
                        "id": "my_order_1",
                        "tag": "strategy_1",
                        "comment": "Test order"
                    }
                },
                {
                    "id": "12346",
                    "instrument": "GBP_JPY",
                    "units": "-500",
                    "type": "MARKET_IF_TOUCHED",
                    "state": "PENDING",
                    "createTime": "2023-01-01T11:00:00.000000000Z",
                    "price": "150.0",
                    "timeInForce": "GTC"
                }
            ]
        }
    
    def test_get_orders_success(self, provider, mock_orders_response):
        """Test successful retrieval of orders."""
        with patch.object(provider, '_make_request', return_value=mock_orders_response):
            orders = provider.get_orders()
            
            assert len(orders) == 2
            
            # Check first order
            order1 = orders[0]
            assert isinstance(order1, Order)
            assert order1.order_id == "12345"
            assert order1.instrument == "EUR_USD"
            assert order1.units == 1000.0
            assert order1.order_type == "LIMIT"
            assert order1.state == "PENDING"
            assert order1.price == 1.1000
            assert order1.time_in_force == "GTC"
            assert order1.client_extensions["id"] == "my_order_1"
            
            # Check second order
            order2 = orders[1]
            assert order2.order_id == "12346"
            assert order2.instrument == "GBP_JPY"
            assert order2.units == -500.0
            assert order2.order_type == "MARKET_IF_TOUCHED"
            assert order2.state == "PENDING"
            assert order2.price == 150.0
    
    def test_get_orders_with_filters(self, provider, mock_orders_response):
        """Test get_orders with various filters."""
        with patch.object(provider, '_make_request', return_value=mock_orders_response) as mock_request:
            # Test with instrument filter
            provider.get_orders(instrument="EUR_USD")
            call_args = mock_request.call_args
            assert call_args[1]['params']['instrument'] == "EUR_USD"
            
            # Test with state filter
            provider.get_orders(state="FILLED")
            call_args = mock_request.call_args
            assert call_args[1]['params']['state'] == "FILLED"
            
            # Test with count
            provider.get_orders(count=100)
            call_args = mock_request.call_args
            assert call_args[1]['params']['count'] == 100
            
            # Test with order IDs
            provider.get_orders(order_ids=["12345", "12346"])
            call_args = mock_request.call_args
            assert call_args[1]['params']['ids'] == "12345,12346"
    
    def test_get_orders_invalid_count(self, provider):
        """Test get_orders with invalid count parameter."""
        with pytest.raises(ValueError, match="Maximum count is 500"):
            provider.get_orders(count=501)
    
    def test_get_orders_no_orders_data(self, provider):
        """Test get_orders when API returns no orders data."""
        with patch.object(provider, '_make_request', return_value={}):
            with pytest.raises(ValueError, match="No orders data returned"):
                provider.get_orders()
    
    def test_get_orders_api_error(self, provider):
        """Test get_orders when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="API Error"):
                provider.get_orders()
    
    def test_get_pending_orders_success(self, provider, mock_orders_response):
        """Test successful retrieval of pending orders."""
        with patch.object(provider, '_make_request', return_value=mock_orders_response):
            orders = provider.get_pending_orders()
            
            assert len(orders) == 2
            assert all(order.state == "PENDING" for order in orders)
    
    def test_get_pending_orders_api_error(self, provider):
        """Test get_pending_orders when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="Failed to retrieve pending orders"):
                provider.get_pending_orders()
    
    def test_get_order_success(self, provider):
        """Test successful retrieval of specific order."""
        mock_response = {
            "order": {
                "id": "12345",
                "instrument": "EUR_USD",
                "units": "1000",
                "type": "LIMIT",
                "state": "PENDING",
                "createTime": "2023-01-01T10:00:00.000000000Z",
                "price": "1.1000",
                "timeInForce": "GTC"
            }
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            order = provider.get_order("12345")
            
            assert isinstance(order, Order)
            assert order.order_id == "12345"
            assert order.instrument == "EUR_USD"
            assert order.units == 1000.0
            assert order.price == 1.1000
    
    def test_get_order_not_found(self, provider):
        """Test get_order when order is not found."""
        with patch.object(provider, '_make_request', side_effect=ValueError("Resource not found")):
            order = provider.get_order("nonexistent")
            
            assert order is None
    
    def test_get_order_no_order_data(self, provider):
        """Test get_order when API returns no order data."""
        with patch.object(provider, '_make_request', return_value={}):
            order = provider.get_order("12345")
            
            assert order is None
    
    def test_get_order_empty_id(self, provider):
        """Test get_order with empty order ID."""
        with pytest.raises(ValueError, match="Order ID is required"):
            provider.get_order("")
    
    def test_get_order_api_error(self, provider):
        """Test get_order when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="Failed to retrieve order"):
                provider.get_order("12345")
    
    def test_cancel_order_success(self, provider):
        """Test successful order cancellation."""
        mock_response = {
            "orderCancelTransaction": {
                "id": "12346",
                "orderID": "12345",
                "reason": "CLIENT_REQUEST",
                "type": "ORDER_CANCEL"
            }
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.cancel_order("12345")
            
            assert result is True
    
    def test_cancel_order_no_transaction(self, provider):
        """Test order cancellation when no cancel transaction is returned."""
        mock_response = {}
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.cancel_order("12345")
            
            assert result is False
    
    def test_cancel_order_empty_id(self, provider):
        """Test cancel_order with empty order ID."""
        with pytest.raises(ValueError, match="Order ID is required"):
            provider.cancel_order("")
    
    def test_cancel_order_api_error(self, provider):
        """Test cancel_order when API request fails."""
        with patch.object(provider, '_make_request', side_effect=ConnectionError("API Error")):
            with pytest.raises(ConnectionError, match="Failed to cancel order"):
                provider.cancel_order("12345")
    
    def test_cancel_order_with_client_id(self, provider):
        """Test order cancellation using client order ID."""
        mock_response = {
            "orderCancelTransaction": {
                "id": "12346",
                "orderID": "12345",
                "clientOrderID": "my_order_1",
                "reason": "CLIENT_REQUEST",
                "type": "ORDER_CANCEL"
            }
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            result = provider.cancel_order("@my_order_1")
            
            assert result is True
    
    def test_parse_order_data_complete(self, provider):
        """Test parsing complete order data."""
        order_data = {
            "id": "12345",
            "instrument": "EUR_USD",
            "units": "1000",
            "type": "LIMIT",
            "state": "PENDING",
            "createTime": "2023-01-01T10:00:00.000000000Z",
            "price": "1.1000",
            "timeInForce": "GTC",
            "clientExtensions": {
                "id": "my_order_1",
                "tag": "strategy_1",
                "comment": "Test order"
            }
        }
        
        order = provider._parse_order_data(order_data)
        
        assert isinstance(order, Order)
        assert order.order_id == "12345"
        assert order.instrument == "EUR_USD"
        assert order.units == 1000.0
        assert order.order_type == "LIMIT"
        assert order.state == "PENDING"
        assert order.create_time == "2023-01-01T10:00:00.000000000Z"
        assert order.price == 1.1000
        assert order.time_in_force == "GTC"
        assert order.client_extensions["id"] == "my_order_1"
    
    def test_parse_order_data_minimal(self, provider):
        """Test parsing minimal order data."""
        order_data = {
            "id": "12345",
            "instrument": "EUR_USD",
            "units": "1000",
            "type": "MARKET",
            "state": "FILLED",
            "createTime": "2023-01-01T10:00:00.000000000Z"
        }
        
        order = provider._parse_order_data(order_data)
        
        assert isinstance(order, Order)
        assert order.order_id == "12345"
        assert order.instrument == "EUR_USD"
        assert order.units == 1000.0
        assert order.order_type == "MARKET"
        assert order.state == "FILLED"
        assert order.price is None
        assert order.time_in_force is None
        assert order.client_extensions == {}
    
    def test_parse_order_data_negative_units(self, provider):
        """Test parsing order data with negative units (sell order)."""
        order_data = {
            "id": "12345",
            "instrument": "EUR_USD",
            "units": "-1000",
            "type": "MARKET",
            "state": "FILLED",
            "createTime": "2023-01-01T10:00:00.000000000Z"
        }
        
        order = provider._parse_order_data(order_data)
        
        assert order.units == -1000.0