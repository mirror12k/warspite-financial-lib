"""
OANDA provider implementation for warspite_financial library.

This module implements the OANDA provider using REST API calls without SDK dependencies.
Provides both data retrieval and trading capabilities for forex and CFD markets.
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import TradingProvider, OrderResult, AccountInfo, Position, Order


class OANDAProvider(TradingProvider):
    """
    OANDA provider implementation using REST API calls.
    
    Supports both data retrieval and live trading for forex and CFD markets.
    No SDK dependencies - uses direct REST API calls.
    """
    
    # OANDA API endpoints
    LIVE_API_URL = "https://api-fxtrade.oanda.com"
    PRACTICE_API_URL = "https://api-fxpractice.oanda.com"
    
    # OANDA-specific intervals mapping
    OANDA_INTERVALS = {
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
    
    def __init__(self, api_token: str, account_id: str, practice: bool = True):
        """
        Initialize OANDA provider.
        
        Args:
            api_token: OANDA API access token
            account_id: OANDA account ID
            practice: If True, use practice environment; if False, use live environment
        """
        super().__init__()
        
        if not api_token:
            raise ValueError("API token is required")
        if not account_id:
            raise ValueError("Account ID is required")
        
        self.api_token = api_token
        self.account_id = account_id
        self.practice = practice
        
        # Set base URL based on environment
        self.base_url = self.PRACTICE_API_URL if practice else self.LIVE_API_URL
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        })
        
        # Cache for available instruments
        self._instruments_cache = None
        self._cache_timestamp = None
        self._cache_duration = 3600  # 1 hour cache
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated request to OANDA API.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            
        Returns:
            JSON response as dictionary
            
        Raises:
            ConnectionError: If API request fails
            ValueError: If API returns error response
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, params=params, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = self.session.put(url, params=params, json=data, timeout=30)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for HTTP errors
            if response.status_code == 401:
                raise PermissionError("Invalid API token or insufficient permissions")
            elif response.status_code == 404:
                raise ValueError("Resource not found - check account ID or instrument")
            elif response.status_code == 429:
                raise ConnectionError("Rate limit exceeded - please retry later")
            elif response.status_code >= 400:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'errorMessage' in error_data:
                        error_msg += f": {error_data['errorMessage']}"
                except:
                    error_msg += f": {response.text}"
                raise ConnectionError(error_msg)
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise ConnectionError("Request timeout - OANDA API may be unavailable")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Connection error - check internet connection")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request failed: {str(e)}")
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                 interval: str = '1d') -> pd.DataFrame:
        """
        Retrieve financial data from OANDA.
        
        Args:
            symbol: OANDA instrument name (e.g., 'EUR_USD', 'GBP_JPY')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        # Validate inputs
        self.validate_interval(interval)
        self.validate_date_range(start_date, end_date)
        
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid OANDA instrument: {symbol}")
        
        # Convert interval to OANDA format
        oanda_granularity = self.OANDA_INTERVALS.get(interval)
        if not oanda_granularity:
            raise ValueError(f"Unsupported interval for OANDA: {interval}")
        
        # Format dates for OANDA API (RFC3339)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        
        # Make API request
        endpoint = f"/v3/instruments/{symbol}/candles"
        params = {
            'granularity': oanda_granularity,
            'from': start_str,
            'to': end_str,
            'price': 'M',  # Mid prices
            'includeFirst': 'true'
        }
        
        try:
            response = self._make_request('GET', endpoint, params=params)
            
            if 'candles' not in response:
                raise ValueError(f"No candle data returned for {symbol}")
            
            candles = response['candles']
            if not candles:
                raise ValueError(f"No data available for {symbol} in the specified date range")
            
            # Convert to DataFrame
            data_rows = []
            for candle in candles:
                if candle['complete']:  # Only use complete candles
                    mid_prices = candle['mid']
                    data_rows.append({
                        'datetime': pd.to_datetime(candle['time']),
                        'Open': float(mid_prices['o']),
                        'High': float(mid_prices['h']),
                        'Low': float(mid_prices['l']),
                        'Close': float(mid_prices['c']),
                        'Volume': int(candle['volume'])
                    })
            
            if not data_rows:
                raise ValueError(f"No complete candle data available for {symbol}")
            
            df = pd.DataFrame(data_rows)
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve data from OANDA: {str(e)}")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available instruments from OANDA.
        
        Returns:
            List of available instrument names
        """
        # Check cache first
        current_time = time.time()
        if (self._instruments_cache is not None and 
            self._cache_timestamp is not None and
            current_time - self._cache_timestamp < self._cache_duration):
            return self._instruments_cache
        
        try:
            endpoint = f"/v3/accounts/{self.account_id}/instruments"
            response = self._make_request('GET', endpoint)
            
            if 'instruments' not in response:
                raise ValueError("No instruments data returned from OANDA")
            
            instruments = [instr['name'] for instr in response['instruments']]
            
            # Update cache
            self._instruments_cache = instruments
            self._cache_timestamp = current_time
            
            return instruments
            
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve instruments from OANDA: {str(e)}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if an instrument is available on OANDA.
        
        Args:
            symbol: OANDA instrument name to validate
            
        Returns:
            True if instrument is valid, False otherwise
        """
        try:
            available_symbols = self.get_available_symbols()
            return symbol in available_symbols
        except:
            # If we can't fetch symbols, try a simple format check
            # OANDA instruments are typically in format XXX_YYY for forex
            return '_' in symbol and len(symbol.split('_')) == 2
    
    # Trading Provider methods
    
    def get_orders(self, instrument: Optional[str] = None, state: str = "PENDING", 
                   count: int = 50, order_ids: Optional[List[str]] = None) -> List[Order]:
        """
        Get orders from OANDA account with optional filtering.
        
        Args:
            instrument: Filter orders by instrument (e.g., 'EUR_USD')
            state: Filter by order state ('PENDING', 'FILLED', 'CANCELLED', 'ALL')
            count: Maximum number of orders to return (default 50, max 500)
            order_ids: List of specific order IDs to retrieve
            
        Returns:
            List of Order objects
            
        Raises:
            ConnectionError: If API request fails
            ValueError: If parameters are invalid
        """
        if count > 500:
            raise ValueError("Maximum count is 500")
        
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders"
            params = {
                'count': count
            }
            
            # Add optional filters
            if instrument:
                params['instrument'] = instrument
            if state != "ALL":
                params['state'] = state
            if order_ids:
                params['ids'] = ','.join(order_ids)
            
            response = self._make_request('GET', endpoint, params=params)
            
            if 'orders' not in response:
                raise ValueError("No orders data returned from OANDA")
            
            orders = []
            for order_data in response['orders']:
                order = self._parse_order_data(order_data)
                orders.append(order)
            
            return orders
            
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve orders: {str(e)}")
    
    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders from OANDA account.
        
        Returns:
            List of Order objects with state 'PENDING'
            
        Raises:
            ConnectionError: If API request fails
        """
        try:
            endpoint = f"/v3/accounts/{self.account_id}/pendingOrders"
            response = self._make_request('GET', endpoint)
            
            if 'orders' not in response:
                raise ValueError("No orders data returned from OANDA")
            
            orders = []
            for order_data in response['orders']:
                order = self._parse_order_data(order_data)
                orders.append(order)
            
            return orders
            
        except Exception as e:
            if isinstance(e, (ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve pending orders: {str(e)}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get a specific order by ID from OANDA account.
        
        Args:
            order_id: The order ID or client order ID (prefixed with @)
            
        Returns:
            Order object if found, None otherwise
            
        Raises:
            ConnectionError: If API request fails
            ValueError: If order_id is invalid
        """
        if not order_id:
            raise ValueError("Order ID is required")
        
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}"
            response = self._make_request('GET', endpoint)
            
            if 'order' not in response:
                return None
            
            return self._parse_order_data(response['order'])
            
        except ValueError as e:
            if "Resource not found" in str(e):
                return None
            raise
        except Exception as e:
            if isinstance(e, PermissionError):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve order {order_id}: {str(e)}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.
        
        Args:
            order_id: The order ID or client order ID (prefixed with @) to cancel
            
        Returns:
            True if order was successfully cancelled, False otherwise
            
        Raises:
            ConnectionError: If API request fails
            ValueError: If order_id is invalid
        """
        if not order_id:
            raise ValueError("Order ID is required")
        
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            response = self._make_request('PUT', endpoint)
            
            # Check if cancellation was successful
            return 'orderCancelTransaction' in response
            
        except Exception as e:
            if isinstance(e, (ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to cancel order {order_id}: {str(e)}")
    
    def _parse_order_data(self, order_data: Dict[str, Any]) -> Order:
        """
        Parse order data from OANDA API response into Order object.
        
        Args:
            order_data: Raw order data from OANDA API
            
        Returns:
            Order object
        """
        order_id = order_data.get('id', '')
        instrument = order_data.get('instrument', '')
        units = float(order_data.get('units', 0))
        order_type = order_data.get('type', '')
        state = order_data.get('state', '')
        create_time = order_data.get('createTime', '')
        price = None
        if 'price' in order_data:
            price = float(order_data['price'])
        time_in_force = order_data.get('timeInForce')
        client_extensions = order_data.get('clientExtensions', {})
        
        return Order(
            order_id=order_id,
            instrument=instrument,
            units=units,
            order_type=order_type,
            state=state,
            create_time=create_time,
            price=price,
            time_in_force=time_in_force,
            client_extensions=client_extensions
        )
    
    def place_order(self, symbol: str, quantity: float, order_type: str = 'market') -> OrderResult:
        """
        Place a trading order on OANDA.
        
        Args:
            symbol: OANDA instrument name
            quantity: Order quantity (positive for buy, negative for sell)
            order_type: Type of order ('market', 'limit', etc.)
            
        Returns:
            OrderResult containing order details and status
        """
        if not self.validate_symbol(symbol):
            return OrderResult("", False, f"Invalid instrument: {symbol}")
        
        if quantity == 0:
            return OrderResult("", False, "Order quantity cannot be zero")
        
        # Determine order side and units
        units = int(quantity)  # OANDA uses integer units
        
        # Prepare order data
        order_data = {
            "order": {
                "type": order_type.upper(),
                "instrument": symbol,
                "units": str(units)
            }
        }
        
        try:
            endpoint = f"/v3/accounts/{self.account_id}/orders"
            response = self._make_request('POST', endpoint, data=order_data)
            
            if 'orderCreateTransaction' in response:
                transaction = response['orderCreateTransaction']
                order_id = transaction.get('id', '')
                return OrderResult(order_id, True, f"Order placed successfully: {order_id}")
            else:
                return OrderResult("", False, "Order creation failed - no transaction ID returned")
                
        except Exception as e:
            error_msg = f"Failed to place order: {str(e)}"
            return OrderResult("", False, error_msg)
    
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information from OANDA.
        
        Returns:
            AccountInfo with balance and account details
        """
        try:
            endpoint = f"/v3/accounts/{self.account_id}"
            response = self._make_request('GET', endpoint)
            
            if 'account' not in response:
                raise ValueError("No account data returned from OANDA")
            
            account = response['account']
            balance = float(account.get('balance', 0))
            currency = account.get('currency', 'USD')
            
            return AccountInfo(self.account_id, balance, currency)
            
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve account info: {str(e)}")
    
    def get_positions(self) -> List[Position]:
        """
        Get current open positions from OANDA.
        
        Returns:
            List of Position objects representing open positions
        """
        try:
            endpoint = f"/v3/accounts/{self.account_id}/positions"
            response = self._make_request('GET', endpoint)
            
            if 'positions' not in response:
                raise ValueError("No positions data returned from OANDA")
            
            positions = []
            for pos_data in response['positions']:
                # OANDA returns separate long and short positions
                long_units = float(pos_data.get('long', {}).get('units', 0))
                short_units = float(pos_data.get('short', {}).get('units', 0))
                
                # Calculate net position
                net_units = long_units + short_units  # short_units is negative
                
                if net_units != 0:  # Only include positions with net exposure
                    symbol = pos_data['instrument']
                    
                    # Get current price for P&L calculation
                    current_price = 0.0
                    unrealized_pnl = 0.0
                    
                    try:
                        # Get current price from pricing endpoint
                        price_endpoint = f"/v3/accounts/{self.account_id}/pricing"
                        price_params = {'instruments': symbol}
                        price_response = self._make_request('GET', price_endpoint, params=price_params)
                        
                        if 'prices' in price_response and price_response['prices']:
                            price_data = price_response['prices'][0]
                            # Use mid price
                            bid = float(price_data.get('bids', [{}])[0].get('price', 0))
                            ask = float(price_data.get('asks', [{}])[0].get('price', 0))
                            current_price = (bid + ask) / 2 if bid and ask else 0.0
                        
                        # Calculate unrealized P&L
                        if long_units != 0:
                            unrealized_pnl += float(pos_data.get('long', {}).get('unrealizedPL', 0))
                        if short_units != 0:
                            unrealized_pnl += float(pos_data.get('short', {}).get('unrealizedPL', 0))
                            
                    except:
                        # If we can't get current price, use 0
                        pass
                    
                    position = Position(
                        position_id=f"{symbol}_{self.account_id}",
                        symbol=symbol,
                        quantity=net_units,
                        current_price=current_price,
                        unrealized_pnl=unrealized_pnl
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError, PermissionError)):
                raise
            else:
                raise ConnectionError(f"Failed to retrieve positions: {str(e)}")
    
    def close_position(self, position_id: str) -> bool:
        """
        Close a specific position on OANDA.
        
        Args:
            position_id: Position identifier (format: "INSTRUMENT_ACCOUNTID")
            
        Returns:
            True if position was successfully closed, False otherwise
        """
        try:
            # Extract instrument from position_id
            if '_' not in position_id:
                return False
            
            parts = position_id.split('_')
            if len(parts) < 2:
                return False
            
            # Reconstruct instrument name (everything except the last part which is account_id)
            instrument = '_'.join(parts[:-1])
            
            # Close both long and short positions for this instrument
            endpoint = f"/v3/accounts/{self.account_id}/positions/{instrument}/close"
            
            close_data = {
                "longUnits": "ALL",
                "shortUnits": "ALL"
            }
            
            response = self._make_request('PUT', endpoint, data=close_data)
            
            # Check if any positions were actually closed
            long_closed = 'longOrderCreateTransaction' in response
            short_closed = 'shortOrderCreateTransaction' in response
            
            return long_closed or short_closed
            
        except Exception:
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions on OANDA.
        
        Returns:
            True if all positions were successfully closed, False otherwise
        """
        try:
            positions = self.get_positions()
            
            if not positions:
                return True  # No positions to close
            
            success_count = 0
            for position in positions:
                if self.close_position(position.position_id):
                    success_count += 1
            
            # Return True if all positions were closed successfully
            return success_count == len(positions)
            
        except Exception:
            return False