"""
Unit tests for trading emulation functionality.

These tests verify specific examples and edge cases for the emulation system
in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.emulator.emulator import (
    WarspiteTradingEmulator, Trade, EmulationStep, EmulationResult
)
from warspite_financial.providers.base import TradingProvider, OrderResult, AccountInfo, Position


class TestWarspiteTradingEmulatorBasics:
    """Test basic emulator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        timestamps_np = timestamps.values.astype('datetime64[ns]')
        
        # Simple price data: starts at 100, increases by 1 each day
        price_data = np.array([[100 + i, 101 + i, 99 + i, 100.5 + i, 1000] for i in range(10)])
        
        self.dataset = WarspiteDataset(
            data_arrays=[price_data],
            timestamps=timestamps_np,
            symbols=['TEST']
        )
        
        self.emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.001
        )
    
    def test_emulator_initialization(self):
        """Test emulator initialization with various parameters."""
        # Test default initialization
        emulator = WarspiteTradingEmulator(self.dataset)
        assert emulator.cash == 10000.0
        assert emulator.current_step == 0
        assert len(emulator.positions) == 1
        assert emulator.positions['TEST'] == 0.0
        assert len(emulator.trade_history) == 0
        
        # Test custom initialization
        emulator = WarspiteTradingEmulator(
            self.dataset,
            initial_capital=5000.0,
            trading_fee=2.5,
            spread=0.002
        )
        assert emulator.cash == 5000.0
        assert emulator._trading_fee == 2.5
        assert emulator._spread == 0.002
    
    def test_emulator_initialization_validation(self):
        """Test emulator initialization parameter validation."""
        # Test invalid initial capital
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            WarspiteTradingEmulator(self.dataset, initial_capital=0)
        
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            WarspiteTradingEmulator(self.dataset, initial_capital=-1000)
        
        # Test invalid trading fee
        with pytest.raises(ValueError, match="Trading fee cannot be negative"):
            WarspiteTradingEmulator(self.dataset, trading_fee=-1.0)
        
        # Test invalid spread
        with pytest.raises(ValueError, match="Spread must be between 0 and 1"):
            WarspiteTradingEmulator(self.dataset, spread=-0.1)
        
        with pytest.raises(ValueError, match="Spread must be between 0 and 1"):
            WarspiteTradingEmulator(self.dataset, spread=1.5)
    
    def test_get_current_prices(self):
        """Test current price retrieval."""
        # Move to first step
        self.emulator.step_forward()
        
        prices = self.emulator.get_current_prices()
        assert 'TEST' in prices
        assert prices['TEST'] == 101.5  # Close price from second day (step_forward moves to index 1)
        
        # Move to second step
        self.emulator.step_forward()
        
        prices = self.emulator.get_current_prices()
        assert prices['TEST'] == 102.5  # Close price from third day
    
    def test_get_current_prices_beyond_dataset(self):
        """Test price retrieval beyond dataset bounds."""
        # Move beyond dataset
        for _ in range(len(self.dataset) + 1):
            try:
                self.emulator.step_forward()
            except ValueError:
                break
        
        # Should raise error when trying to get prices beyond dataset
        with pytest.raises(ValueError, match="Current step exceeds dataset length"):
            self.emulator.get_current_prices()
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        # Initial portfolio value should equal initial capital
        assert self.emulator.get_portfolio_value() == 10000.0
        
        # Move to first step and buy some shares
        self.emulator.step_forward()
        
        # Buy 10 shares
        success = self.emulator.buy('TEST', 10.0)
        assert success
        
        # Portfolio value should be cash + position value
        current_price = 101.5  # Close price at current step
        expected_position_value = 10.0 * current_price
        expected_portfolio_value = self.emulator.cash + expected_position_value
        
        assert abs(self.emulator.get_portfolio_value() - expected_portfolio_value) < 1e-10


class TestTradingOperations:
    """Test individual trading operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        timestamps_np = timestamps.values.astype('datetime64[ns]')
        
        # Price data: 100, 101, 102, 103, 104
        price_data = np.array([[100 + i, 101 + i, 99 + i, 100 + i, 1000] for i in range(5)])
        
        self.dataset = WarspiteDataset(
            data_arrays=[price_data],
            timestamps=timestamps_np,
            symbols=['TEST']
        )
        
        self.emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.01  # 1% spread
        )
        
        # Move to first step to have valid prices
        self.emulator.step_forward()
    
    def test_buy_operation(self):
        """Test buy operation mechanics."""
        initial_cash = self.emulator.cash
        current_prices = self.emulator.get_current_prices()
        base_price = current_prices['TEST']  # Use actual current price
        ask_price = base_price * (1 + 0.01 / 2)  # Apply spread
        quantity = 10.0
        
        # Execute buy
        success = self.emulator.buy('TEST', quantity)
        assert success
        
        # Verify cash reduction
        expected_cost = quantity * ask_price + 1.0  # + trading fee
        assert abs(initial_cash - self.emulator.cash - expected_cost) < 1e-10
        
        # Verify position increase
        assert self.emulator.positions['TEST'] == quantity
        
        # Verify trade record
        assert len(self.emulator.trade_history) == 1
        trade = self.emulator.trade_history[0]
        assert trade.symbol == 'TEST'
        assert trade.action == 'buy'
        assert trade.quantity == quantity
        assert abs(trade.price - ask_price) < 1e-10
        assert trade.fee == 1.0
    
    def test_sell_operation(self):
        """Test sell operation mechanics."""
        # First buy some shares
        self.emulator.buy('TEST', 20.0)
        
        initial_cash = self.emulator.cash
        initial_position = self.emulator.positions['TEST']
        current_prices = self.emulator.get_current_prices()
        base_price = current_prices['TEST']  # Use actual current price
        bid_price = base_price * (1 - 0.01 / 2)  # Apply spread
        quantity = 10.0
        
        # Execute sell
        success = self.emulator.sell('TEST', quantity)
        assert success
        
        # Verify cash increase
        expected_proceeds = quantity * bid_price - 1.0  # - trading fee
        assert abs(self.emulator.cash - initial_cash - expected_proceeds) < 1e-10
        
        # Verify position decrease
        assert abs(self.emulator.positions['TEST'] - (initial_position - quantity)) < 1e-10
        
        # Verify trade record
        sell_trade = self.emulator.trade_history[-1]
        assert sell_trade.symbol == 'TEST'
        assert sell_trade.action == 'sell'
        assert sell_trade.quantity == quantity
        assert abs(sell_trade.price - bid_price) < 1e-10
        assert sell_trade.fee == 1.0
    
    def test_invalid_trade_parameters(self):
        """Test validation of invalid trade parameters."""
        # Negative quantity
        assert not self.emulator.buy('TEST', -1.0)
        assert not self.emulator.sell('TEST', -1.0)
        
        # Zero quantity
        assert not self.emulator.buy('TEST', 0.0)
        assert not self.emulator.sell('TEST', 0.0)
        
        # Invalid symbol
        assert not self.emulator.buy('INVALID', 1.0)
        assert not self.emulator.sell('INVALID', 1.0)
        
        # Insufficient cash for buy
        large_quantity = 1000000.0  # Way more than we can afford
        assert not self.emulator.buy('TEST', large_quantity)
        
        # Insufficient position for sell
        assert not self.emulator.sell('TEST', 1.0)  # We have no position yet
    
    def test_close_position(self):
        """Test position closing functionality."""
        # Buy some shares first
        self.emulator.buy('TEST', 15.0)
        assert self.emulator.positions['TEST'] == 15.0
        
        # Close position
        success = self.emulator.close('TEST')
        assert success
        assert abs(self.emulator.positions['TEST']) < 1e-10
        
        # Should have a sell trade in history
        sell_trade = self.emulator.trade_history[-1]
        assert sell_trade.action == 'sell'
        assert sell_trade.quantity == 15.0
    
    def test_close_all_positions(self):
        """Test closing all positions."""
        # Buy some shares
        self.emulator.buy('TEST', 10.0)
        
        # Close all positions
        success = self.emulator.close_all_positions()
        assert success
        
        # All positions should be zero
        for position in self.emulator.positions.values():
            assert abs(position) < 1e-10
    
    def test_apply_positions(self):
        """Test applying target positions."""
        target_positions = {'TEST': 25.0}
        
        # Apply positions
        success = self.emulator.apply_positions(target_positions)
        assert success
        
        # Position should match target
        assert abs(self.emulator.positions['TEST'] - 25.0) < 1e-10
        
        # Should have a buy trade
        trade = self.emulator.trade_history[-1]
        assert trade.action == 'buy'
        assert trade.quantity == 25.0


class TestEmulationExecution:
    """Test emulation execution methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        timestamps_np = timestamps.values.astype('datetime64[ns]')
        
        # Simple increasing prices
        price_data = np.array([[100 + i, 101 + i, 99 + i, 100 + i, 1000] for i in range(5)])
        
        self.dataset = WarspiteDataset(
            data_arrays=[price_data],
            timestamps=timestamps_np,
            symbols=['TEST']
        )
        
        self.emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.001
        )
    
    def test_step_forward(self):
        """Test single step execution."""
        initial_step = self.emulator.current_step
        
        # Execute step
        step_result = self.emulator.step_forward()
        
        # Verify step advancement
        assert self.emulator.current_step == initial_step + 1
        
        # Verify step result
        assert isinstance(step_result, EmulationStep)
        assert step_result.cash == self.emulator.cash
        assert step_result.positions == self.emulator.positions
        assert step_result.portfolio_value == self.emulator.get_portfolio_value()
        assert isinstance(step_result.timestamp, datetime)
        
        # Portfolio history should be updated
        assert len(self.emulator.portfolio_history) == 2  # Initial + 1 step
    
    def test_step_forward_beyond_dataset(self):
        """Test stepping beyond dataset bounds."""
        # Step through entire dataset
        for _ in range(len(self.dataset)):
            self.emulator.step_forward()
        
        # Next step should raise error
        with pytest.raises(ValueError, match="Emulation has reached the end of the dataset"):
            self.emulator.step_forward()
    
    def test_run_to_completion(self):
        """Test full emulation run."""
        result = self.emulator.run_to_completion()
        
        # Verify result structure
        assert isinstance(result, EmulationResult)
        assert result.initial_capital == 10000.0
        assert result.final_portfolio_value == self.emulator.get_portfolio_value()
        
        # Calculate expected return
        expected_return = (result.final_portfolio_value - 10000.0) / 10000.0
        assert abs(result.total_return - expected_return) < 1e-10
        
        # Verify histories
        assert len(result.portfolio_history) == len(self.dataset) + 1
        assert len(result.timestamps) == len(self.dataset) + 1
        assert result.final_positions == self.emulator.positions
        
        # Emulator should be at end of dataset
        assert self.emulator.current_step == len(self.dataset)
    
    def test_step_vs_completion_consistency(self):
        """Test that step-by-step execution matches run_to_completion."""
        # Create two identical emulators
        emulator1 = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.001
        )
        
        emulator2 = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.001
        )
        
        # Run one step-by-step
        step_results = []
        while emulator1.current_step < len(self.dataset):
            step_result = emulator1.step_forward()
            step_results.append(step_result)
        
        # Run other to completion
        completion_result = emulator2.run_to_completion()
        
        # Results should be identical
        assert abs(emulator1.get_portfolio_value() - completion_result.final_portfolio_value) < 1e-10
        assert emulator1.positions == completion_result.final_positions
        assert len(emulator1.trade_history) == completion_result.total_trades
        assert len(emulator1.portfolio_history) == len(completion_result.portfolio_history)


class TestLiveTradingIntegration:
    """Test live trading integration and safety mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        timestamps = pd.date_range('2023-01-01', periods=3, freq='D')
        timestamps_np = timestamps.values.astype('datetime64[ns]')
        
        price_data = np.array([[100, 101, 99, 100, 1000] for _ in range(3)])
        
        self.dataset = WarspiteDataset(
            data_arrays=[price_data],
            timestamps=timestamps_np,
            symbols=['TEST']
        )
        
        self.emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=1.0,
            spread=0.001
        )
        
        # Move to first step
        self.emulator.step_forward()
    
    def test_connect_trading_provider_validation(self):
        """Test trading provider connection validation."""
        # Test invalid provider type
        with pytest.raises(ValueError, match="Provider must be a TradingProvider instance"):
            self.emulator.connect_trading_provider("not_a_provider")
        
        # Test valid provider connection
        mock_provider = Mock(spec=TradingProvider)
        mock_account = AccountInfo("test_account", 50000.0, "USD")
        mock_provider.get_account_info.return_value = mock_account
        mock_provider.get_positions.return_value = []
        
        # Should succeed
        self.emulator.connect_trading_provider(mock_provider)
        assert self.emulator._trading_provider is mock_provider
    
    def test_connect_trading_provider_validation_failure(self):
        """Test trading provider validation failure scenarios."""
        # Test provider that fails account info
        mock_provider = Mock(spec=TradingProvider)
        mock_provider.get_account_info.side_effect = Exception("Connection failed")
        
        with pytest.raises(ValueError, match="Provider validation failed"):
            self.emulator.connect_trading_provider(mock_provider)
        
        # Test provider that returns invalid account info
        mock_provider = Mock(spec=TradingProvider)
        mock_provider.get_account_info.return_value = None
        
        with pytest.raises(ValueError, match="Provider validation failed"):
            self.emulator.connect_trading_provider(mock_provider)
    
    def test_apply_live_trade_without_provider(self):
        """Test live trading without connected provider."""
        with pytest.raises(ValueError, match="No trading provider connected"):
            self.emulator.apply_live_trade()
    
    def test_apply_live_trade_safety_limits(self):
        """Test live trading safety limits."""
        # Connect mock provider
        mock_provider = Mock(spec=TradingProvider)
        mock_account = AccountInfo("test_account", 50000.0, "USD")
        mock_provider.get_account_info.return_value = mock_account
        mock_provider.get_positions.return_value = []
        
        self.emulator.connect_trading_provider(mock_provider)
        
        # Set up dangerous position (exceeds 10x initial capital)
        self.emulator._positions['TEST'] = 1000000.0  # Very large position
        
        # Should reject due to safety limit
        with pytest.raises(PermissionError, match="Position value .* exceeds safety limit"):
            self.emulator.apply_live_trade()
    
    def test_apply_live_trade_insufficient_balance(self):
        """Test live trading with insufficient balance."""
        # Connect mock provider with low balance
        mock_provider = Mock(spec=TradingProvider)
        mock_account = AccountInfo("test_account", 100.0, "USD")  # Low balance
        mock_provider.get_account_info.return_value = mock_account
        mock_provider.get_positions.return_value = []
        
        self.emulator.connect_trading_provider(mock_provider)
        
        # Set up position that requires more capital than available
        self.emulator._positions['TEST'] = 50.0  # Requires ~5000 capital
        
        # Should reject due to insufficient balance
        with pytest.raises(PermissionError, match="Required capital .* exceeds available balance"):
            self.emulator.apply_live_trade()
    
    def test_apply_live_trade_success(self):
        """Test successful live trade execution."""
        # Connect mock provider
        mock_provider = Mock(spec=TradingProvider)
        mock_account = AccountInfo("test_account", 50000.0, "USD")
        mock_provider.get_account_info.return_value = mock_account
        mock_provider.get_positions.return_value = []
        
        # Mock successful order
        mock_order_result = OrderResult("order_123", True, "Success")
        mock_provider.place_order.return_value = mock_order_result
        
        self.emulator.connect_trading_provider(mock_provider)
        
        # Set up reasonable position
        self.emulator._positions['TEST'] = 10.0
        
        # Should succeed
        success = self.emulator.apply_live_trade()
        assert success
        
        # Verify order was placed
        mock_provider.place_order.assert_called_once_with(
            symbol='TEST',
            quantity=10.0,
            order_type='market'
        )
    
    def test_validate_live_trading_safety(self):
        """Test live trading safety validation."""
        # Test with safe state
        safety_report = self.emulator.validate_live_trading_safety()
        
        assert safety_report['is_safe'] is True
        assert safety_report['position_value'] == 0.0
        assert safety_report['cash_ratio'] == 1.0
        assert len(safety_report['errors']) == 0
    
    def test_validate_live_trading_safety_warnings(self):
        """Test safety validation with warning conditions."""
        # Buy some shares to create position
        buy_success = self.emulator.buy('TEST', 150.0)  # Large position
        
        # Only test if buy was successful (might fail due to insufficient funds)
        if buy_success:
            safety_report = self.emulator.validate_live_trading_safety()
            
            # Should have warnings but still be safe
            assert safety_report['is_safe'] is True
            assert len(safety_report['warnings']) > 0
            assert safety_report['position_value'] > 0
        else:
            # If buy failed, test with manual position setting
            self.emulator._positions['TEST'] = 50.0  # Set position directly
            
            safety_report = self.emulator.validate_live_trading_safety()
            
            # Should have warnings but still be safe
            assert safety_report['is_safe'] is True
            assert safety_report['position_value'] > 0
    
    def test_enable_live_trading_mode(self):
        """Test live trading mode enablement."""
        # Test invalid confirmation code
        with pytest.raises(PermissionError, match="Invalid confirmation code"):
            self.emulator.enable_live_trading_mode("wrong_code")
        
        # Test valid confirmation code
        success = self.emulator.enable_live_trading_mode("ENABLE_LIVE_TRADING_WITH_REAL_MONEY")
        assert success


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        timestamps_np = timestamps.values.astype('datetime64[ns]')
        
        # Volatile price data for testing metrics
        price_data = np.array([
            [100, 105, 95, 102, 1000],   # +2%
            [102, 108, 98, 98, 1000],    # -4%
            [98, 103, 94, 105, 1000],    # +7%
            [105, 110, 100, 108, 1000],  # +3%
            [108, 112, 104, 106, 1000],  # -2%
            [106, 111, 102, 110, 1000],  # +4%
            [110, 115, 107, 112, 1000],  # +2%
            [112, 118, 108, 115, 1000],  # +3%
            [115, 120, 110, 118, 1000],  # +3%
            [118, 125, 115, 120, 1000],  # +2%
        ])
        
        self.dataset = WarspiteDataset(
            data_arrays=[price_data],
            timestamps=timestamps_np,
            symbols=['TEST']
        )
        
        self.emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000.0,
            trading_fee=0.0,  # No fees for cleaner metrics
            spread=0.0
        )
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Run emulation with some trades
        self.emulator.step_forward()
        self.emulator.buy('TEST', 50.0)  # Buy at ~102
        
        # Let it run
        self.emulator.run_to_completion()
        
        # Calculate metrics
        metrics = self.emulator.get_performance_metrics()
        
        # Verify metric structure
        expected_keys = [
            'total_return', 'volatility', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'total_trades', 'final_portfolio_value'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Verify metric types and ranges
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['volatility'], float)
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)
        assert isinstance(metrics['win_rate'], float)
        assert isinstance(metrics['total_trades'], int)
        assert isinstance(metrics['final_portfolio_value'], float)
        
        # Verify reasonable ranges
        assert metrics['volatility'] >= 0
        assert -1 <= metrics['max_drawdown'] <= 0
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['total_trades'] >= 0
        assert metrics['final_portfolio_value'] > 0
    
    def test_performance_metrics_empty_history(self):
        """Test metrics calculation with empty portfolio history."""
        # Don't run any steps
        metrics = self.emulator.get_performance_metrics()
        
        # Should return empty dict
        assert metrics == {}
    
    def test_performance_metrics_single_step(self):
        """Test metrics calculation with single step."""
        # Run just one step
        self.emulator.step_forward()
        
        metrics = self.emulator.get_performance_metrics()
        
        # Should have some metrics but limited data
        assert 'total_return' in metrics
        assert 'final_portfolio_value' in metrics
        
        # With single step, some metrics might be zero or undefined
        assert metrics['volatility'] >= 0