"""
Property-based tests for trading emulation.

These tests verify universal properties that should hold across all valid inputs
for the emulation system in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.emulator.emulator import WarspiteTradingEmulator, Trade, EmulationStep, EmulationResult


# Hypothesis strategies for generating test data
@st.composite
def valid_emulator_dataset(draw):
    """Generate valid dataset for emulator testing."""
    # Generate a dataset with at least 5 timestamps for meaningful testing
    num_timestamps = draw(st.integers(5, 20))
    num_symbols = draw(st.integers(1, 3))
    
    symbols = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=num_symbols,
        max_size=num_symbols,
        unique=True
    ))
    
    # Generate sequential timestamps (daily intervals)
    base_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 1, 1)
    ))
    
    timestamps = []
    for i in range(num_timestamps):
        timestamps.append(base_date + timedelta(days=i))
    
    timestamps = np.array(timestamps, dtype='datetime64[ns]')
    
    # Generate price data arrays (OHLCV format)
    data_arrays = []
    for _ in range(num_symbols):
        # Generate realistic price data
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0))
        
        # Generate OHLCV data with realistic constraints
        ohlcv_data = np.zeros((num_timestamps, 5))
        
        current_price = base_price
        for i in range(num_timestamps):
            # Generate daily price movement
            price_change = draw(st.floats(min_value=-0.1, max_value=0.1))
            current_price = max(0.01, current_price * (1 + price_change))
            
            # Generate OHLCV for this day
            open_price = current_price
            close_price = max(0.01, current_price * (1 + draw(st.floats(min_value=-0.05, max_value=0.05))))
            
            high_price = max(open_price, close_price) * (1 + draw(st.floats(min_value=0.0, max_value=0.03)))
            low_price = min(open_price, close_price) * (1 - draw(st.floats(min_value=0.0, max_value=0.03)))
            
            volume = draw(st.floats(min_value=1000, max_value=1000000))
            
            ohlcv_data[i] = [open_price, high_price, low_price, close_price, volume]
            current_price = close_price
        
        data_arrays.append(ohlcv_data)
    
    return WarspiteDataset(data_arrays, timestamps, symbols)


@st.composite
def valid_emulator_config(draw):
    """Generate valid emulator configuration parameters."""
    initial_capital = draw(st.floats(min_value=1000.0, max_value=100000.0))
    trading_fee = draw(st.floats(min_value=0.0, max_value=10.0))
    spread = draw(st.floats(min_value=0.0, max_value=0.01))
    
    return initial_capital, trading_fee, spread


@st.composite
def valid_trade_parameters(draw, symbols):
    """Generate valid trade parameters for testing."""
    symbol = draw(st.sampled_from(symbols))
    quantity = draw(st.floats(min_value=0.1, max_value=100.0))
    
    return symbol, quantity


class TestEmulatorTradeExecutionLogic:
    """
    Property-based tests for Emulator Trade Execution Logic.
    
    **Feature: warspite-financial-library, Property 5: Emulator Trade Execution Logic**
    **Validates: Requirements 5.2**
    """
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=100, deadline=None)
    def test_emulator_trade_execution_consistency(self, dataset, config):
        """
        Property 5: Emulator Trade Execution Logic
        
        For any sequence of valid trading signals, the emulator should execute trades 
        that correctly reflect the signal timing and portfolio constraints.
        
        **Feature: warspite-financial-library, Property 5: Emulator Trade Execution Logic**
        **Validates: Requirements 5.2**
        """
        initial_capital, trading_fee, spread = config
        
        # Create emulator
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        # Property assertions: Trade execution consistency
        
        # 1. Initial state should be correct
        assert emulator.cash == initial_capital, "Initial cash should equal initial capital"
        assert emulator.get_portfolio_value() == initial_capital, "Initial portfolio value should equal initial capital"
        assert all(pos == 0.0 for pos in emulator.positions.values()), "Initial positions should be zero"
        assert len(emulator.trade_history) == 0, "Initial trade history should be empty"
        
        # 2. Execute some trades and verify consistency
        if len(dataset) > 0:
            # Move to first step to have valid prices
            emulator.step_forward()
            
            current_prices = emulator.get_current_prices()
            
            # Test buy operation
            test_symbol = dataset.symbols[0]
            test_quantity = min(10.0, initial_capital / (current_prices[test_symbol] * 2))  # Ensure we can afford it
            
            if test_quantity > 0:
                initial_cash = emulator.cash
                initial_position = emulator.positions[test_symbol]
                
                # Execute buy
                buy_success = emulator.buy(test_symbol, test_quantity)
                
                if buy_success:
                    # Verify trade execution consistency
                    ask_price = current_prices[test_symbol] * (1 + spread / 2)
                    expected_cost = test_quantity * ask_price + trading_fee
                    
                    assert emulator.cash == initial_cash - expected_cost, \
                        "Cash should be reduced by trade cost"
                    assert emulator.positions[test_symbol] == initial_position + test_quantity, \
                        "Position should be increased by trade quantity"
                    assert len(emulator.trade_history) == 1, \
                        "Trade history should contain one trade"
                    
                    # Verify trade record
                    trade = emulator.trade_history[0]
                    assert isinstance(trade, Trade), "Trade should be Trade instance"
                    assert trade.symbol == test_symbol, "Trade symbol should match"
                    assert trade.action == 'buy', "Trade action should be 'buy'"
                    assert trade.quantity == test_quantity, "Trade quantity should match"
                    assert abs(trade.price - ask_price) < 1e-10, "Trade price should match ask price"
                    assert trade.fee == trading_fee, "Trade fee should match configured fee"
                
                # Test sell operation
                if emulator.positions[test_symbol] > 0:
                    sell_quantity = min(test_quantity / 2, emulator.positions[test_symbol])
                    
                    pre_sell_cash = emulator.cash
                    pre_sell_position = emulator.positions[test_symbol]
                    
                    sell_success = emulator.sell(test_symbol, sell_quantity)
                    
                    if sell_success:
                        bid_price = current_prices[test_symbol] * (1 - spread / 2)
                        expected_proceeds = sell_quantity * bid_price - trading_fee
                        
                        assert emulator.cash == pre_sell_cash + expected_proceeds, \
                            "Cash should be increased by trade proceeds"
                        assert emulator.positions[test_symbol] == pre_sell_position - sell_quantity, \
                            "Position should be decreased by sell quantity"
        
        # 3. Portfolio value should always be non-negative
        portfolio_value = emulator.get_portfolio_value()
        assert portfolio_value >= 0, "Portfolio value should never be negative"
        
        # 4. Cash should never exceed initial capital + proceeds from all sales
        total_sales_proceeds = sum(
            trade.quantity * trade.price - trade.fee 
            for trade in emulator.trade_history 
            if trade.action == 'sell'
        )
        max_possible_cash = initial_capital + total_sales_proceeds
        assert emulator.cash <= max_possible_cash + 1e-10, \
            "Cash should not exceed initial capital plus sales proceeds"
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=50, deadline=None)
    def test_trade_validation_and_constraints(self, dataset, config):
        """
        Test that trade validation and constraints are properly enforced.
        """
        initial_capital, trading_fee, spread = config
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        if len(dataset) > 0:
            emulator.step_forward()
            
            test_symbol = dataset.symbols[0]
            current_prices = emulator.get_current_prices()
            
            # Test invalid trade parameters
            
            # 1. Negative quantity should fail
            assert not emulator.buy(test_symbol, -1.0), "Negative buy quantity should fail"
            assert not emulator.sell(test_symbol, -1.0), "Negative sell quantity should fail"
            
            # 2. Zero quantity should fail
            assert not emulator.buy(test_symbol, 0.0), "Zero buy quantity should fail"
            assert not emulator.sell(test_symbol, 0.0), "Zero sell quantity should fail"
            
            # 3. Invalid symbol should fail
            assert not emulator.buy("INVALID_SYMBOL", 1.0), "Invalid symbol should fail"
            assert not emulator.sell("INVALID_SYMBOL", 1.0), "Invalid symbol should fail"
            
            # 4. Insufficient cash should prevent buy
            ask_price = current_prices[test_symbol] * (1 + spread / 2)
            max_affordable_quantity = (emulator.cash - trading_fee) / ask_price
            
            if max_affordable_quantity > 0:
                # Try to buy more than we can afford
                excessive_quantity = max_affordable_quantity * 2
                assert not emulator.buy(test_symbol, excessive_quantity), \
                    "Insufficient cash should prevent excessive buy"
            
            # 5. Insufficient position should prevent sell
            # First buy something
            affordable_quantity = max(0.1, max_affordable_quantity / 2)
            if affordable_quantity > 0:
                emulator.buy(test_symbol, affordable_quantity)
                
                # Try to sell more than we have
                excessive_sell = emulator.positions[test_symbol] * 2
                assert not emulator.sell(test_symbol, excessive_sell), \
                    "Insufficient position should prevent excessive sell"
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=50, deadline=None)
    def test_step_forward_consistency(self, dataset, config):
        """
        Test that step_forward method maintains consistency.
        """
        initial_capital, trading_fee, spread = config
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        # Test stepping through the dataset
        step_count = 0
        while emulator.current_step < len(dataset):
            pre_step = emulator.current_step
            pre_portfolio_value = emulator.get_portfolio_value()
            
            # Execute step
            step_result = emulator.step_forward()
            
            # Verify step result consistency
            assert isinstance(step_result, EmulationStep), "Step should return EmulationStep"
            assert emulator.current_step == pre_step + 1, "Step should advance by 1"
            assert step_result.cash == emulator.cash, "Step result cash should match emulator cash"
            assert step_result.positions == emulator.positions, "Step result positions should match"
            assert step_result.portfolio_value == emulator.get_portfolio_value(), \
                "Step result portfolio value should match calculated value"
            
            # Portfolio history should be updated
            assert len(emulator.portfolio_history) == step_count + 2, \
                "Portfolio history should grow with each step"
            
            step_count += 1
        
        # Should not be able to step beyond dataset
        with pytest.raises(ValueError, match="Emulation has reached the end of the dataset"):
            emulator.step_forward()
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=30, deadline=None)
    def test_run_to_completion_consistency(self, dataset, config):
        """
        Test that run_to_completion produces consistent results.
        """
        initial_capital, trading_fee, spread = config
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        # Run to completion
        result = emulator.run_to_completion()
        
        # Verify result consistency
        assert isinstance(result, EmulationResult), "Should return EmulationResult"
        assert result.initial_capital == initial_capital, "Initial capital should match"
        assert result.final_portfolio_value == emulator.get_portfolio_value(), \
            "Final portfolio value should match emulator state"
        
        expected_return = (result.final_portfolio_value - initial_capital) / initial_capital
        assert abs(result.total_return - expected_return) < 1e-10, \
            "Total return should be calculated correctly"
        
        assert result.total_trades == len(emulator.trade_history), \
            "Total trades should match trade history length"
        assert result.trade_history == emulator.trade_history, \
            "Trade history should match emulator history"
        assert result.final_positions == emulator.positions, \
            "Final positions should match emulator positions"
        
        # Portfolio history should have correct length
        assert len(result.portfolio_history) == len(dataset) + 1, \
            "Portfolio history should have entry for each step plus initial"
        assert len(result.timestamps) == len(dataset) + 1, \
            "Timestamp history should match portfolio history length"
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=30, deadline=None)
    def test_position_management_consistency(self, dataset, config):
        """
        Test that position management operations are consistent.
        """
        initial_capital, trading_fee, spread = config
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        if len(dataset) > 0:
            emulator.step_forward()
            
            test_symbol = dataset.symbols[0]
            current_prices = emulator.get_current_prices()
            
            # Test apply_positions method
            ask_price = current_prices[test_symbol] * (1 + spread / 2)
            max_affordable = (emulator.cash - trading_fee) / ask_price / 2  # Conservative amount
            
            if max_affordable > 0.1:
                target_positions = {test_symbol: max_affordable}
                
                # Apply positions
                success = emulator.apply_positions(target_positions)
                
                if success:
                    assert abs(emulator.positions[test_symbol] - max_affordable) < 1e-10, \
                        "Position should match target after apply_positions"
                
                # Test close position
                if emulator.positions[test_symbol] > 0:
                    close_success = emulator.close(test_symbol)
                    
                    if close_success:
                        assert abs(emulator.positions[test_symbol]) < 1e-10, \
                            "Position should be zero after close"
                
                # Test close_all_positions
                # First create some positions
                if max_affordable > 0.1:
                    emulator.buy(test_symbol, max_affordable / 2)
                    
                    close_all_success = emulator.close_all_positions()
                    
                    if close_all_success:
                        assert all(abs(pos) < 1e-10 for pos in emulator.positions.values()), \
                            "All positions should be zero after close_all_positions"


class TestTradingCostApplicationAccuracy:
    """
    Property-based tests for Trading Cost Application Accuracy.
    
    **Feature: warspite-financial-library, Property 6: Trading Cost Application Accuracy**
    **Validates: Requirements 5.4**
    """
    
    @given(
        dataset=valid_emulator_dataset(),
        config=valid_emulator_config()
    )
    @settings(max_examples=100, deadline=None)
    def test_trading_cost_application_accuracy(self, dataset, config):
        """
        Property 6: Trading Cost Application Accuracy
        
        For any trade execution with configured spreads and fees, the final portfolio 
        value should correctly reflect all trading costs applied.
        
        **Feature: warspite-financial-library, Property 6: Trading Cost Application Accuracy**
        **Validates: Requirements 5.4**
        """
        initial_capital, trading_fee, spread = config
        
        # Create emulator with specific costs
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        if len(dataset) > 0:
            # Move to first step to have valid prices
            emulator.step_forward()
            
            current_prices = emulator.get_current_prices()
            test_symbol = dataset.symbols[0]
            base_price = current_prices[test_symbol]
            
            # Calculate expected prices with spread
            ask_price = base_price * (1 + spread / 2)
            bid_price = base_price * (1 - spread / 2)
            
            # Test buy operation cost accuracy
            test_quantity = min(5.0, (initial_capital - trading_fee) / ask_price / 2)
            
            if test_quantity > 0.1:
                initial_cash = emulator.cash
                initial_portfolio_value = emulator.get_portfolio_value()
                
                # Execute buy trade
                buy_success = emulator.buy(test_symbol, test_quantity)
                
                if buy_success:
                    # Property assertions: Cost application accuracy
                    
                    # 1. Cash should be reduced by exact cost (quantity * ask_price + fee)
                    expected_cost = test_quantity * ask_price + trading_fee
                    actual_cash_reduction = initial_cash - emulator.cash
                    
                    assert abs(actual_cash_reduction - expected_cost) < 1e-10, \
                        f"Cash reduction should equal expected cost: {actual_cash_reduction} vs {expected_cost}"
                    
                    # 2. Trade record should reflect correct prices and fees
                    trade = emulator.trade_history[-1]
                    assert abs(trade.price - ask_price) < 1e-10, \
                        f"Trade price should equal ask price: {trade.price} vs {ask_price}"
                    assert trade.fee == trading_fee, \
                        f"Trade fee should equal configured fee: {trade.fee} vs {trading_fee}"
                    
                    # 3. Position value should reflect current market price (not trade price)
                    position_value = emulator.positions[test_symbol] * base_price
                    expected_portfolio_value = emulator.cash + position_value
                    actual_portfolio_value = emulator.get_portfolio_value()
                    
                    assert abs(actual_portfolio_value - expected_portfolio_value) < 1e-10, \
                        "Portfolio value should be calculated using current market prices"
                    
                    # 4. Total cost impact should be spread + fee
                    # The difference between what we paid and current value should equal spread cost + fee
                    spread_cost = test_quantity * (ask_price - base_price)
                    total_expected_cost_impact = spread_cost + trading_fee
                    actual_cost_impact = initial_portfolio_value - actual_portfolio_value
                    
                    assert abs(actual_cost_impact - total_expected_cost_impact) < 1e-10, \
                        f"Total cost impact should equal spread cost + fee: {actual_cost_impact} vs {total_expected_cost_impact}"
                    
                    # Test sell operation cost accuracy
                    sell_quantity = min(test_quantity / 2, emulator.positions[test_symbol])
                    
                    if sell_quantity > 0.1:
                        pre_sell_cash = emulator.cash
                        pre_sell_portfolio_value = emulator.get_portfolio_value()
                        
                        # Execute sell trade
                        sell_success = emulator.sell(test_symbol, sell_quantity)
                        
                        if sell_success:
                            # 5. Cash should be increased by exact proceeds (quantity * bid_price - fee)
                            expected_proceeds = sell_quantity * bid_price - trading_fee
                            actual_cash_increase = emulator.cash - pre_sell_cash
                            
                            assert abs(actual_cash_increase - expected_proceeds) < 1e-10, \
                                f"Cash increase should equal expected proceeds: {actual_cash_increase} vs {expected_proceeds}"
                            
                            # 6. Sell trade record should reflect correct prices and fees
                            sell_trade = emulator.trade_history[-1]
                            assert abs(sell_trade.price - bid_price) < 1e-10, \
                                f"Sell trade price should equal bid price: {sell_trade.price} vs {bid_price}"
                            assert sell_trade.fee == trading_fee, \
                                f"Sell trade fee should equal configured fee: {sell_trade.fee} vs {trading_fee}"
                            
                            # 7. Round-trip cost should equal 2 * fee + spread cost on quantity
                            if len(emulator.trade_history) >= 2:
                                buy_trade = emulator.trade_history[-2]
                                sell_trade = emulator.trade_history[-1]
                                
                                # Calculate round-trip cost
                                buy_cost = sell_quantity * buy_trade.price + buy_trade.fee
                                sell_proceeds = sell_quantity * sell_trade.price - sell_trade.fee
                                round_trip_cost = buy_cost - sell_proceeds
                                
                                # Expected round-trip cost: spread on quantity + 2 fees
                                expected_round_trip_cost = sell_quantity * (ask_price - bid_price) + 2 * trading_fee
                                
                                assert abs(round_trip_cost - expected_round_trip_cost) < 1e-10, \
                                    f"Round-trip cost should equal spread + 2*fee: {round_trip_cost} vs {expected_round_trip_cost}"
    
    @given(
        dataset=valid_emulator_dataset(),
        trading_fee=st.floats(min_value=0.0, max_value=5.0),
        spread=st.floats(min_value=0.0, max_value=0.005)
    )
    @settings(max_examples=50, deadline=None)
    def test_cost_accumulation_accuracy(self, dataset, trading_fee, spread):
        """
        Test that costs accumulate correctly over multiple trades.
        """
        initial_capital = 10000.0
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        
        if len(dataset) > 0:
            emulator.step_forward()
            
            test_symbol = dataset.symbols[0]
            current_prices = emulator.get_current_prices()
            base_price = current_prices[test_symbol]
            
            ask_price = base_price * (1 + spread / 2)
            bid_price = base_price * (1 - spread / 2)
            
            # Execute multiple small trades
            trade_quantity = min(1.0, (initial_capital - trading_fee) / ask_price / 10)
            
            if trade_quantity > 0.1:
                total_fees_paid = 0.0
                total_spread_cost = 0.0
                
                # Execute several buy trades
                for i in range(3):
                    if emulator.cash > trade_quantity * ask_price + trading_fee:
                        pre_trade_cash = emulator.cash
                        
                        buy_success = emulator.buy(test_symbol, trade_quantity)
                        
                        if buy_success:
                            # Track costs
                            total_fees_paid += trading_fee
                            total_spread_cost += trade_quantity * (ask_price - base_price)
                            
                            # Verify cash reduction
                            expected_cost = trade_quantity * ask_price + trading_fee
                            actual_cost = pre_trade_cash - emulator.cash
                            
                            assert abs(actual_cost - expected_cost) < 1e-10, \
                                f"Trade {i+1} cost should be accurate"
                
                # Execute several sell trades
                for i in range(2):
                    if emulator.positions[test_symbol] >= trade_quantity:
                        pre_trade_cash = emulator.cash
                        
                        sell_success = emulator.sell(test_symbol, trade_quantity)
                        
                        if sell_success:
                            # Track costs
                            total_fees_paid += trading_fee
                            total_spread_cost += trade_quantity * (base_price - bid_price)
                            
                            # Verify cash increase
                            expected_proceeds = trade_quantity * bid_price - trading_fee
                            actual_proceeds = emulator.cash - pre_trade_cash
                            
                            assert abs(actual_proceeds - expected_proceeds) < 1e-10, \
                                f"Sell trade {i+1} proceeds should be accurate"
                
                # Verify total cost accumulation
                total_fees_from_history = sum(trade.fee for trade in emulator.trade_history)
                assert abs(total_fees_from_history - total_fees_paid) < 1e-10, \
                    "Total fees in trade history should match expected"
                
                # Portfolio value should reflect all costs
                current_portfolio_value = emulator.get_portfolio_value()
                position_value = emulator.positions[test_symbol] * base_price
                
                # The difference from initial capital should account for all costs
                total_cost_impact = initial_capital - current_portfolio_value
                
                # This should be approximately equal to total fees + spread costs
                # (allowing for some rounding in the position value calculation)
                expected_total_cost = total_fees_paid + total_spread_cost
                
                assert abs(total_cost_impact - expected_total_cost) < 1e-8, \
                    f"Total cost impact should match fees + spread costs: {total_cost_impact} vs {expected_total_cost}"
    
    @given(
        dataset=valid_emulator_dataset()
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_cost_trading(self, dataset):
        """
        Test trading with zero fees and spread to verify cost calculation logic.
        """
        initial_capital = 10000.0
        
        # Create emulator with zero costs
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=0.0,
            spread=0.0
        )
        
        if len(dataset) > 0:
            emulator.step_forward()
            
            test_symbol = dataset.symbols[0]
            current_prices = emulator.get_current_prices()
            base_price = current_prices[test_symbol]
            
            # With zero costs, ask and bid prices should equal base price
            test_quantity = 5.0
            
            if test_quantity * base_price < initial_capital:
                initial_cash = emulator.cash
                
                # Execute buy trade
                buy_success = emulator.buy(test_symbol, test_quantity)
                
                if buy_success:
                    # With zero costs, cash should be reduced by exactly quantity * price
                    expected_cost = test_quantity * base_price
                    actual_cost = initial_cash - emulator.cash
                    
                    assert abs(actual_cost - expected_cost) < 1e-10, \
                        "With zero costs, buy should cost exactly quantity * price"
                    
                    # Portfolio value should remain unchanged (no trading costs)
                    portfolio_value = emulator.get_portfolio_value()
                    assert abs(portfolio_value - initial_capital) < 1e-10, \
                        "Portfolio value should be unchanged with zero trading costs"
                    
                    # Execute sell trade
                    pre_sell_cash = emulator.cash
                    
                    sell_success = emulator.sell(test_symbol, test_quantity)
                    
                    if sell_success:
                        # With zero costs, should get back exactly what we paid
                        expected_proceeds = test_quantity * base_price
                        actual_proceeds = emulator.cash - pre_sell_cash
                        
                        assert abs(actual_proceeds - expected_proceeds) < 1e-10, \
                            "With zero costs, sell should return exactly quantity * price"
                        
                        # Should be back to initial cash (round-trip with zero costs)
                        assert abs(emulator.cash - initial_cash) < 1e-10, \
                            "Round-trip with zero costs should return to initial cash"
                        
                        # Portfolio value should be back to initial
                        final_portfolio_value = emulator.get_portfolio_value()
                        assert abs(final_portfolio_value - initial_capital) < 1e-10, \
                            "Round-trip with zero costs should return to initial portfolio value"