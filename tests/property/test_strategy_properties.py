"""
Property-based tests for strategy implementations.

These tests verify universal properties that should hold across all valid inputs
for the strategy system in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.strategies.base import BaseStrategy
from warspite_financial.strategies.random import RandomStrategy
from warspite_financial.strategies.perfect import PerfectStrategy
from warspite_financial.strategies.sma import SMAStrategy
from warspite_financial.strategies.buy_and_hold import BuyAndHoldStrategy
from warspite_financial.strategies.short import ShortStrategy


# Hypothesis strategies for generating test data
@st.composite
def valid_dataset_for_strategy(draw):
    """Generate valid datasets for strategy testing."""
    # Generate dataset with sufficient length for strategy calculations
    num_timestamps = draw(st.integers(5, 100))  # 5 to 100 timestamps
    
    # Generate symbols (1-3 symbols)
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
    
    # Generate data arrays with realistic financial data
    data_arrays = []
    for _ in range(num_symbols):
        # Decide if this should be 1D (single value) or 2D (OHLCV)
        is_ohlcv = draw(st.booleans())
        
        if is_ohlcv:
            # Generate OHLCV data (5 columns)
            array = draw(arrays(
                dtype=np.float64,
                shape=(num_timestamps, 5),
                elements=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
            ))
            
            # Ensure OHLCV constraints: High >= max(Open, Close), Low <= min(Open, Close)
            for i in range(num_timestamps):
                open_price = array[i, 0]
                close_price = array[i, 3]
                
                # Set High to be at least max(Open, Close)
                array[i, 1] = max(array[i, 1], max(open_price, close_price))
                
                # Set Low to be at most min(Open, Close)
                array[i, 2] = min(array[i, 2], min(open_price, close_price))
                
                # Ensure Volume is positive
                array[i, 4] = abs(array[i, 4])
        else:
            # Generate 1D data (single values, e.g., close prices)
            array = draw(arrays(
                dtype=np.float64,
                shape=(num_timestamps,),
                elements=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
            ))
        
        data_arrays.append(array)
    
    # Create dataset
    dataset = WarspiteDataset(data_arrays, timestamps, symbols)
    
    return dataset


@st.composite
def valid_random_strategy(draw):
    """Generate valid RandomStrategy instances."""
    correct_percent = draw(st.floats(min_value=0.0, max_value=1.0))
    seed = draw(st.one_of(st.none(), st.integers(0, 1000000)))
    return RandomStrategy(correct_percent=correct_percent, seed=seed)


@st.composite
def valid_perfect_strategy(draw):
    """Generate valid PerfectStrategy instances."""
    lookahead_periods = draw(st.integers(1, 10))  # Lookahead from 1 to 10
    return PerfectStrategy(lookahead_periods=lookahead_periods)


@st.composite
def valid_sma_strategy(draw):
    """Generate valid SMAStrategy instances."""
    days = draw(st.integers(1, 30))  # Days from 1 to 30
    return SMAStrategy(days=days)


def valid_buy_and_hold_strategy():
    """Generate valid BuyAndHoldStrategy instances."""
    return BuyAndHoldStrategy()


def valid_short_strategy():
    """Generate valid ShortStrategy instances."""
    return ShortStrategy()


@st.composite
def strategy_and_dataset(draw):
    """Generate a strategy and compatible dataset pair."""
    dataset = draw(valid_dataset_for_strategy())
    
    # Choose strategy type (random, perfect, sma, buy_and_hold, or short)
    strategy_type = draw(st.sampled_from(['random', 'perfect', 'sma', 'buy_and_hold', 'short']))
    
    if strategy_type == 'random':
        strategy = draw(valid_random_strategy())
    elif strategy_type == 'perfect':
        # Ensure dataset is long enough for perfect strategy lookahead
        max_lookahead = min(len(dataset) - 1, 10)
        assume(max_lookahead >= 1)
        lookahead_periods = draw(st.integers(1, max_lookahead))
        strategy = PerfectStrategy(lookahead_periods=lookahead_periods)
    elif strategy_type == 'sma':
        # Ensure dataset is long enough for SMA calculation
        max_days = min(len(dataset), 30)
        assume(max_days >= 1)
        days = draw(st.integers(1, max_days))
        strategy = SMAStrategy(days=days)
    elif strategy_type == 'buy_and_hold':
        strategy = BuyAndHoldStrategy()
    else:  # short
        strategy = ShortStrategy()
    
    return strategy, dataset


class TestStrategySignalGenerationConsistency:
    """
    Property-based tests for Strategy Signal Generation Consistency.
    
    **Feature: warspite-financial-library, Property 4: Strategy Signal Generation Consistency**
    **Validates: Requirements 4.4**
    """
    
    @given(strategy_dataset=strategy_and_dataset())
    @settings(max_examples=100, deadline=None)
    def test_strategy_signal_generation_consistency(self, strategy_dataset):
        """
        Property 4: Strategy Signal Generation Consistency
        
        For any strategy applied to any valid dataset, the generated signals should 
        have the same length as the input dataset and contain only valid signal values.
        
        **Feature: warspite-financial-library, Property 4: Strategy Signal Generation Consistency**
        **Validates: Requirements 4.4**
        """
        strategy, dataset = strategy_dataset
        
        # Generate positions using the strategy
        positions = strategy.generate_positions(dataset)
        
        # Property assertions: Signal generation consistency
        
        # 1. Positions should be a numpy array
        assert isinstance(positions, np.ndarray), "Positions should be a numpy array"
        
        # 2. Positions array should have correct shape
        expected_shape = (len(dataset), len(dataset.symbols))
        assert positions.shape == expected_shape, \
            f"Positions shape {positions.shape} should match expected shape {expected_shape}"
        
        # 3. All position values should be within valid range [-1.0, 1.0]
        assert np.all(positions >= -1.0), "All positions should be >= -1.0 (no excessive short positions)"
        assert np.all(positions <= 1.0), "All positions should be <= 1.0 (no excessive long positions)"
        
        # 4. Positions should be finite (no NaN or infinity values)
        assert np.all(np.isfinite(positions)), "All positions should be finite (no NaN or infinity)"
        
        # 5. Positions array should have correct dtype
        assert positions.dtype in [np.float64, np.float32], "Positions should be floating point numbers"
        
        # 6. Strategy should be deterministic for deterministic strategies
        # RandomStrategy without seed is non-deterministic by design
        if not (isinstance(strategy, RandomStrategy) and strategy.get_parameters().get('seed') is None):
            positions2 = strategy.generate_positions(dataset)
            assert np.array_equal(positions, positions2), \
                "Deterministic strategy should produce same positions for same dataset"
        
        # 7. Strategy should handle the dataset without errors
        # (This is implicitly tested by successful execution above)
        
        # 8. Positions should be properly shaped (2D array)
        assert positions.ndim == 2, "Positions should be a 2D array (timestamps, symbols)"
        
        # 9. Strategy should preserve its parameters after execution
        original_params = strategy.get_parameters()
        strategy.generate_positions(dataset)  # Execute again
        current_params = strategy.get_parameters()
        assert original_params == current_params, \
            "Strategy parameters should not change during position generation"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=50, deadline=None)
    def test_random_strategy_specific_properties(self, dataset):
        """
        Test RandomStrategy specific properties.
        """
        # Test with fixed seed for reproducibility
        strategy = RandomStrategy(correct_percent=0.6, seed=42)
        
        positions1 = strategy.generate_positions(dataset)
        positions2 = strategy.generate_positions(dataset)
        
        # Random strategy with fixed seed should be deterministic
        assert np.array_equal(positions1, positions2), \
            "RandomStrategy with fixed seed should be deterministic"
        
        # Test without seed (should be different each time)
        strategy_no_seed = RandomStrategy(correct_percent=0.6, seed=None)
        positions3 = strategy_no_seed.generate_positions(dataset)
        positions4 = strategy_no_seed.generate_positions(dataset)
        
        # Without seed, results might be different (but not guaranteed)
        # We just check they're both valid
        assert np.all(positions3 >= -1.0) and np.all(positions3 <= 1.0), \
            "RandomStrategy without seed should produce valid positions"
        assert np.all(positions4 >= -1.0) and np.all(positions4 <= 1.0), \
            "RandomStrategy without seed should produce valid positions"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=50, deadline=None)
    def test_perfect_strategy_specific_properties(self, dataset):
        """
        Test PerfectStrategy specific properties.
        """
        # Ensure dataset is long enough for perfect strategy
        assume(len(dataset) > 1)
        
        lookahead = min(2, len(dataset) - 1)
        strategy = PerfectStrategy(lookahead_periods=lookahead)
        
        positions = strategy.generate_positions(dataset)
        
        # Perfect strategy specific properties
        
        # 1. Last few positions (where lookahead isn't possible) should be 0.0 for all symbols
        final_positions = positions[-lookahead:, :]
        assert np.all(final_positions == 0.0), \
            f"Final {lookahead} positions should be 0.0 (neutral) where lookahead isn't possible"
        
        # 2. Earlier positions should be based on future price movements for each symbol
        if len(dataset) > lookahead:
            # Check each symbol
            for symbol_idx, data_array in enumerate(dataset.data_arrays):
                # Extract close prices
                if data_array.ndim == 1:
                    prices = data_array
                else:
                    prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
                
                # Check that positions align with future price movements
                for i in range(len(positions) - lookahead):
                    current_price = prices[i]
                    future_price = prices[i + lookahead]
                    position = positions[i, symbol_idx]
                    
                    if future_price > current_price:
                        assert position == 1.0, f"Position should be 1.0 (long) when future price rises for symbol {symbol_idx}"
                    elif future_price < current_price:
                        assert position == -1.0, f"Position should be -1.0 (short) when future price falls for symbol {symbol_idx}"
                    else:
                        assert position == 0.0, f"Position should be 0.0 (neutral) when future price unchanged for symbol {symbol_idx}"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=50, deadline=None)
    def test_sma_strategy_specific_properties(self, dataset):
        """
        Test SMAStrategy specific properties.
        """
        # Ensure dataset is long enough for SMA calculation
        assume(len(dataset) >= 5)
        
        days = min(5, len(dataset))
        strategy = SMAStrategy(days=days)
        
        positions = strategy.generate_positions(dataset)
        
        # SMA strategy specific properties
        
        # 1. First (days-1) positions should be 0.0 (neutral) because SMA isn't available yet for all symbols
        initial_positions = positions[:days-1, :]
        assert np.all(initial_positions == 0.0), \
            f"First {days-1} positions should be 0.0 (neutral) where SMA isn't available yet"
        
        # 2. Later positions should be based on price vs SMA relationship for each symbol
        if len(dataset) > days:
            # Check each symbol
            for symbol_idx, data_array in enumerate(dataset.data_arrays):
                # Extract close prices
                if data_array.ndim == 1:
                    prices = data_array
                else:
                    prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
                
                # Calculate SMA manually for verification
                import pandas as pd
                price_series = pd.Series(prices)
                sma = price_series.rolling(window=days, min_periods=days).mean()
                
                # Check that positions align with price vs SMA relationship
                for i in range(days-1, len(positions)):
                    current_price = prices[i]
                    current_sma = sma.iloc[i]
                    position = positions[i, symbol_idx]
                    
                    if not pd.isna(current_sma):
                        if current_price > current_sma:
                            assert position == 1.0, f"Position should be 1.0 (long) when price > SMA for symbol {symbol_idx}"
                        elif current_price < current_sma:
                            assert position == -1.0, f"Position should be -1.0 (short) when price < SMA for symbol {symbol_idx}"
                        else:
                            assert position == 0.0, f"Position should be 0.0 (neutral) when price == SMA for symbol {symbol_idx}"
    
    @given(strategy_dataset=strategy_and_dataset())
    @settings(max_examples=30, deadline=None)
    def test_strategy_parameter_validation(self, strategy_dataset):
        """
        Test that strategies properly validate their parameters.
        """
        strategy, dataset = strategy_dataset
        
        # Get current parameters
        original_params = strategy.get_parameters()
        
        # Test parameter retrieval
        assert isinstance(original_params, dict), "get_parameters should return a dictionary"
        
        # Test parameter setting with valid values
        if isinstance(strategy, RandomStrategy):
            # Test valid correct_percent
            strategy.set_parameters(correct_percent=0.7)
            assert strategy.get_parameters()['correct_percent'] == 0.7, "Valid correct_percent should be set"
            
            # Test invalid correct_percent
            with pytest.raises(ValueError):
                strategy.set_parameters(correct_percent=1.5)  # Invalid: > 1.0
            
            with pytest.raises(ValueError):
                strategy.set_parameters(correct_percent=-0.1)  # Invalid: < 0.0
        
        elif isinstance(strategy, PerfectStrategy):
            # Test valid lookahead_periods
            strategy.set_parameters(lookahead_periods=3)
            assert strategy.get_parameters()['lookahead_periods'] == 3, "Valid lookahead_periods should be set"
            
            # Test invalid lookahead_periods
            with pytest.raises(ValueError):
                strategy.set_parameters(lookahead_periods=0)  # Invalid: must be > 0
            
            with pytest.raises(ValueError):
                strategy.set_parameters(lookahead_periods=-1)  # Invalid: must be > 0
        
        elif isinstance(strategy, SMAStrategy):
            # Test valid days
            strategy.set_parameters(days=10)
            assert strategy.get_parameters()['days'] == 10, "Valid days should be set"
            
            # Test invalid days
            with pytest.raises(ValueError):
                strategy.set_parameters(days=0)  # Invalid: must be > 0
            
            with pytest.raises(ValueError):
                strategy.set_parameters(days=-1)  # Invalid: must be > 0
        
        elif isinstance(strategy, (BuyAndHoldStrategy, ShortStrategy)):
            # BuyAndHold and Short strategies accept all parameters by default
            strategy.set_parameters(test_param='test_value')
            assert strategy.get_parameters()['test_param'] == 'test_value', \
                "BuyAndHold/Short strategies should accept arbitrary parameters"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=30, deadline=None)
    def test_strategy_error_handling(self, dataset):
        """
        Test that strategies properly handle error conditions.
        """
        strategy = RandomStrategy(correct_percent=0.5)
        
        # Test with None dataset
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            strategy.generate_positions(None)
        
        # Test with empty dataset (create minimal empty dataset)
        empty_timestamps = np.array([], dtype='datetime64[ns]')
        empty_data_arrays = [np.array([])]
        empty_symbols = ['TEST']
        
        with pytest.raises(ValueError):
            empty_dataset = WarspiteDataset(empty_data_arrays, empty_timestamps, empty_symbols)
            strategy.generate_positions(empty_dataset)
    
    @given(strategy_dataset=strategy_and_dataset())
    @settings(max_examples=30, deadline=None)
    def test_strategy_representation(self, strategy_dataset):
        """
        Test that strategies have proper string representations.
        """
        strategy, _ = strategy_dataset
        
        # Test __repr__ method
        repr_str = repr(strategy)
        assert isinstance(repr_str, str), "Strategy should have string representation"
        assert len(repr_str) > 0, "Strategy representation should not be empty"
        
        # Representation should include strategy type
        strategy_type = type(strategy).__name__
        assert strategy_type in repr_str, f"Representation should include strategy type '{strategy_type}'"
        
        # Representation should include key parameters
        if isinstance(strategy, RandomStrategy):
            assert 'correct_percent=' in repr_str, "RandomStrategy representation should include correct_percent"
        elif isinstance(strategy, PerfectStrategy):
            assert 'lookahead_periods=' in repr_str, "PerfectStrategy representation should include lookahead_periods"
        elif isinstance(strategy, SMAStrategy):
            assert 'days=' in repr_str, "SMAStrategy representation should include days"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=50, deadline=None)
    def test_buy_and_hold_strategy_specific_properties(self, dataset):
        """
        Test BuyAndHoldStrategy specific properties.
        """
        strategy = BuyAndHoldStrategy()
        
        positions = strategy.generate_positions(dataset)
        
        # BuyAndHold strategy specific properties
        
        # 1. All positions should be positive (long positions only)
        assert np.all(positions > 0), "All BuyAndHold positions should be positive (long positions)"
        
        # 2. All positions should be equal across time for each symbol
        for symbol_idx in range(len(dataset.symbols)):
            symbol_positions = positions[:, symbol_idx]
            assert np.all(symbol_positions == symbol_positions[0]), \
                f"BuyAndHold positions should be constant over time for symbol {symbol_idx}"
        
        # 3. Each symbol should get equal weight (1/n_symbols)
        n_symbols = len(dataset.symbols)
        expected_weight = 1.0 / n_symbols
        assert np.allclose(positions, expected_weight), \
            f"Each symbol should get equal weight {expected_weight}"
        
        # 4. Sum of positions across symbols should be 1.0 at each timestamp
        position_sums = np.sum(positions, axis=1)
        assert np.allclose(position_sums, 1.0), \
            "Sum of BuyAndHold positions across symbols should be 1.0"
    
    @given(dataset=valid_dataset_for_strategy())
    @settings(max_examples=50, deadline=None)
    def test_short_strategy_specific_properties(self, dataset):
        """
        Test ShortStrategy specific properties.
        """
        strategy = ShortStrategy()
        
        positions = strategy.generate_positions(dataset)
        
        # Short strategy specific properties
        
        # 1. All positions should be negative (short positions only)
        assert np.all(positions < 0), "All Short positions should be negative (short positions)"
        
        # 2. All positions should be equal across time for each symbol
        for symbol_idx in range(len(dataset.symbols)):
            symbol_positions = positions[:, symbol_idx]
            assert np.all(symbol_positions == symbol_positions[0]), \
                f"Short positions should be constant over time for symbol {symbol_idx}"
        
        # 3. Each symbol should get equal weight (-1/n_symbols)
        n_symbols = len(dataset.symbols)
        expected_weight = -1.0 / n_symbols
        assert np.allclose(positions, expected_weight), \
            f"Each symbol should get equal weight {expected_weight}"
        
        # 4. Sum of positions across symbols should be -1.0 at each timestamp
        position_sums = np.sum(positions, axis=1)
        assert np.allclose(position_sums, -1.0), \
            "Sum of Short positions across symbols should be -1.0"