"""
Unit tests for strategy implementations.

This module contains unit tests for the trading strategies in the warspite_financial library.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.strategies.buy_and_hold import BuyAndHoldStrategy
from warspite_financial.strategies.short import ShortStrategy
from warspite_financial.strategies.bollinger_bands import BollingerBandsStrategy
from warspite_financial.strategies.contrarian import ContrarianStrategy


class TestBuyAndHoldStrategy:
    """Unit tests for BuyAndHoldStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test dataset with multiple symbols
        timestamps = np.array([
            datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)
        ], dtype='datetime64[ns]')
        
        # Create data arrays for 3 symbols
        data_arrays = [
            np.array([100.0, 101.0, 102.0, 103.0, 104.0]),  # Symbol 1
            np.array([50.0, 51.0, 52.0, 53.0, 54.0]),       # Symbol 2
            np.array([200.0, 201.0, 202.0, 203.0, 204.0])   # Symbol 3
        ]
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        self.strategy = BuyAndHoldStrategy()
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BuyAndHoldStrategy()
        assert isinstance(strategy, BuyAndHoldStrategy)
        assert strategy.get_parameters() == {}
    
    def test_generate_positions_basic(self):
        """Test basic position generation."""
        positions = self.strategy.generate_positions(self.dataset)
        
        # Check shape
        assert positions.shape == (5, 3)  # 5 timestamps, 3 symbols
        
        # Check that all positions are equal weight (1/3 for 3 symbols)
        expected_weight = 1.0 / 3.0
        assert np.allclose(positions, expected_weight)
        
        # Check data type
        assert positions.dtype == np.float64
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None dataset
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            self.strategy.generate_positions(None)
    
    def test_string_representation(self):
        """Test string representation."""
        repr_str = repr(self.strategy)
        assert repr_str == "BuyAndHoldStrategy()"
        assert isinstance(repr_str, str)


class TestShortStrategy:
    """Unit tests for ShortStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test dataset with multiple symbols
        timestamps = np.array([
            datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)
        ], dtype='datetime64[ns]')
        
        # Create data arrays for 3 symbols
        data_arrays = [
            np.array([100.0, 101.0, 102.0, 103.0, 104.0]),  # Symbol 1
            np.array([50.0, 51.0, 52.0, 53.0, 54.0]),       # Symbol 2
            np.array([200.0, 201.0, 202.0, 203.0, 204.0])   # Symbol 3
        ]
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        self.strategy = ShortStrategy()
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ShortStrategy()
        assert isinstance(strategy, ShortStrategy)
        assert strategy.get_parameters() == {}
    
    def test_generate_positions_basic(self):
        """Test basic position generation."""
        positions = self.strategy.generate_positions(self.dataset)
        
        # Check shape
        assert positions.shape == (5, 3)  # 5 timestamps, 3 symbols
        
        # Check that all positions are equal short weight (-1/3 for 3 symbols)
        expected_weight = -1.0 / 3.0
        assert np.allclose(positions, expected_weight)
        
        # Check data type
        assert positions.dtype == np.float64
        
        # Verify all positions are negative (short positions)
        assert np.all(positions < 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None dataset
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            self.strategy.generate_positions(None)
    
    def test_string_representation(self):
        """Test string representation."""
        repr_str = repr(self.strategy)
        assert repr_str == "ShortStrategy()"
        assert isinstance(repr_str, str)


class TestBollingerBandsStrategy:
    """Unit tests for BollingerBandsStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test dataset with trending price data
        timestamps = np.array([
            datetime(2023, 1, 1) + timedelta(days=i) for i in range(25)
        ], dtype='datetime64[ns]')
        
        # Create price data that will trigger Bollinger Band signals
        # Start with base price and add some volatility
        base_prices = np.linspace(100, 120, 25)
        volatility = np.sin(np.linspace(0, 4*np.pi, 25)) * 5
        prices = base_prices + volatility
        
        # Add extreme values to test band triggers
        prices[20] = 85   # Below lower band
        prices[22] = 135  # Above upper band
        
        data_arrays = [prices]
        symbols = ['TEST']
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        self.strategy = BollingerBandsStrategy(period=10, std_multiplier=2.0)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BollingerBandsStrategy()
        assert isinstance(strategy, BollingerBandsStrategy)
        params = strategy.get_parameters()
        assert params['period'] == 20  # default
        assert params['std_multiplier'] == 2.0  # default
        
        # Test custom parameters
        custom_strategy = BollingerBandsStrategy(period=15, std_multiplier=1.5)
        custom_params = custom_strategy.get_parameters()
        assert custom_params['period'] == 15
        assert custom_params['std_multiplier'] == 1.5
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid period
        with pytest.raises(ValueError, match="period must be an integer >= 2"):
            BollingerBandsStrategy(period=1)
        
        with pytest.raises(ValueError, match="period must be an integer >= 2"):
            BollingerBandsStrategy(period=-5)
        
        # Test invalid std_multiplier
        with pytest.raises(ValueError, match="std_multiplier must be a positive number"):
            BollingerBandsStrategy(std_multiplier=0)
        
        with pytest.raises(ValueError, match="std_multiplier must be a positive number"):
            BollingerBandsStrategy(std_multiplier=-1.5)
    
    def test_generate_positions_basic(self):
        """Test basic position generation."""
        positions = self.strategy.generate_positions(self.dataset)
        
        # Check shape
        assert positions.shape == (25, 1)  # 25 timestamps, 1 symbol
        
        # Check that early positions are 0 (not enough data for bands)
        assert np.all(positions[:9, 0] == 0.0)  # First 9 positions should be 0
        
        # Check that we have some non-zero positions after the initial period
        assert np.any(positions[10:, 0] != 0.0)
        
        # Check position values are within valid range
        assert np.all(positions >= -1.0)
        assert np.all(positions <= 1.0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None dataset
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            self.strategy.generate_positions(None)
        
        # Test with insufficient data
        short_timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        short_data = [np.array([100.0])]
        short_dataset = WarspiteDataset(short_data, short_timestamps, ['TEST'])
        
        with pytest.raises(ValueError, match="Dataset length .* must be at least .* for Bollinger Bands calculation"):
            self.strategy.generate_positions(short_dataset)
    
    def test_string_representation(self):
        """Test string representation."""
        repr_str = repr(self.strategy)
        assert "BollingerBandsStrategy" in repr_str
        assert "period=10" in repr_str
        assert "std_multiplier=2.0" in repr_str


class TestContrarianStrategy:
    """Unit tests for ContrarianStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test dataset with volatile price data
        timestamps = np.array([
            datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)
        ], dtype='datetime64[ns]')
        
        # Create price data with clear trends and reversals
        prices = np.array([
            100, 102, 104, 106, 108, 110, 112, 114, 116, 118,  # Strong uptrend
            120, 118, 116, 114, 112, 110, 108, 106, 104, 102,  # Strong downtrend
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109   # Moderate uptrend
        ], dtype=float)
        
        data_arrays = [prices]
        symbols = ['TEST']
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        self.strategy = ContrarianStrategy(lookback_period=5, threshold_percentile=0.8)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ContrarianStrategy()
        assert isinstance(strategy, ContrarianStrategy)
        params = strategy.get_parameters()
        assert params['lookback_period'] == 10  # default
        assert params['threshold_percentile'] == 0.8  # default
        
        # Test custom parameters
        custom_strategy = ContrarianStrategy(lookback_period=7, threshold_percentile=0.75)
        custom_params = custom_strategy.get_parameters()
        assert custom_params['lookback_period'] == 7
        assert custom_params['threshold_percentile'] == 0.75
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid lookback_period
        with pytest.raises(ValueError, match="lookback_period must be an integer >= 2"):
            ContrarianStrategy(lookback_period=1)
        
        with pytest.raises(ValueError, match="lookback_period must be an integer >= 2"):
            ContrarianStrategy(lookback_period=-3)
        
        # Test invalid threshold_percentile
        with pytest.raises(ValueError, match="threshold_percentile must be between 0.5 and 1.0"):
            ContrarianStrategy(threshold_percentile=0.5)
        
        with pytest.raises(ValueError, match="threshold_percentile must be between 0.5 and 1.0"):
            ContrarianStrategy(threshold_percentile=1.0)
        
        with pytest.raises(ValueError, match="threshold_percentile must be between 0.5 and 1.0"):
            ContrarianStrategy(threshold_percentile=0.3)
    
    def test_generate_positions_basic(self):
        """Test basic position generation."""
        positions = self.strategy.generate_positions(self.dataset)
        
        # Check shape
        assert positions.shape == (30, 1)  # 30 timestamps, 1 symbol
        
        # Check that early positions are 0 (not enough data)
        assert np.all(positions[:5, 0] == 0.0)  # First 5 positions should be 0
        
        # Check that we have some non-zero positions after the initial period
        assert np.any(positions[10:, 0] != 0.0)
        
        # Check position values are within valid range
        assert np.all(positions >= -1.0)
        assert np.all(positions <= 1.0)
        
        # Check that positions are discrete (-1, 0, or 1)
        unique_positions = np.unique(positions)
        assert all(pos in [-1.0, 0.0, 1.0] for pos in unique_positions)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None dataset
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            self.strategy.generate_positions(None)
        
        # Test with insufficient data
        short_timestamps = np.array([datetime(2023, 1, 1), datetime(2023, 1, 2)], dtype='datetime64[ns]')
        short_data = [np.array([100.0, 101.0])]
        short_dataset = WarspiteDataset(short_data, short_timestamps, ['TEST'])
        
        with pytest.raises(ValueError, match="Dataset length .* must be at least .* for contrarian calculation"):
            self.strategy.generate_positions(short_dataset)
    
    def test_string_representation(self):
        """Test string representation."""
        repr_str = repr(self.strategy)
        assert "ContrarianStrategy" in repr_str
        assert "lookback_period=5" in repr_str
        assert "threshold_percentile=0.8" in repr_str