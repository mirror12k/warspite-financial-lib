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