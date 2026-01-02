"""
Perfect strategy implementation.

This module contains the PerfectStrategy class that generates optimal trading signals
based on future price knowledge for theoretical maximum performance benchmarking.
"""

import numpy as np
from typing import Any
from .base import BaseStrategy


class PerfectStrategy(BaseStrategy):
    """
    Perfect trading strategy.
    
    This strategy generates optimal positions based on future price knowledge.
    It implements look-ahead bias for comparison purposes and represents the
    theoretical maximum performance that could be achieved with perfect foresight.
    """
    
    def __init__(self, lookahead_periods: int = 1, **kwargs):
        """
        Initialize Perfect strategy.
        
        Args:
            lookahead_periods: Number of periods to look ahead for optimal positioning (default 1)
            **kwargs: Additional strategy parameters
            
        Raises:
            ValueError: If lookahead_periods is not a positive integer
        """
        if not isinstance(lookahead_periods, int) or lookahead_periods < 1:
            raise ValueError("lookahead_periods must be a positive integer")
        
        super().__init__(lookahead_periods=lookahead_periods, **kwargs)
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate optimal trading positions based on future price knowledge.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of optimal position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid or insufficient for strategy
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        lookahead_periods = self._parameters['lookahead_periods']
        
        if len(dataset) <= lookahead_periods:
            raise ValueError(f"Dataset length ({len(dataset)}) must be greater than lookahead_periods ({lookahead_periods})")
        
        n_timestamps = len(dataset)
        n_symbols = len(dataset.symbols)
        
        # Initialize positions array for all symbols
        positions = np.zeros((n_timestamps, n_symbols))
        
        # Process each symbol
        for symbol_idx, data_array in enumerate(dataset.data_arrays):
            # Extract close prices
            if data_array.ndim == 1:
                # Single price series
                prices = data_array
            else:
                # Multi-column data (OHLCV), use Close prices (column 3)
                if data_array.shape[1] > 3:
                    prices = data_array[:, 3]  # Close prices
                else:
                    prices = data_array[:, -1]  # Last column if less than 4 columns
            
            # Generate optimal positions based on future price movements
            for i in range(n_timestamps - lookahead_periods):
                current_price = prices[i]
                future_price = prices[i + lookahead_periods]
                
                # Calculate the optimal position based on future price movement
                price_change = future_price - current_price
                
                if price_change > 0:
                    # Price will go up, take long position
                    positions[i, symbol_idx] = 1.0
                elif price_change < 0:
                    # Price will go down, take short position
                    positions[i, symbol_idx] = -1.0
                else:
                    # No price change, neutral position
                    positions[i, symbol_idx] = 0.0
            
            # For the last few periods where we can't look ahead, positions remain 0.0 (already initialized)
        
        return positions
    
    def _validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter is valid, False otherwise
        """
        if name == 'lookahead_periods':
            return isinstance(value, int) and value > 0
        
        # Accept other parameters by default
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"PerfectStrategy(lookahead_periods={self._parameters['lookahead_periods']})"