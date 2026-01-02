"""
Buy and Hold strategy implementation.

This module contains the BuyAndHoldStrategy class that generates equal-weight
long positions across all symbols in the dataset.
"""

import numpy as np
from typing import Any
from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold trading strategy.
    
    This strategy generates equal-weight long positions across all symbols.
    Each symbol receives a position of 1/n where n is the number of symbols,
    maintaining these positions throughout the entire time period.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Buy and Hold strategy.
        
        Args:
            **kwargs: Additional strategy parameters (currently unused)
        """
        super().__init__(**kwargs)
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate equal-weight long positions for all symbols.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of position values where each symbol gets 1/n_symbols position
            Shape: (n_timestamps, n_symbols) - equal long positions for each symbol
            
        Raises:
            ValueError: If dataset is invalid
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        n_timestamps = len(dataset)
        n_symbols = len(dataset.symbols)
        
        if n_symbols == 0:
            raise ValueError("Dataset must contain at least one symbol")
        
        # Calculate equal weight for each symbol
        equal_weight = 1.0 / n_symbols
        
        # Initialize positions array with equal weights for all symbols at all times
        positions = np.full((n_timestamps, n_symbols), equal_weight, dtype=np.float64)
        
        return positions
    
    def _validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter is valid (accepts all parameters by default)
        """
        # Buy and Hold strategy has no specific parameters to validate
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return "BuyAndHoldStrategy()"