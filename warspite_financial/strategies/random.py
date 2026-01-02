"""
Random strategy implementation.

This module contains the RandomStrategy class that generates random trading signals
with configurable success probability for baseline comparisons.
"""

import numpy as np
from typing import Any
from .base import BaseStrategy


class RandomStrategy(BaseStrategy):
    """
    Random trading strategy.
    
    This strategy generates random trading positions with a configurable
    success probability. It's useful for baseline comparisons and Monte Carlo analysis.
    """
    
    def __init__(self, correct_percent: float = 0.52, seed: int = None, **kwargs):
        """
        Initialize Random strategy.
        
        Args:
            correct_percent: Probability of generating "correct" signals (default 0.52)
            seed: Random seed for reproducible results (default None)
            **kwargs: Additional strategy parameters
            
        Raises:
            ValueError: If correct_percent is not between 0 and 1
        """
        if not isinstance(correct_percent, (int, float)) or not (0 <= correct_percent <= 1):
            raise ValueError("correct_percent must be a number between 0 and 1")
        
        super().__init__(correct_percent=correct_percent, seed=seed, **kwargs)
        
        # Create a local random number generator for reproducible results
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate random trading positions.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of random position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        # Reset random state for deterministic behavior if seed is set
        seed = self._parameters.get('seed')
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        
        correct_percent = self._parameters['correct_percent']
        n_timestamps = len(dataset)
        n_symbols = len(dataset.symbols)
        
        # Generate random positions for all symbols
        # Use the correct_percent to bias towards profitable positions
        # For simplicity, we'll generate random values between -1 and 1
        # and then apply the success probability
        
        # Generate base random positions
        positions = self._rng.uniform(-1.0, 1.0, (n_timestamps, n_symbols))
        
        # Apply success probability by potentially flipping some positions
        # This is a simplified interpretation - in practice, "correct" would
        # depend on future price movements which we don't have access to here
        flip_probability = 1.0 - correct_percent
        flip_mask = self._rng.random((n_timestamps, n_symbols)) < flip_probability
        
        # Flip positions for the "incorrect" signals
        positions[flip_mask] *= -1
        
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
        if name == 'correct_percent':
            return isinstance(value, (int, float)) and 0 <= value <= 1
        elif name == 'seed':
            return value is None or isinstance(value, int)
        
        # Accept other parameters by default
        return True
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters and reset random number generator if seed changes.
        
        Args:
            **kwargs: Parameter names and new values
            
        Raises:
            ValueError: If parameter names or values are invalid
        """
        # Call parent implementation
        super().set_parameters(**kwargs)
        
        # Reset random number generator if seed was updated
        if 'seed' in kwargs:
            seed = kwargs['seed']
            self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"RandomStrategy(correct_percent={self._parameters['correct_percent']}, seed={self._parameters.get('seed', None)})"