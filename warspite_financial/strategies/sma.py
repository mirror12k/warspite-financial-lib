"""
Simple Moving Average (SMA) strategy implementation.

This module contains the SMAStrategy class that generates trading signals
based on price vs Simple Moving Average crossover patterns.
"""

import numpy as np
import pandas as pd
from typing import Any
from .base import BaseStrategy


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average trading strategy.
    
    This strategy generates trading positions based on the relationship between
    current price and its Simple Moving Average (SMA). When price is above SMA,
    it generates long positions; when below, it generates short positions.
    """
    
    def __init__(self, days: int = 30, period: int = None, **kwargs):
        """
        Initialize SMA strategy.
        
        Args:
            days: Number of days for SMA calculation (default 30)
            period: Alias for days parameter (for backward compatibility)
            **kwargs: Additional strategy parameters
            
        Raises:
            ValueError: If days is not a positive integer
        """
        # Handle period alias
        if period is not None:
            days = period
            
        if not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")
        
        super().__init__(days=days, **kwargs)
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate trading positions based on price vs SMA crossover.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid or insufficient for SMA calculation
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        days = self._parameters['days']
        
        if len(dataset) < days:
            raise ValueError(f"Dataset length ({len(dataset)}) must be at least {days} for SMA calculation")
        
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
            
            # Calculate Simple Moving Average using pandas for efficiency
            price_series = pd.Series(prices)
            sma = price_series.rolling(window=days, min_periods=days).mean()
            
            # Generate positions based on price vs SMA relationship
            for i in range(days - 1, n_timestamps):
                current_price = prices[i]
                current_sma = sma.iloc[i]
                
                if pd.isna(current_sma):
                    # SMA not available yet, neutral position
                    positions[i, symbol_idx] = 0.0
                elif current_price > current_sma:
                    # Price above SMA, take long position
                    positions[i, symbol_idx] = 1.0
                elif current_price < current_sma:
                    # Price below SMA, take short position
                    positions[i, symbol_idx] = -1.0
                else:
                    # Price equals SMA, neutral position
                    positions[i, symbol_idx] = 0.0
            
            # For the first (days-1) periods, positions remain 0.0 (already initialized)
        
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
        if name == 'days':
            return isinstance(value, int) and value > 0
        
        # Accept other parameters by default
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"SMAStrategy(days={self._parameters['days']})"