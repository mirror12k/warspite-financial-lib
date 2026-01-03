"""
Contrarian strategy implementation.

This module contains the ContrarianStrategy class that generates trading signals
based on contrarian investment principles - buying when others are selling and vice versa.
"""

import numpy as np
import pandas as pd
from typing import Any
from .base import BaseStrategy


class ContrarianStrategy(BaseStrategy):
    """
    Contrarian trading strategy.
    
    This strategy generates trading positions based on contrarian principles:
    - Takes positions opposite to recent price momentum
    - Long position when price has declined significantly over lookback period
    - Short position when price has increased significantly over lookback period
    - Uses price change percentiles to determine "significant" moves
    
    The strategy assumes that extreme price movements tend to reverse,
    making it profitable to trade against the prevailing trend.
    """
    
    def __init__(self, lookback_period: int = 10, threshold_percentile: float = 0.8, **kwargs):
        """
        Initialize Contrarian strategy.
        
        Args:
            lookback_period: Number of periods to look back for momentum calculation (default 10)
            threshold_percentile: Percentile threshold for significant moves (default 0.8)
                                 Values above this percentile trigger short positions,
                                 values below (1 - threshold_percentile) trigger long positions
            **kwargs: Additional strategy parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(lookback_period, int) or lookback_period < 2:
            raise ValueError("lookback_period must be an integer >= 2")
        
        if not isinstance(threshold_percentile, (int, float)) or not (0.5 < threshold_percentile < 1.0):
            raise ValueError("threshold_percentile must be between 0.5 and 1.0")
        
        super().__init__(lookback_period=lookback_period, threshold_percentile=threshold_percentile, **kwargs)
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate contrarian trading positions based on recent price momentum.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid or insufficient for contrarian calculation
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        lookback_period = self._parameters['lookback_period']
        threshold_percentile = self._parameters['threshold_percentile']
        
        if len(dataset) < lookback_period + 1:
            raise ValueError(f"Dataset length ({len(dataset)}) must be at least {lookback_period + 1} for contrarian calculation")
        
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
            
            # Calculate price changes over lookback period
            price_series = pd.Series(prices)
            
            # Calculate rolling returns over lookback period
            returns = price_series.pct_change(periods=lookback_period)
            
            # Calculate rolling percentiles for the entire series to establish thresholds
            # We need a longer window to calculate meaningful percentiles
            percentile_window = min(max(lookback_period * 5, 50), len(prices))
            
            # Generate positions based on contrarian logic
            for i in range(lookback_period, n_timestamps):
                current_return = returns.iloc[i]
                
                if pd.isna(current_return):
                    # Return not available, neutral position
                    positions[i, symbol_idx] = 0.0
                    continue
                
                # Calculate percentiles using a rolling window of historical returns
                start_idx = max(0, i - percentile_window + 1)
                end_idx = i + 1
                historical_returns = returns.iloc[start_idx:end_idx].dropna()
                
                if len(historical_returns) < lookback_period:
                    # Not enough historical data, neutral position
                    positions[i, symbol_idx] = 0.0
                    continue
                
                # Calculate percentile thresholds
                upper_threshold = historical_returns.quantile(threshold_percentile)
                lower_threshold = historical_returns.quantile(1.0 - threshold_percentile)
                
                # Apply contrarian logic
                if current_return >= upper_threshold:
                    # Price has increased significantly, take short position (expect reversal)
                    positions[i, symbol_idx] = -1.0
                elif current_return <= lower_threshold:
                    # Price has decreased significantly, take long position (expect reversal)
                    positions[i, symbol_idx] = 1.0
                else:
                    # Price change not significant enough, neutral position
                    positions[i, symbol_idx] = 0.0
        
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
        if name == 'lookback_period':
            return isinstance(value, int) and value >= 2
        elif name == 'threshold_percentile':
            return isinstance(value, (int, float)) and 0.5 < value < 1.0
        
        # Accept other parameters by default
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"ContrarianStrategy(lookback_period={self._parameters['lookback_period']}, threshold_percentile={self._parameters['threshold_percentile']})"