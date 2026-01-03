"""
Bollinger Bands strategy implementation.

This module contains the BollingerBandsStrategy class that generates trading signals
based on Bollinger Bands technical indicator.
"""

import numpy as np
import pandas as pd
from typing import Any
from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands trading strategy.
    
    This strategy generates trading positions based on Bollinger Bands:
    - Long position when price touches or goes below the lower band
    - Short position when price touches or goes above the upper band
    - Neutral position when price is within the bands
    
    Bollinger Bands consist of:
    - Middle band: Simple Moving Average (SMA)
    - Upper band: SMA + (standard deviation * multiplier)
    - Lower band: SMA - (standard deviation * multiplier)
    """
    
    def __init__(self, period: int = 20, std_multiplier: float = 2.0, **kwargs):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            period: Number of periods for SMA and standard deviation calculation (default 20)
            std_multiplier: Standard deviation multiplier for band calculation (default 2.0)
            **kwargs: Additional strategy parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(period, int) or period < 2:
            raise ValueError("period must be an integer >= 2")
        
        if not isinstance(std_multiplier, (int, float)) or std_multiplier <= 0:
            raise ValueError("std_multiplier must be a positive number")
        
        super().__init__(period=period, std_multiplier=std_multiplier, **kwargs)
    
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate trading positions based on Bollinger Bands.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid or insufficient for Bollinger Bands calculation
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        period = self._parameters['period']
        std_multiplier = self._parameters['std_multiplier']
        
        if len(dataset) < period:
            raise ValueError(f"Dataset length ({len(dataset)}) must be at least {period} for Bollinger Bands calculation")
        
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
            
            # Calculate Bollinger Bands using pandas for efficiency
            price_series = pd.Series(prices)
            
            # Calculate middle band (SMA)
            middle_band = price_series.rolling(window=period, min_periods=period).mean()
            
            # Calculate standard deviation
            rolling_std = price_series.rolling(window=period, min_periods=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_multiplier)
            lower_band = middle_band - (rolling_std * std_multiplier)
            
            # Generate positions based on price vs Bollinger Bands
            for i in range(period - 1, n_timestamps):
                current_price = prices[i]
                current_upper = upper_band.iloc[i]
                current_lower = lower_band.iloc[i]
                current_middle = middle_band.iloc[i]
                
                if pd.isna(current_upper) or pd.isna(current_lower) or pd.isna(current_middle):
                    # Bands not available yet, neutral position
                    positions[i, symbol_idx] = 0.0
                elif current_price <= current_lower:
                    # Price at or below lower band, take long position (oversold)
                    positions[i, symbol_idx] = 1.0
                elif current_price >= current_upper:
                    # Price at or above upper band, take short position (overbought)
                    positions[i, symbol_idx] = -1.0
                else:
                    # Price within bands, neutral position
                    positions[i, symbol_idx] = 0.0
            
            # For the first (period-1) periods, positions remain 0.0 (already initialized)
        
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
        if name == 'period':
            return isinstance(value, int) and value >= 2
        elif name == 'std_multiplier':
            return isinstance(value, (int, float)) and value > 0
        
        # Accept other parameters by default
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"BollingerBandsStrategy(period={self._parameters['period']}, std_multiplier={self._parameters['std_multiplier']})"