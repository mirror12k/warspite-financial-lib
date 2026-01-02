"""
Base strategy interface for warspite_financial library.

This module defines the abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This class defines the interface that all trading strategies must implement
    to generate trading positions based on market data.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the strategy with parameters.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        self._parameters = kwargs
    
    @abstractmethod
    def generate_positions(self, dataset) -> np.ndarray:
        """
        Generate trading positions based on dataset.
        
        Args:
            dataset: WarspiteDataset containing market data with 2-level MultiIndex structure
            
        Returns:
            numpy array of position values from -1.0 (full short) to 1.0 (full long)
            Shape: (n_timestamps, n_symbols) - positions for each symbol at each timestamp
            
        Raises:
            ValueError: If dataset is invalid or insufficient for strategy
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return self._parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters.
        
        Args:
            **kwargs: Parameter names and new values
            
        Raises:
            ValueError: If parameter names or values are invalid
        """
        for key, value in kwargs.items():
            if self._validate_parameter(key, value):
                self._parameters[key] = value
            else:
                raise ValueError(f"Invalid parameter: {key}={value}")
    
    def _validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate a parameter name and value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter is valid, False otherwise
        """
        # Default implementation accepts all parameters
        # Subclasses should override for specific validation
        return True