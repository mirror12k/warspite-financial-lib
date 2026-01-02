"""
Exception classes for warspite_financial library.

This module defines custom exceptions for different components of the library.
"""


class WarspiteError(Exception):
    """Base exception class for all warspite_financial errors."""
    pass


class ProviderError(WarspiteError):
    """Exception raised for provider-related errors."""
    pass


class DatasetError(WarspiteError):
    """Exception raised for dataset-related errors."""
    pass


class StrategyError(WarspiteError):
    """Exception raised for strategy-related errors."""
    pass


class EmulatorError(WarspiteError):
    """Exception raised for emulator-related errors."""
    pass


class TradingError(WarspiteError):
    """Exception raised for trading-related errors."""
    pass


class VisualizationError(WarspiteError):
    """Exception raised for visualization-related errors."""
    pass


class SerializationError(WarspiteError):
    """Exception raised for serialization-related errors."""
    pass