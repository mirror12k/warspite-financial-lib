"""
warspite_financial - A comprehensive Python library for financial data processing, 
trading strategies, and emulation.

This library provides a modular framework for:
- Loading financial data from various providers
- Applying trading strategies 
- Simulating trading scenarios
- Executing real trades through supported trading providers
"""

__version__ = "0.1.1"
__author__ = "warspite_financial"

# Import all submodules and dynamically export their contents
from . import providers
from . import strategies  
from . import datasets
from . import emulator
from . import forecasting
from . import cli
from . import visualization
from . import utils
from . import examples

# Build __all__ dynamically from submodule exports
__all__ = []

# Add exports from each submodule
for module in [providers, strategies, datasets, emulator, forecasting, cli, visualization, utils, examples]:
    if hasattr(module, '__all__'):
        __all__.extend(module.__all__)
        # Import each exported item into this namespace
        for item_name in module.__all__:
            if hasattr(module, item_name):
                globals()[item_name] = getattr(module, item_name)

# Remove duplicates while preserving order
__all__ = list(dict.fromkeys(__all__))