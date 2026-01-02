"""
Provider module for warspite_financial library.

This module contains base interfaces and implementations for financial data providers.
"""

from .base import BaseProvider, TradingProvider

# Optional provider imports (only if dependencies are available)
providers_available = ["BaseProvider", "TradingProvider"]

try:
    from .yfinance import YFinanceProvider
    providers_available.append("YFinanceProvider")
except ImportError:
    pass

try:
    from .oanda import OANDAProvider
    providers_available.append("OANDAProvider")
except ImportError:
    pass

# BrownianMotionProvider is always available (no external dependencies)
from .brownian_motion import BrownianMotionProvider
providers_available.append("BrownianMotionProvider")

__all__ = providers_available