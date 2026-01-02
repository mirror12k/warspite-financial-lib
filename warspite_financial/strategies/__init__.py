"""
Strategy module for warspite_financial library.

This module contains base interfaces and implementations for trading strategies.
"""

from .base import BaseStrategy
from .random import RandomStrategy
from .perfect import PerfectStrategy
from .sma import SMAStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .short import ShortStrategy

__all__ = ["BaseStrategy", "RandomStrategy", "PerfectStrategy", "SMAStrategy", "BuyAndHoldStrategy", "ShortStrategy"]