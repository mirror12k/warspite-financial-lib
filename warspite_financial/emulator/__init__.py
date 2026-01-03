"""
Emulator module for warspite_financial library.

This module contains trading emulation and position management functionality.
"""

from .emulator import WarspiteTradingEmulator
from .helpers import run_strategy_backtest

__all__ = ["WarspiteTradingEmulator", "run_strategy_backtest"]