"""
Command line interface for warspite_financial library.

This module provides CLI functionality for interactive trading and dataset visualization.
"""

from .cli import WarspiteCLI
from .main import main

__all__ = ['WarspiteCLI', 'main']