"""
Helper functions for dataset operations.

This module provides convenience functions for common dataset creation patterns.
"""

from .dataset import WarspiteDataset
from ..utils.exceptions import WarspiteError


def create_dataset_from_provider(provider, symbols, start_date, end_date, interval='1d'):
    """
    Convenience function to create a dataset from a provider.
    
    Args:
        provider: Instance of BaseProvider
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data  
        interval: Data interval (default '1d')
        
    Returns:
        WarspiteDataset: Created dataset
        
    Example:
        >>> from warspite_financial import BrownianMotionProvider
        >>> from warspite_financial.datasets.helpers import create_dataset_from_provider
        >>> from datetime import datetime, timedelta
        >>> provider = BrownianMotionProvider()
        >>> end_date = datetime.now()
        >>> start_date = end_date - timedelta(days=30)
        >>> dataset = create_dataset_from_provider(provider, ['BM-AAPL'], start_date, end_date)
    """
    try:
        return WarspiteDataset.from_provider(provider, symbols, start_date, end_date, interval)
    except Exception as e:
        raise WarspiteError(f"Failed to create dataset from provider: {str(e)}") from e


__all__ = ['create_dataset_from_provider']