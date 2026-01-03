"""
Helper functions for emulator operations.

This module provides convenience functions for common backtesting patterns.
"""

from .emulator import WarspiteTradingEmulator
from ..utils.exceptions import EmulatorError


def run_strategy_backtest(dataset, strategy, initial_capital=10000, trading_fee=0.0, spread=0.0001):
    """
    Convenience function to run a complete strategy backtest.
    
    Args:
        dataset: WarspiteDataset to test against
        strategy: BaseStrategy instance
        initial_capital: Starting capital (default 10000)
        trading_fee: Trading fee per transaction (default 0.0)
        spread: Bid-ask spread (default 0.0001)
        
    Returns:
        EmulationResult: Complete backtest results
        
    Example:
        >>> from warspite_financial import SMAStrategy
        >>> from warspite_financial.emulator.helpers import run_strategy_backtest
        >>> strategy = SMAStrategy(period=20)
        >>> result = run_strategy_backtest(dataset, strategy)
        >>> print(f"Final portfolio value: {result.final_portfolio_value}")
    """
    try:
        # Generate strategy positions and add them to the dataset for visualization
        positions = strategy.generate_positions(dataset)
        dataset.add_strategy_results(positions)
        
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=trading_fee,
            spread=spread
        )
        emulator.add_strategy(strategy)
        return emulator.run_to_completion()
    except Exception as e:
        raise EmulatorError(f"Failed to run strategy backtest: {str(e)}") from e


__all__ = ['run_strategy_backtest']