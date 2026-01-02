"""
warspite_financial - A comprehensive Python library for financial data processing, 
trading strategies, and emulation.

This library provides a modular framework for:
- Loading financial data from various providers
- Applying trading strategies 
- Simulating trading scenarios
- Executing real trades through supported trading providers
"""

__version__ = "0.1.0"
__author__ = "warspite_financial"

# Core interfaces
from .providers import BaseProvider, TradingProvider, BrownianMotionProvider
from .strategies import BaseStrategy, RandomStrategy, PerfectStrategy, SMAStrategy
from .datasets import WarspiteDataset, WarspiteDatasetSerializer
from .emulator import WarspiteTradingEmulator
from .forecasting import WarspiteHeuristicForecaster
from .cli import WarspiteCLI
from .visualization import (
    WarspiteDatasetRenderer, 
    MatplotlibRenderer, 
    ASCIIRenderer,
    PDFRenderer,
    CSVRenderer
)

# Error handling
from .utils.exceptions import (
    WarspiteError,
    ProviderError,
    DatasetError,
    StrategyError,
    EmulatorError,
    TradingError,
    VisualizationError,
    SerializationError
)

# Examples
from .examples import (
    basic_backtest_example,
    multi_strategy_comparison,
    live_trading_example,
    visualization_example
)

# Optional provider imports (graceful degradation if dependencies missing)
_optional_providers = []
try:
    from .providers import YFinanceProvider
    _optional_providers.append("YFinanceProvider")
except ImportError:
    pass

try:
    from .providers import OANDAProvider
    _optional_providers.append("OANDAProvider")
except ImportError:
    pass

# Core exports (always available)
__all__ = [
    # Core interfaces
    "BaseProvider",
    "TradingProvider", 
    "BaseStrategy",
    "WarspiteDataset",
    "WarspiteDatasetSerializer",
    "WarspiteTradingEmulator",
    "WarspiteHeuristicForecaster",
    "WarspiteCLI",
    
    # Strategies
    "RandomStrategy",
    "PerfectStrategy", 
    "SMAStrategy",
    
    # Providers (always available)
    "BrownianMotionProvider",
    
    # Visualization
    "WarspiteDatasetRenderer",
    "MatplotlibRenderer", 
    "ASCIIRenderer",
    "PDFRenderer",
    "CSVRenderer",
    
    # Error handling
    "WarspiteError",
    "ProviderError",
    "DatasetError",
    "StrategyError",
    "EmulatorError",
    "TradingError",
    "VisualizationError",
    "SerializationError",
    
    # Examples
    "basic_backtest_example",
    "multi_strategy_comparison",
    "live_trading_example",
    "visualization_example"
] + _optional_providers

# Convenience functions for end-to-end workflows
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
        >>> from warspite_financial import BrownianMotionProvider, create_dataset_from_provider
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
        >>> from warspite_financial import SMAStrategy, run_strategy_backtest
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

def create_visualization(dataset, renderer_type='matplotlib', **kwargs):
    """
    Convenience function to create visualizations.
    
    Args:
        dataset: WarspiteDataset to visualize
        renderer_type: Type of renderer ('matplotlib', 'ascii', 'pdf', 'csv')
        **kwargs: Additional arguments for the renderer
        
    Returns:
        Rendered output (varies by renderer type)
        
    Example:
        >>> from warspite_financial import create_visualization
        >>> chart = create_visualization(dataset, 'matplotlib', title='Stock Prices')
        >>> chart.show()
    """
    renderer_map = {
        'matplotlib': MatplotlibRenderer,
        'ascii': ASCIIRenderer,
        'pdf': PDFRenderer,
        'csv': CSVRenderer
    }
    
    if renderer_type not in renderer_map:
        raise VisualizationError(f"Unknown renderer type: {renderer_type}. Available: {list(renderer_map.keys())}")
    
    try:
        renderer_class = renderer_map[renderer_type]
        renderer = renderer_class(dataset)
        return renderer.render(**kwargs)
    except Exception as e:
        raise VisualizationError(f"Failed to create visualization: {str(e)}") from e

# Add convenience functions to exports
__all__.extend(['create_dataset_from_provider', 'run_strategy_backtest', 'create_visualization'])