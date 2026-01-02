"""
End-to-end workflow examples for warspite_financial library.

This module demonstrates complete workflows from data loading to visualization.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from ..providers import BrownianMotionProvider
from ..strategies import SMAStrategy, RandomStrategy, PerfectStrategy
from ..datasets import WarspiteDataset
from ..emulator import WarspiteTradingEmulator
from ..visualization import MatplotlibRenderer, ASCIIRenderer
from ..utils.exceptions import WarspiteError, ProviderError, EmulatorError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_backtest_example(
    symbols: List[str] = None,
    days: int = 30,
    initial_capital: float = 10000,
    sma_period: int = 20
) -> Dict[str, Any]:
    """
    Complete basic backtesting workflow example.
    
    Args:
        symbols: List of symbols to test (default ['AAPL'])
        days: Number of days of historical data (default 30)
        initial_capital: Starting capital (default 10000)
        sma_period: SMA strategy period (default 20)
        
    Returns:
        Dict containing results and visualizations
        
    Raises:
        WarspiteError: If any step in the workflow fails
    """
    if symbols is None:
        symbols = ['AAPL']
    
    try:
        logger.info("Starting basic backtest example")
        
        # Step 1: Create provider and load data
        logger.info("Loading data from provider")
        provider = BrownianMotionProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dataset = WarspiteDataset.from_provider(
            provider=provider,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        logger.info(f"Loaded dataset with {len(dataset.timestamps)} data points")
        
        # Step 2: Create and configure strategy
        logger.info(f"Creating SMA strategy with period {sma_period}")
        strategy = SMAStrategy(period=sma_period)
        
        # Step 3: Run backtest
        logger.info("Running backtest simulation")
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=initial_capital,
            trading_fee=0.001,  # 0.1% fee
            spread=0.0001
        )
        emulator.add_strategy(strategy)
        result = emulator.run_to_completion()
        
        # Step 4: Create visualizations
        logger.info("Generating visualizations")
        matplotlib_renderer = MatplotlibRenderer(dataset)
        chart = matplotlib_renderer.render(
            title=f"Backtest Results - {', '.join(symbols)}",
            show_strategy_signals=True,
            show_trades=True
        )
        
        ascii_renderer = ASCIIRenderer(dataset)
        ascii_chart = ascii_renderer.render()
        
        logger.info("Backtest completed successfully")
        
        return {
            'dataset': dataset,
            'strategy': strategy,
            'result': result,
            'chart': chart,
            'ascii_chart': ascii_chart,
            'final_value': result.final_portfolio_value,
            'total_return': (result.final_portfolio_value - initial_capital) / initial_capital,
            'num_trades': len(result.trades)
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise WarspiteError(f"Basic backtest example failed: {str(e)}") from e


def multi_strategy_comparison(
    symbols: List[str] = None,
    days: int = 60,
    initial_capital: float = 10000
) -> Dict[str, Any]:
    """
    Compare multiple strategies on the same dataset.
    
    Args:
        symbols: List of symbols to test (default ['AAPL', 'GOOGL'])
        days: Number of days of historical data (default 60)
        initial_capital: Starting capital for each strategy (default 10000)
        
    Returns:
        Dict containing comparison results
        
    Raises:
        WarspiteError: If any step in the workflow fails
    """
    if symbols is None:
        symbols = ['AAPL', 'GOOGL']
    
    try:
        logger.info("Starting multi-strategy comparison")
        
        # Step 1: Load data
        provider = BrownianMotionProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dataset = WarspiteDataset.from_provider(
            provider=provider,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        # Step 2: Define strategies to compare
        strategies = {
            'SMA_10': SMAStrategy(period=10),
            'SMA_20': SMAStrategy(period=20),
            'SMA_50': SMAStrategy(period=50),
            'Random_52': RandomStrategy(correct_percent=0.52),
            'Perfect': PerfectStrategy()
        }
        
        # Step 3: Run backtests for each strategy
        results = {}
        for name, strategy in strategies.items():
            logger.info(f"Testing strategy: {name}")
            
            emulator = WarspiteTradingEmulator(
                dataset=dataset,
                initial_capital=initial_capital,
                trading_fee=0.001,
                spread=0.0001
            )
            emulator.add_strategy(strategy)
            result = emulator.run_to_completion()
            
            results[name] = {
                'strategy': strategy,
                'result': result,
                'final_value': result.final_portfolio_value,
                'total_return': (result.final_portfolio_value - initial_capital) / initial_capital,
                'num_trades': len(result.trades)
            }
        
        # Step 4: Create comparison visualization
        renderer = MatplotlibRenderer(dataset)
        comparison_chart = renderer.render(
            title="Strategy Comparison",
            show_strategy_signals=False,
            show_trades=False
        )
        
        logger.info("Multi-strategy comparison completed")
        
        return {
            'dataset': dataset,
            'strategies': strategies,
            'results': results,
            'comparison_chart': comparison_chart,
            'best_strategy': max(results.keys(), key=lambda k: results[k]['total_return'])
        }
        
    except Exception as e:
        logger.error(f"Multi-strategy comparison failed: {str(e)}")
        raise WarspiteError(f"Multi-strategy comparison failed: {str(e)}") from e


def live_trading_example(
    trading_provider,
    symbols: List[str] = None,
    strategy_type: str = 'sma',
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Example of live trading integration (with safety mechanisms).
    
    Args:
        trading_provider: TradingProvider instance
        symbols: List of symbols to trade (default ['EUR_USD'])
        strategy_type: Strategy to use ('sma', 'random') (default 'sma')
        dry_run: If True, simulate trades without executing (default True)
        
    Returns:
        Dict containing live trading results
        
    Raises:
        WarspiteError: If any step in the workflow fails
    """
    if symbols is None:
        symbols = ['EUR_USD']
    
    try:
        logger.info(f"Starting live trading example (dry_run={dry_run})")
        
        # Step 1: Validate trading provider
        if not hasattr(trading_provider, 'place_order'):
            raise WarspiteError("Provider does not support trading")
        
        # Step 2: Load recent data for strategy initialization
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        dataset = WarspiteDataset.from_provider(
            provider=trading_provider,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1h'
        )
        
        # Step 3: Create strategy
        if strategy_type == 'sma':
            strategy = SMAStrategy(period=20)
        elif strategy_type == 'random':
            strategy = RandomStrategy(correct_percent=0.52)
        else:
            raise WarspiteError(f"Unknown strategy type: {strategy_type}")
        
        # Step 4: Set up emulator with live trading
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=1000,  # Small amount for safety
            trading_fee=0.001,
            spread=0.0002
        )
        emulator.add_strategy(strategy)
        
        if not dry_run:
            emulator.connect_trading_provider(trading_provider)
        
        # Step 5: Execute one step (for safety)
        logger.info("Executing single trading step")
        step_result = emulator.step_forward()
        
        if not dry_run and step_result.trades:
            logger.warning("Live trades executed!")
        
        logger.info("Live trading example completed")
        
        return {
            'dataset': dataset,
            'strategy': strategy,
            'step_result': step_result,
            'dry_run': dry_run,
            'portfolio_value': emulator.get_portfolio_value()
        }
        
    except Exception as e:
        logger.error(f"Live trading example failed: {str(e)}")
        raise WarspiteError(f"Live trading example failed: {str(e)}") from e


def visualization_example(
    symbols: List[str] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Demonstrate various visualization capabilities.
    
    Args:
        symbols: List of symbols to visualize (default ['AAPL'])
        days: Number of days of data (default 30)
        
    Returns:
        Dict containing various visualizations
        
    Raises:
        WarspiteError: If any step in the workflow fails
    """
    if symbols is None:
        symbols = ['AAPL']
    
    try:
        logger.info("Starting visualization example")
        
        # Step 1: Create dataset with strategy results
        provider = BrownianMotionProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dataset = WarspiteDataset.from_provider(
            provider=provider,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        # Step 2: Run strategy to get signals
        strategy = SMAStrategy(period=10)
        emulator = WarspiteTradingEmulator(dataset=dataset)
        emulator.add_strategy(strategy)
        result = emulator.run_to_completion()
        
        # Step 3: Create various visualizations
        visualizations = {}
        
        # Matplotlib chart
        matplotlib_renderer = MatplotlibRenderer(dataset)
        visualizations['matplotlib'] = matplotlib_renderer.render(
            title="Price Chart with Strategy Signals",
            show_strategy_signals=True,
            show_trades=True
        )
        
        # ASCII chart
        ascii_renderer = ASCIIRenderer(dataset)
        visualizations['ascii'] = ascii_renderer.render()
        
        logger.info("Visualization example completed")
        
        return {
            'dataset': dataset,
            'strategy': strategy,
            'result': result,
            'visualizations': visualizations
        }
        
    except Exception as e:
        logger.error(f"Visualization example failed: {str(e)}")
        raise WarspiteError(f"Visualization example failed: {str(e)}") from e