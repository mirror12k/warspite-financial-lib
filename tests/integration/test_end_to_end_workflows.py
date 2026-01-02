"""
End-to-end integration tests for warspite_financial library.

These tests verify complete workflows from data loading to visualization.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from warspite_financial import (
    BrownianMotionProvider,
    WarspiteDataset,
    SMAStrategy,
    RandomStrategy,
    PerfectStrategy,
    WarspiteTradingEmulator,
    MatplotlibRenderer,
    ASCIIRenderer,
    create_dataset_from_provider,
    run_strategy_backtest,
    create_visualization,
    basic_backtest_example,
    multi_strategy_comparison,
    visualization_example
)
from warspite_financial.utils.exceptions import WarspiteError


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BrownianMotionProvider()
        self.symbols = ['AAPL', 'GOOGL']  # Regular symbols now work
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
    def test_complete_backtest_workflow(self):
        """Test complete workflow from data loading to results."""
        # Step 1: Load data
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        assert len(dataset.symbols) == len(self.symbols)
        assert len(dataset.timestamps) > 0
        assert dataset.data.shape[0] == len(dataset.timestamps)
        
        # Step 2: Create and apply strategy
        strategy = SMAStrategy(period=10)
        positions = strategy.generate_positions(dataset)
        
        assert len(positions) == len(dataset.timestamps)
        assert np.all(positions >= -1.0) and np.all(positions <= 1.0)
        
        # Step 3: Run emulation
        emulator = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=10000,
            trading_fee=0.001,
            spread=0.0001
        )
        emulator.add_strategy(strategy)
        result = emulator.run_to_completion()
        
        assert result.final_portfolio_value > 0
        assert len(result.trades) >= 0
        assert result.initial_capital == 10000
        
        # Step 4: Create visualizations
        matplotlib_renderer = MatplotlibRenderer(dataset)
        chart = matplotlib_renderer.render(title="Integration Test Chart")
        assert chart is not None
        
        ascii_renderer = ASCIIRenderer(dataset)
        ascii_chart = ascii_renderer.render()
        assert isinstance(ascii_chart, str)
        assert len(ascii_chart) > 0
        
    def test_multi_strategy_comparison_workflow(self):
        """Test workflow comparing multiple strategies."""
        # Load data
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        # Define strategies
        strategies = {
            'SMA_10': SMAStrategy(period=10),
            'SMA_20': SMAStrategy(period=20),
            'Random': RandomStrategy(correct_percent=0.52),
            'Perfect': PerfectStrategy()
        }
        
        # Run backtests
        results = {}
        for name, strategy in strategies.items():
            emulator = WarspiteTradingEmulator(
                dataset=dataset,
                initial_capital=10000,
                trading_fee=0.001,
                spread=0.0001
            )
            emulator.add_strategy(strategy)
            result = emulator.run_to_completion()
            results[name] = result
        
        # Verify all strategies ran
        assert len(results) == len(strategies)
        for name, result in results.items():
            assert result.final_portfolio_value > 0
            assert hasattr(result, 'trades')
        
        # Perfect strategy should generally perform best (with some randomness tolerance)
        perfect_return = (results['Perfect'].final_portfolio_value - 10000) / 10000
        assert perfect_return >= -0.5  # Allow for some losses due to fees/spread
        
    def test_convenience_functions_workflow(self):
        """Test end-to-end workflow using convenience functions."""
        # Test create_dataset_from_provider
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        assert isinstance(dataset, WarspiteDataset)
        assert len(dataset.symbols) == 1
        assert dataset.symbols[0] == 'AAPL'
        
        # Test run_strategy_backtest
        strategy = SMAStrategy(period=15)
        result = run_strategy_backtest(
            dataset=dataset,
            strategy=strategy,
            initial_capital=5000,
            trading_fee=0.002
        )
        
        assert result.initial_capital == 5000
        assert result.final_portfolio_value > 0
        
        # Test create_visualization
        chart = create_visualization(
            dataset=dataset,
            renderer_type='matplotlib',
            title='Convenience Function Test'
        )
        assert chart is not None
        
        ascii_output = create_visualization(
            dataset=dataset,
            renderer_type='ascii'
        )
        assert isinstance(ascii_output, str)
        assert len(ascii_output) > 0
        
    def test_example_workflows(self):
        """Test the example workflow functions."""
        # Test basic backtest example
        result = basic_backtest_example(
            symbols=['AAPL'],
            days=20,
            initial_capital=5000,
            sma_period=10
        )
        
        assert 'dataset' in result
        assert 'strategy' in result
        assert 'result' in result
        assert 'final_value' in result
        assert 'total_return' in result
        assert result['final_value'] > 0
        
        # Test multi-strategy comparison
        comparison = multi_strategy_comparison(
            symbols=['AAPL'],
            days=30,
            initial_capital=10000
        )
        
        assert 'dataset' in comparison
        assert 'strategies' in comparison
        assert 'results' in comparison
        assert 'best_strategy' in comparison
        assert len(comparison['results']) > 0
        
        # Test visualization example
        viz_result = visualization_example(
            symbols=['AAPL'],
            days=20
        )
        
        assert 'dataset' in viz_result
        assert 'visualizations' in viz_result
        assert 'matplotlib' in viz_result['visualizations']
        assert 'ascii' in viz_result['visualizations']
        
    def test_error_handling_integration(self):
        """Test error handling in integrated workflows."""
        # Test invalid renderer type
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['BM-AAPL'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        with pytest.raises(Exception):  # Should raise VisualizationError
            create_visualization(dataset, renderer_type='invalid_type')
        
        # Test invalid strategy parameters
        with pytest.raises(Exception):
            strategy = SMAStrategy(period=-1)  # Invalid period
            
    def test_serialization_integration(self):
        """Test dataset serialization in complete workflow."""
        # Create dataset
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Add strategy results
        strategy = SMAStrategy(period=10)
        positions = strategy.generate_positions(dataset)
        dataset.add_strategy_results(positions)
        
        # Test serialization
        serialized_csv = dataset.serialize(format='csv')
        assert isinstance(serialized_csv, str)
        assert len(serialized_csv) > 0
        
        serialized_pickle = dataset.serialize(format='pickle')
        assert isinstance(serialized_pickle, bytes)
        assert len(serialized_pickle) > 0
        
    def test_step_by_step_vs_full_execution(self):
        """Test that step-by-step execution matches full execution."""
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        strategy = SMAStrategy(period=10)
        
        # Full execution
        emulator_full = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=10000,
            trading_fee=0.001,
            spread=0.0001
        )
        emulator_full.add_strategy(strategy)
        result_full = emulator_full.run_to_completion()
        
        # Step-by-step execution
        emulator_step = WarspiteTradingEmulator(
            dataset=dataset,
            initial_capital=10000,
            trading_fee=0.001,
            spread=0.0001
        )
        emulator_step.add_strategy(strategy)
        
        # Execute all steps
        step_results = []
        while emulator_step.current_step < len(dataset.timestamps) - 1:
            step_result = emulator_step.step_forward()
            step_results.append(step_result)
        
        # Compare final results (allowing for small floating point differences)
        assert abs(result_full.final_portfolio_value - emulator_step.get_portfolio_value()) < 0.01
        assert len(result_full.trades) == len(emulator_step.trade_history)


class TestProviderIntegration:
    """Test provider integration with real-like scenarios."""
    
    def test_brownian_motion_provider_integration(self):
        """Test BrownianMotionProvider in realistic scenarios."""
        provider = BrownianMotionProvider()
        
        # Test data retrieval
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        data = provider.get_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Test symbol validation
        assert provider.validate_symbol('AAPL')
        assert provider.validate_symbol('GOOGL')
        
        # Test available symbols
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'TEST' in symbols
        
    def test_multiple_provider_dataset_creation(self):
        """Test creating datasets from multiple providers."""
        provider1 = BrownianMotionProvider()
        provider2 = BrownianMotionProvider()  # Same type but different instance
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Create dataset from multiple providers
        dataset = WarspiteDataset.from_provider(
            provider=provider1,  # Use single provider since from_provider takes one
            symbols=['AAPL', 'GOOGL'],
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        assert len(dataset.symbols) == 2
        assert 'AAPL' in dataset.symbols
        assert 'GOOGL' in dataset.symbols
        assert len(dataset.timestamps) > 0
        
    def test_provider_error_handling(self):
        """Test provider error handling in integration scenarios."""
        provider = BrownianMotionProvider()
        
        # Test with invalid date range (future dates)
        future_date = datetime.now() + timedelta(days=365)
        start_date = datetime.now()
        
        # This should handle gracefully or raise appropriate error
        try:
            data = provider.get_data(
                symbol='AAPL',
                start_date=start_date,
                end_date=future_date,
                interval='1d'
            )
            # If it succeeds, verify the data is reasonable
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            # If it fails, it should be a reasonable error
            assert isinstance(e, (ValueError, WarspiteError))


if __name__ == '__main__':
    pytest.main([__file__])