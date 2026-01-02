"""
Test cases for multi-symbol visualization bug.

This module contains regression tests for the visualization system to ensure
that all symbols in a multi-symbol dataset are properly rendered.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from warspite_financial import create_visualization, create_dataset_from_provider
from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.providers.base import BaseProvider


class MockMultiSymbolProvider(BaseProvider):
    """Mock provider for testing multi-symbol datasets."""
    
    def get_data(self, symbol, start_date, end_date, interval='1d'):
        """Return mock OHLCV data for testing."""
        import pandas as pd
        
        # Generate 30 days of mock data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Create different price patterns for different symbols
        if symbol == 'AAPL':
            base_price = 150.0
            trend = 0.1  # Upward trend
        elif symbol == 'NVDA':
            base_price = 300.0
            trend = 0.2  # Stronger upward trend
        elif symbol == 'GOOGL':
            base_price = 2500.0
            trend = -0.05  # Downward trend
        else:
            base_price = 100.0
            trend = 0.0
        
        # Generate price data with different patterns
        np.random.seed(42 + hash(symbol) % 1000)  # Consistent but different per symbol
        price_changes = np.random.normal(trend/100, 0.02, n_days)  # Scale down the trend
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Create OHLCV data
        data = {
            'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def get_available_symbols(self):
        return ['AAPL', 'NVDA', 'GOOGL', 'MSFT', 'TSLA']
    
    def validate_symbol(self, symbol):
        return symbol in self.get_available_symbols()


class TestMultiSymbolVisualization:
    """Test cases for multi-symbol visualization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = MockMultiSymbolProvider()
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()
    
    def test_create_multi_symbol_dataset(self):
        """Test that multi-symbol datasets are created correctly."""
        symbols = ['AAPL', 'NVDA']
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verify dataset structure
        assert len(dataset.symbols) == 2
        assert dataset.symbols == symbols
        assert len(dataset.data_arrays) == 2
        
        # Verify each symbol has different data
        aapl_data = dataset.data_arrays[0]
        nvda_data = dataset.data_arrays[1]
        
        # Close prices should be different (column 3 is Close)
        aapl_close = aapl_data[:, 3]
        nvda_close = nvda_data[:, 3]
        
        # They should not be identical (different price patterns)
        assert not np.array_equal(aapl_close, nvda_close)
        
        # AAPL should be around 150, NVDA around 300
        assert 100 < np.mean(aapl_close) < 200
        assert 250 < np.mean(nvda_close) < 350
    
    def test_price_chart_renders_all_symbols(self):
        """Test that price chart renders all symbols in multi-symbol dataset."""
        symbols = ['AAPL', 'NVDA', 'GOOGL']
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Create price chart
        fig = create_visualization(dataset, 'matplotlib', chart_type='price')
        
        # Get the price axis (should be the first or only axis)
        axes = fig.get_axes()
        price_ax = axes[0]  # First axis should be price
        
        # Check legend labels to see which symbols are plotted
        legend = price_ax.get_legend()
        if legend:
            legend_labels = [text.get_text() for text in legend.get_texts()]
        else:
            # If no legend, check line labels
            lines = price_ax.get_lines()
            legend_labels = [line.get_label() for line in lines if not line.get_label().startswith('_')]
        
        # Verify all symbols appear in the legend/labels
        for symbol in symbols:
            symbol_found = any(symbol in label for label in legend_labels)
            assert symbol_found, f"Symbol {symbol} not found in chart labels: {legend_labels}"
        
        # Verify we have the expected number of lines (one per symbol)
        lines = price_ax.get_lines()
        # Filter out lines that start with '_' (internal matplotlib lines)
        visible_lines = [line for line in lines if not line.get_label().startswith('_')]
        assert len(visible_lines) >= len(symbols), f"Expected at least {len(symbols)} lines, got {len(visible_lines)}"
    
    def test_strategy_chart_renders_all_symbols(self):
        """Test that strategy chart renders all symbols (currently fails - this is the bug)."""
        symbols = ['AAPL', 'NVDA']
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Create strategy chart
        fig = create_visualization(dataset, 'matplotlib', chart_type='strategy')
        
        # Get the price axis (should be the first axis)
        axes = fig.get_axes()
        price_ax = axes[0]  # First axis should be price
        
        # Check what symbols are actually plotted
        legend = price_ax.get_legend()
        if legend:
            legend_labels = [text.get_text() for text in legend.get_texts()]
        else:
            lines = price_ax.get_lines()
            legend_labels = [line.get_label() for line in lines if not line.get_label().startswith('_')]
        
        # This test will currently fail because only AAPL is rendered
        # After the fix, both symbols should be present
        for symbol in symbols:
            symbol_found = any(symbol in label for label in legend_labels)
            assert symbol_found, f"Symbol {symbol} not found in strategy chart labels: {legend_labels}"
    
    def test_combined_chart_renders_all_symbols(self):
        """Test that combined chart renders all symbols (currently fails - this is the bug)."""
        symbols = ['AAPL', 'NVDA']
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Create combined chart
        fig = create_visualization(dataset, 'matplotlib', chart_type='combined')
        
        # Get the price axis (should be the first axis)
        axes = fig.get_axes()
        price_ax = axes[0]  # First axis should be price
        
        # Check what symbols are actually plotted
        legend = price_ax.get_legend()
        if legend:
            legend_labels = [text.get_text() for text in legend.get_texts()]
        else:
            lines = price_ax.get_lines()
            legend_labels = [line.get_label() for line in lines if not line.get_label().startswith('_')]
        
        # This test will currently fail because only AAPL is rendered
        # After the fix, both symbols should be present
        for symbol in symbols:
            symbol_found = any(symbol in label for label in legend_labels)
            assert symbol_found, f"Symbol {symbol} not found in combined chart labels: {legend_labels}"
    
    def test_multi_symbol_with_different_price_ranges(self):
        """Test visualization with symbols having very different price ranges."""
        symbols = ['AAPL', 'GOOGL']  # ~150 vs ~2500 price range
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Create price chart
        fig = create_visualization(dataset, 'matplotlib', chart_type='price')
        
        # Should not raise any errors and should render both symbols
        axes = fig.get_axes()
        price_ax = axes[0]
        
        # Check that both symbols are represented
        lines = price_ax.get_lines()
        visible_lines = [line for line in lines if not line.get_label().startswith('_')]
        assert len(visible_lines) >= 2, "Should have lines for both symbols"
        
        # Verify the y-axis range accommodates both price ranges
        y_min, y_max = price_ax.get_ylim()
        assert y_min < 200, "Y-axis should accommodate AAPL prices (~150)"
        assert y_max > 2000, "Y-axis should accommodate GOOGL prices (~2500)"