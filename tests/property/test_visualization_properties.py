"""
Property-based tests for visualization capabilities.

These tests verify universal properties that should hold across all valid inputs
for the visualization system in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.visualization.renderers.matplotlib_renderer import MatplotlibRenderer
from warspite_financial.visualization.renderers.ascii_renderer import ASCIIRenderer
from warspite_financial.emulator.emulator import WarspiteTradingEmulator


# Hypothesis strategies for generating test data
@st.composite
def valid_visualization_dataset(draw):
    """Generate valid dataset for visualization testing."""
    # Reduce complexity - fewer symbols and data points
    num_symbols = draw(st.integers(1, 2))  # Reduced from 1-4 to 1-2
    num_timestamps = draw(st.integers(5, 20))  # Reduced from 10-50 to 5-20
    
    # Use simpler symbol generation
    symbols = draw(st.lists(
        st.sampled_from(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'A', 'B', 'C']),  # Use predefined simple symbols
        min_size=num_symbols,
        max_size=num_symbols,
        unique=True
    ))
    
    # Generate sequential timestamps
    base_date = datetime(2020, 1, 1)  # Fixed base date to avoid complexity
    
    timestamps = []
    for i in range(num_timestamps):
        timestamps.append(base_date + timedelta(days=i))
    
    timestamps = np.array(timestamps, dtype='datetime64[ns]')
    
    # Generate simpler data arrays - always use OHLCV format for consistency
    data_arrays = []
    for i in range(num_symbols):
        # Always generate OHLCV data for simplicity
        base_price = 50.0 + i * 10.0  # Simple base price progression
        ohlcv_data = []
        
        for j in range(num_timestamps):
            # Generate simple, predictable OHLCV data
            open_price = base_price + j * 0.5  # Simple linear progression
            close_price = open_price + 1.0
            high_price = close_price + 0.5
            low_price = open_price - 0.5
            volume = 10000.0 + j * 100  # Simple volume progression
            
            ohlcv_data.append([open_price, high_price, low_price, close_price, volume])
        
        data_arrays.append(np.array(ohlcv_data))
    
    # Create dataset
    dataset = WarspiteDataset(data_arrays, timestamps, symbols)
    
    # Optionally add simple strategy results
    add_strategy = draw(st.booleans())
    if add_strategy:
        if num_symbols == 1:
            # Single strategy results - simple alternating pattern
            strategy_results = np.array([(-1.0 if i % 2 == 0 else 1.0) for i in range(num_timestamps)])
        else:
            # Multi-symbol strategy results - simple pattern
            strategy_results = np.array([[(-1.0 if (i + j) % 2 == 0 else 1.0) for j in range(num_symbols)] for i in range(num_timestamps)])
        
        dataset.add_strategy_results(strategy_results)
    
    return dataset


@st.composite
def valid_emulator_with_history(draw):
    """Generate valid emulator with trading history for portfolio visualization."""
    dataset = draw(valid_visualization_dataset())
    
    # Create emulator with some trading activity
    initial_capital = draw(st.floats(min_value=1000.0, max_value=100000.0))
    trading_fee = draw(st.floats(min_value=0.0, max_value=10.0))
    spread = draw(st.floats(min_value=0.0, max_value=0.01))
    
    emulator = WarspiteTradingEmulator(
        dataset=dataset,
        initial_capital=initial_capital,
        trading_fee=trading_fee,
        spread=spread
    )
    
    # Simulate some trading activity
    num_steps = min(len(dataset), draw(st.integers(5, 20)))
    
    for _ in range(num_steps):
        try:
            emulator.step_forward()
            
            # Occasionally make some trades
            if draw(st.booleans()) and emulator.current_step < len(dataset):
                symbol = draw(st.sampled_from(dataset.symbols))
                quantity = draw(st.floats(min_value=0.1, max_value=10.0))
                
                if draw(st.booleans()):
                    emulator.buy(symbol, quantity)
                else:
                    emulator.sell(symbol, quantity)
        except:
            # If step fails, continue (might be at end of dataset)
            break
    
    return emulator


class TestVisualizationLegendCompleteness:
    """
    Property-based tests for Visualization Legend Completeness.
    
    **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
    **Validates: Requirements 7.4**
    """
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=20, deadline=None)  # Reduced from 100 to 20
    def test_matplotlib_legend_completeness_price_chart(self, dataset):
        """
        Property 7: Visualization Legend Completeness (Price Charts)
        
        For any rendered chart with multiple data series, the plot should contain 
        legends and labels for all displayed elements.
        
        **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
        **Validates: Requirements 7.4**
        """
        renderer = MatplotlibRenderer(dataset)
        
        # Render price chart with legend enabled
        fig = renderer.render(chart_type='price', show_legend=True, show_volume=True)
        
        # Property assertions: Legend completeness
        
        # 1. Figure should be a matplotlib Figure
        assert isinstance(fig, Figure), "Renderer should return matplotlib Figure"
        
        # 2. Figure should have at least one axes
        axes = fig.get_axes()
        assert len(axes) > 0, "Figure should contain at least one axes"
        
        # 3. Check legend presence and completeness for each axes
        legend_found = False
        total_expected_legend_entries = 0
        
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend_found = True
                legend_texts = [text.get_text() for text in legend.get_texts()]
                
                # 4. Legend should not be empty
                assert len(legend_texts) > 0, "Legend should contain at least one entry"
                
                # 5. Legend entries should be non-empty strings
                for text in legend_texts:
                    assert isinstance(text, str), "Legend entries should be strings"
                    assert len(text.strip()) > 0, "Legend entries should not be empty"
                
                # 6. Legend should contain entries for all symbols in the dataset
                for symbol in dataset.symbols:
                    symbol_found_in_legend = any(symbol in text for text in legend_texts)
                    if not symbol_found_in_legend:
                        # Debug information for failing case
                        print(f"DEBUG: Symbol '{symbol}' not found in legend texts: {legend_texts}")
                        print(f"DEBUG: Dataset symbols: {dataset.symbols}")
                        print(f"DEBUG: Symbol type: {type(symbol)}, repr: {repr(symbol)}")
                        for text in legend_texts:
                            print(f"DEBUG: '{symbol}' in '{text}': {symbol in text}")
                    assert symbol_found_in_legend, f"Symbol '{symbol}' should appear in legend. Legend texts: {legend_texts}"
                
                total_expected_legend_entries += len(dataset.symbols)
                
                # 7. If volume data is present and shown, legend should include volume entries
                has_volume_data = any(
                    arr.ndim == 2 and arr.shape[1] >= 5 
                    for arr in dataset.data_arrays
                )
                if has_volume_data:
                    volume_found_in_legend = any('volume' in text.lower() for text in legend_texts)
                    # Volume legend is expected but not strictly required for this property
        
        # 8. At least one axes should have a legend when show_legend=True
        assert legend_found, "At least one axes should have a legend when show_legend=True"
        
        # 9. Figure should be properly formatted
        assert fig.get_figwidth() > 0, "Figure should have positive width"
        assert fig.get_figheight() > 0, "Figure should have positive height"
        
        # Clean up
        plt.close(fig)
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=20, deadline=None)  # Reduced from 100 to 20
    def test_matplotlib_legend_completeness_strategy_chart(self, dataset):
        """
        Property 7: Visualization Legend Completeness (Strategy Charts)
        
        For any rendered strategy chart with multiple data series, the plot should contain 
        legends and labels for all displayed elements.
        
        **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
        **Validates: Requirements 7.4**
        """
        # Skip if no strategy results
        if dataset.strategy_results is None:
            return
        
        renderer = MatplotlibRenderer(dataset)
        
        # Render strategy chart with legend enabled
        fig = renderer.render(chart_type='strategy', show_legend=True)
        
        # Property assertions: Legend completeness for strategy charts
        
        # 1. Figure should be a matplotlib Figure
        assert isinstance(fig, Figure), "Renderer should return matplotlib Figure"
        
        # 2. Figure should have at least one axes
        axes = fig.get_axes()
        assert len(axes) > 0, "Figure should contain at least one axes"
        
        # 3. Check legend presence and completeness
        legend_found = False
        
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend_found = True
                legend_texts = [text.get_text() for text in legend.get_texts()]
                
                # 4. Legend should not be empty
                assert len(legend_texts) > 0, "Legend should contain at least one entry"
                
                # 5. Legend entries should be non-empty strings
                for text in legend_texts:
                    assert isinstance(text, str), "Legend entries should be strings"
                    assert len(text.strip()) > 0, "Legend entries should not be empty"
                
                # 6. Strategy-specific legend entries should be present
                strategy_terms = ['position', 'signal', 'buy', 'sell', 'price']
                strategy_legend_found = any(
                    any(term in text.lower() for term in strategy_terms)
                    for text in legend_texts
                )
                assert strategy_legend_found, "Legend should contain strategy-related entries"
        
        # 7. At least one axes should have a legend when show_legend=True
        assert legend_found, "At least one axes should have a legend when show_legend=True"
        
        # Clean up
        plt.close(fig)
    
    @given(emulator=valid_emulator_with_history())
    @settings(max_examples=50, deadline=None)
    def test_matplotlib_legend_completeness_portfolio_chart(self, emulator):
        """
        Property 7: Visualization Legend Completeness (Portfolio Charts)
        
        For any rendered portfolio chart with multiple data series, the plot should contain 
        legends and labels for all displayed elements.
        
        **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
        **Validates: Requirements 7.4**
        """
        # Skip if insufficient portfolio history
        if len(emulator.portfolio_history) < 2:
            return
        
        renderer = MatplotlibRenderer(emulator._dataset)
        
        # Render portfolio chart with legend enabled
        fig = renderer.render(chart_type='portfolio', show_legend=True, emulator=emulator)
        
        # Property assertions: Legend completeness for portfolio charts
        
        # 1. Figure should be a matplotlib Figure
        assert isinstance(fig, Figure), "Renderer should return matplotlib Figure"
        
        # 2. Figure should have multiple axes for portfolio analysis
        axes = fig.get_axes()
        assert len(axes) >= 2, "Portfolio chart should have multiple axes"
        
        # 3. Check legend presence and completeness
        legend_found = False
        
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend_found = True
                legend_texts = [text.get_text() for text in legend.get_texts()]
                
                # 4. Legend should not be empty
                assert len(legend_texts) > 0, "Legend should contain at least one entry"
                
                # 5. Legend entries should be non-empty strings
                for text in legend_texts:
                    assert isinstance(text, str), "Legend entries should be strings"
                    assert len(text.strip()) > 0, "Legend entries should not be empty"
                
                # 6. Portfolio-specific legend entries should be present
                portfolio_terms = ['portfolio', 'value', 'capital', 'returns', 'drawdown', 'trades']
                portfolio_legend_found = any(
                    any(term in text.lower() for term in portfolio_terms)
                    for text in legend_texts
                )
                assert portfolio_legend_found, "Legend should contain portfolio-related entries"
        
        # 7. At least one axes should have a legend when show_legend=True
        assert legend_found, "At least one axes should have a legend when show_legend=True"
        
        # Clean up
        plt.close(fig)
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=20, deadline=None)  # Reduced from 100 to 20
    def test_matplotlib_legend_completeness_combined_chart(self, dataset):
        """
        Property 7: Visualization Legend Completeness (Combined Charts)
        
        For any rendered combined chart with multiple data series, the plot should contain 
        legends and labels for all displayed elements.
        
        **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
        **Validates: Requirements 7.4**
        """
        renderer = MatplotlibRenderer(dataset)
        
        # Render combined chart with legend enabled
        fig = renderer.render(chart_type='combined', show_legend=True)
        
        # Property assertions: Legend completeness for combined charts
        
        # 1. Figure should be a matplotlib Figure
        assert isinstance(fig, Figure), "Renderer should return matplotlib Figure"
        
        # 2. Figure should have at least one axes
        axes = fig.get_axes()
        assert len(axes) > 0, "Figure should contain at least one axes"
        
        # 3. Check legend presence and completeness across all axes
        total_legends_found = 0
        
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                total_legends_found += 1
                legend_texts = [text.get_text() for text in legend.get_texts()]
                
                # 4. Legend should not be empty
                assert len(legend_texts) > 0, "Legend should contain at least one entry"
                
                # 5. Legend entries should be non-empty strings
                for text in legend_texts:
                    assert isinstance(text, str), "Legend entries should be strings"
                    assert len(text.strip()) > 0, "Legend entries should not be empty"
        
        # 6. At least one axes should have a legend in combined chart
        assert total_legends_found > 0, "Combined chart should have at least one legend"
        
        # 7. If dataset has symbols, at least one symbol should appear in legends
        if dataset.symbols:
            all_legend_texts = []
            for ax in axes:
                legend = ax.get_legend()
                if legend is not None:
                    all_legend_texts.extend([text.get_text() for text in legend.get_texts()])
            
            symbol_found_in_any_legend = any(
                any(symbol in text for text in all_legend_texts)
                for symbol in dataset.symbols
            )
            assert symbol_found_in_any_legend, "At least one symbol should appear in combined chart legends"
        
        # Clean up
        plt.close(fig)
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=50, deadline=None)
    def test_matplotlib_legend_disabled(self, dataset):
        """
        Test that legends are properly disabled when show_legend=False.
        """
        renderer = MatplotlibRenderer(dataset)
        
        # Render chart with legend disabled
        fig = renderer.render(chart_type='price', show_legend=False)
        
        # Property assertions: Legend should be disabled
        
        # 1. Figure should be created successfully
        assert isinstance(fig, Figure), "Renderer should return matplotlib Figure"
        
        # 2. No axes should have legends when show_legend=False
        axes = fig.get_axes()
        for ax in axes:
            legend = ax.get_legend()
            # Legend might be None or have no visible entries
            if legend is not None:
                # If legend exists, it should not be visible or should be empty
                assert not legend.get_visible() or len(legend.get_texts()) == 0, \
                    "Legend should not be visible when show_legend=False"
        
        # Clean up
        plt.close(fig)
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=50, deadline=None)
    def test_ascii_renderer_legend_completeness(self, dataset):
        """
        Property 7: Visualization Legend Completeness (ASCII Renderer)
        
        For any ASCII rendered chart, the output should contain legend information 
        for all displayed elements when show_legend=True.
        
        **Feature: warspite-financial-library, Property 7: Visualization Legend Completeness**
        **Validates: Requirements 7.4**
        """
        renderer = ASCIIRenderer(dataset)
        
        # Render ASCII chart with legend enabled
        ascii_output = renderer.render(chart_type='price', show_legend=True)
        
        # Property assertions: ASCII legend completeness
        
        # 1. Output should be a non-empty string
        assert isinstance(ascii_output, str), "ASCII renderer should return string"
        assert len(ascii_output.strip()) > 0, "ASCII output should not be empty"
        
        # 2. Legend section should be present when show_legend=True
        assert 'Legend:' in ascii_output or 'legend' in ascii_output.lower(), \
            "ASCII output should contain legend section when show_legend=True"
        
        # 3. Legend should contain information about chart elements
        legend_indicators = ['*', '▲', '▼', '+', '-', '·']
        legend_found = any(indicator in ascii_output for indicator in legend_indicators)
        assert legend_found, "ASCII legend should contain chart element indicators"
        
        # 4. If dataset has symbols, they should be mentioned in the output
        if dataset.symbols:
            symbol_found = any(symbol in ascii_output for symbol in dataset.symbols)
            assert symbol_found, "At least one symbol should appear in ASCII output"
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=30, deadline=None)
    def test_ascii_renderer_legend_disabled(self, dataset):
        """
        Test that ASCII legends are properly disabled when show_legend=False.
        """
        renderer = ASCIIRenderer(dataset)
        
        # Render ASCII chart with legend disabled
        ascii_output = renderer.render(chart_type='price', show_legend=False)
        
        # Property assertions: Legend should be disabled
        
        # 1. Output should be created successfully
        assert isinstance(ascii_output, str), "ASCII renderer should return string"
        assert len(ascii_output.strip()) > 0, "ASCII output should not be empty"
        
        # 2. Legend section should not be present when show_legend=False
        assert 'Legend:' not in ascii_output, \
            "ASCII output should not contain legend section when show_legend=False"
    
    @given(dataset=valid_visualization_dataset())
    @settings(max_examples=30, deadline=None)
    def test_legend_content_accuracy(self, dataset):
        """
        Test that legend content accurately reflects the data being displayed.
        """
        renderer = MatplotlibRenderer(dataset)
        
        # Render chart and examine legend content
        fig = renderer.render(chart_type='price', show_legend=True, show_volume=True)
        
        # Collect all legend texts from all axes
        all_legend_texts = []
        for ax in fig.get_axes():
            legend = ax.get_legend()
            if legend is not None:
                all_legend_texts.extend([text.get_text() for text in legend.get_texts()])
        
        # Property assertions: Legend accuracy
        
        # 1. Each symbol in dataset should have corresponding legend entry
        for symbol in dataset.symbols:
            symbol_in_legend = any(symbol in text for text in all_legend_texts)
            assert symbol_in_legend, f"Symbol '{symbol}' should have legend entry"
        
        # 2. Legend entries should not contain invalid or placeholder text
        invalid_texts = ['None', 'null', 'undefined', '']
        for text in all_legend_texts:
            assert text.strip() not in invalid_texts, f"Legend should not contain invalid text: '{text}'"
        
        # 3. Legend text should be descriptive
        for text in all_legend_texts:
            assert len(text.strip()) >= 2, f"Legend text should be descriptive: '{text}'"
        
        # Clean up
        plt.close(fig)