"""
Unit tests for visualization capabilities.

These tests verify specific examples and edge cases for the visualization
system in the warspite_financial library.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import tempfile
import os

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.visualization.renderers.base import WarspiteDatasetRenderer
from warspite_financial.visualization.renderers.matplotlib_renderer import MatplotlibRenderer
from warspite_financial.visualization.renderers.ascii_renderer import ASCIIRenderer
from warspite_financial.visualization.renderers.pdf_renderer import PDFRenderer
from warspite_financial.visualization.renderers.csv_renderer import CSVRenderer
from warspite_financial.emulator.emulator import WarspiteTradingEmulator


class TestWarspiteDatasetRenderer:
    """Test the base WarspiteDatasetRenderer class."""
    
    def test_base_renderer_initialization(self):
        """Test base renderer initialization with valid dataset."""
        # Create test dataset
        timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        data_arrays = [np.random.rand(10)]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        
        # Base renderer is abstract, so we can't instantiate it directly
        # Test through a concrete implementation
        renderer = MatplotlibRenderer(dataset)
        
        assert renderer.dataset == dataset
        assert isinstance(renderer.style_options, dict)
        assert renderer.validate_dataset_for_rendering()
    
    def test_base_renderer_invalid_dataset(self):
        """Test base renderer with invalid dataset."""
        with pytest.raises(ValueError, match="Dataset must be a WarspiteDataset instance"):
            MatplotlibRenderer("not_a_dataset")
    
    def test_base_renderer_empty_dataset(self):
        """Test base renderer with empty dataset."""
        # Create empty dataset
        timestamps = np.array([], dtype='datetime64[ns]')
        data_arrays = []
        symbols = []
        
        with pytest.raises(ValueError, match="At least one data array is required"):
            WarspiteDataset(data_arrays, timestamps, symbols)
    
    def test_style_options_management(self):
        """Test style options setting and getting."""
        # Create test dataset
        timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        data_arrays = [np.random.rand(5)]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        renderer = MatplotlibRenderer(dataset)
        
        # Test setting style options
        renderer.set_style_options(figsize=(10, 6), dpi=150)
        
        style_options = renderer.style_options
        assert style_options['figsize'] == (10, 6)
        assert style_options['dpi'] == 150
    
    def test_dataset_summary(self):
        """Test dataset summary generation."""
        # Create test dataset with strategy results
        timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        data_arrays = [np.array([10, 11, 12, 11, 10])]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        dataset.add_strategy_results(np.array([0.5, -0.5, 1.0, -1.0, 0.0]))
        
        renderer = MatplotlibRenderer(dataset)
        summary = renderer.get_dataset_summary()
        
        assert summary['symbols'] == ['TEST']
        assert summary['length'] == 5
        assert summary['has_strategy_results'] is True
        assert 'start_date' in summary
        assert 'end_date' in summary


class TestMatplotlibRenderer:
    """Test the MatplotlibRenderer class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create test dataset
        self.timestamps = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Single symbol with OHLCV data
        ohlcv_data = []
        base_price = 100.0
        for i in range(20):
            open_price = base_price + np.random.uniform(-1, 1)
            close_price = open_price + np.random.uniform(-2, 2)
            high_price = max(open_price, close_price) + np.random.uniform(0, 1)
            low_price = min(open_price, close_price) - np.random.uniform(0, 1)
            volume = np.random.uniform(1000, 10000)
            ohlcv_data.append([open_price, high_price, low_price, close_price, volume])
            base_price = close_price
        
        self.data_arrays = [np.array(ohlcv_data)]
        self.symbols = ['AAPL']
        
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps.values, self.symbols)
        
        # Add strategy results
        strategy_results = np.random.uniform(-1, 1, 20)
        self.dataset.add_strategy_results(strategy_results)
    
    def test_matplotlib_renderer_initialization(self):
        """Test matplotlib renderer initialization."""
        renderer = MatplotlibRenderer(self.dataset)
        
        assert renderer.dataset == self.dataset
        assert 'figsize' in renderer.style_options
        assert 'color_scheme' in renderer.style_options
        assert len(renderer.get_supported_formats()) > 0
    
    def test_price_chart_rendering(self):
        """Test price chart rendering."""
        renderer = MatplotlibRenderer(self.dataset)
        
        fig = renderer.render(chart_type='price', show_volume=True)
        
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 1  # At least price axis
        
        # Check that axes have data
        axes = fig.get_axes()
        price_ax = axes[0]
        assert len(price_ax.get_lines()) > 0  # Should have price line
        
        # Clean up
        plt.close(fig)
    
    def test_strategy_chart_rendering(self):
        """Test strategy chart rendering."""
        renderer = MatplotlibRenderer(self.dataset)
        
        fig = renderer.render(chart_type='strategy')
        
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 2  # Price and position axes
        
        # Clean up
        plt.close(fig)
    
    def test_combined_chart_rendering(self):
        """Test combined chart rendering."""
        renderer = MatplotlibRenderer(self.dataset)
        
        fig = renderer.render(chart_type='combined')
        
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 2  # Multiple axes for combined view
        
        # Clean up
        plt.close(fig)
    
    def test_color_scheme_customization(self):
        """Test custom color scheme functionality."""
        renderer = MatplotlibRenderer(self.dataset)
        
        # Test built-in color schemes
        for scheme in ['default', 'dark', 'minimal']:
            colors = renderer._get_color_scheme(scheme)
            assert 'price' in colors
            assert 'volume' in colors
            assert 'buy_signal' in colors
        
        # Test custom color scheme
        custom_colors = {
            'price': '#FF0000',
            'volume': '#00FF00',
            'buy_signal': '#0000FF',
            'sell_signal': '#FFFF00',
            'position': '#FF00FF',
            'portfolio': '#00FFFF'
        }
        
        renderer.set_custom_color_scheme('custom', custom_colors)
        retrieved_colors = renderer._get_color_scheme('custom')
        assert retrieved_colors == custom_colors
    
    def test_style_theme_customization(self):
        """Test custom style theme functionality."""
        renderer = MatplotlibRenderer(self.dataset)
        
        # Test built-in themes
        for theme in ['default', 'presentation', 'print']:
            style = renderer._get_style_theme(theme)
            assert 'line_width' in style
            assert 'font_size' in style
        
        # Test custom theme
        custom_theme = {
            'line_width': 3.0,
            'font_size': 14,
            'grid_alpha': 0.1
        }
        
        renderer.set_custom_style_theme('custom', custom_theme)
        retrieved_theme = renderer._get_style_theme('custom')
        assert retrieved_theme == custom_theme
    
    def test_save_functionality(self):
        """Test saving rendered output to file."""
        renderer = MatplotlibRenderer(self.dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test PNG save
            png_path = os.path.join(temp_dir, 'test_chart.png')
            fig = renderer.render(chart_type='price')
            renderer.save(png_path)
            
            assert os.path.exists(png_path)
            assert os.path.getsize(png_path) > 0
            
            plt.close(fig)
    
    def test_invalid_chart_type(self):
        """Test handling of invalid chart type."""
        renderer = MatplotlibRenderer(self.dataset)
        
        with pytest.raises(ValueError, match="Unsupported chart type"):
            renderer.render(chart_type='invalid_type')
    
    def test_legend_control(self):
        """Test legend show/hide functionality."""
        renderer = MatplotlibRenderer(self.dataset)
        
        # Test with legend enabled
        fig_with_legend = renderer.render(chart_type='price', show_legend=True)
        axes_with_legend = fig_with_legend.get_axes()
        
        # Test with legend disabled
        fig_without_legend = renderer.render(chart_type='price', show_legend=False)
        axes_without_legend = fig_without_legend.get_axes()
        
        # Both should create figures successfully
        assert isinstance(fig_with_legend, Figure)
        assert isinstance(fig_without_legend, Figure)
        
        # Clean up
        plt.close(fig_with_legend)
        plt.close(fig_without_legend)


class TestASCIIRenderer:
    """Test the ASCIIRenderer class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create simple test dataset
        self.timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        self.data_arrays = [np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])]
        self.symbols = ['TEST']
        
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps.values, self.symbols)
    
    def test_ascii_renderer_initialization(self):
        """Test ASCII renderer initialization."""
        renderer = ASCIIRenderer(self.dataset)
        
        assert renderer.dataset == self.dataset
        assert 'width' in renderer.style_options
        assert 'height' in renderer.style_options
        assert 'chart_chars' in renderer.style_options
    
    def test_price_chart_ascii_rendering(self):
        """Test ASCII price chart rendering."""
        renderer = ASCIIRenderer(self.dataset)
        
        output = renderer.render(chart_type='price')
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert 'Price Chart' in output
        assert 'TEST' in output
    
    def test_summary_rendering(self):
        """Test ASCII summary rendering."""
        renderer = ASCIIRenderer(self.dataset)
        
        output = renderer.render(chart_type='summary')
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert 'Dataset Summary' in output
        assert 'TEST' in output
        assert 'Data points: 10' in output
    
    def test_table_rendering(self):
        """Test ASCII table rendering."""
        renderer = ASCIIRenderer(self.dataset)
        
        output = renderer.render(chart_type='table')
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert 'Dataset Table' in output
    
    def test_strategy_chart_with_results(self):
        """Test ASCII strategy chart with strategy results."""
        # Add strategy results
        strategy_results = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.8, -0.8, 0.3, -0.3, 0.0])
        self.dataset.add_strategy_results(strategy_results)
        
        renderer = ASCIIRenderer(self.dataset)
        
        output = renderer.render(chart_type='strategy')
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert 'Strategy Signals' in output
    
    def test_strategy_chart_without_results(self):
        """Test ASCII strategy chart without strategy results."""
        renderer = ASCIIRenderer(self.dataset)
        
        output = renderer.render(chart_type='strategy')
        
        assert isinstance(output, str)
        assert 'No strategy results available' in output
    
    def test_legend_control_ascii(self):
        """Test ASCII legend show/hide functionality."""
        renderer = ASCIIRenderer(self.dataset)
        
        # Test with legend enabled
        output_with_legend = renderer.render(chart_type='price', show_legend=True)
        assert 'Legend:' in output_with_legend
        
        # Test with legend disabled
        output_without_legend = renderer.render(chart_type='price', show_legend=False)
        assert 'Legend:' not in output_without_legend
    
    def test_save_ascii_output(self):
        """Test saving ASCII output to file."""
        renderer = ASCIIRenderer(self.dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_path = os.path.join(temp_dir, 'test_chart.txt')
            
            output = renderer.render(chart_type='summary')
            renderer.save(txt_path)
            
            assert os.path.exists(txt_path)
            
            # Read back and verify content
            with open(txt_path, 'r') as f:
                saved_content = f.read()
            
            assert len(saved_content) > 0
            assert 'Dataset Summary' in saved_content


class TestPDFRenderer:
    """Test the PDFRenderer class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create test dataset
        self.timestamps = pd.date_range('2023-01-01', periods=15, freq='D')
        self.data_arrays = [np.random.rand(15, 5) * 100 + 50]  # OHLCV data
        self.symbols = ['TEST']
        
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps.values, self.symbols)
    
    def test_pdf_renderer_initialization(self):
        """Test PDF renderer initialization."""
        renderer = PDFRenderer(self.dataset)
        
        assert renderer.dataset == self.dataset
        assert 'pdf_title' in renderer.style_options
        assert 'page_size' in renderer.style_options
        assert 'pdf' in renderer.get_supported_formats()
    
    def test_pdf_rendering(self):
        """Test PDF rendering functionality."""
        renderer = PDFRenderer(self.dataset)
        
        fig = renderer.render(chart_type='price')
        
        assert isinstance(fig, Figure)
        # PDF renderer should optimize for high DPI
        assert fig.dpi >= 300
        
        plt.close(fig)
    
    def test_pdf_save(self):
        """Test saving PDF output."""
        renderer = PDFRenderer(self.dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, 'test_chart.pdf')
            
            fig = renderer.render(chart_type='price')
            renderer.save(pdf_path)
            
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
            
            plt.close(fig)
    
    def test_page_size_options(self):
        """Test different page size options."""
        renderer = PDFRenderer(self.dataset)
        
        # Test different page sizes
        for page_size in ['letter', 'a4', 'legal']:
            for orientation in ['portrait', 'landscape']:
                fig = renderer.render(
                    chart_type='price',
                    page_size=page_size,
                    orientation=orientation
                )
                
                assert isinstance(fig, Figure)
                assert fig.get_figwidth() > 0
                assert fig.get_figheight() > 0
                
                plt.close(fig)


class TestCSVRenderer:
    """Test the CSVRenderer class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create test dataset
        self.timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        self.data_arrays = [
            np.array([[100, 102, 98, 101, 1000],
                     [101, 103, 99, 102, 1100],
                     [102, 104, 100, 103, 1200],
                     [103, 105, 101, 104, 1300],
                     [104, 106, 102, 105, 1400],
                     [105, 107, 103, 106, 1500],
                     [106, 108, 104, 107, 1600],
                     [107, 109, 105, 108, 1700],
                     [108, 110, 106, 109, 1800],
                     [109, 111, 107, 110, 1900]])
        ]
        self.symbols = ['AAPL']
        
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps.values, self.symbols)
    
    def test_csv_renderer_initialization(self):
        """Test CSV renderer initialization."""
        renderer = CSVRenderer(self.dataset)
        
        assert renderer.dataset == self.dataset
        assert 'include_metadata' in renderer.style_options
        assert 'float_precision' in renderer.style_options
        assert 'csv' in renderer.get_supported_formats()
    
    def test_csv_rendering(self):
        """Test CSV rendering functionality."""
        renderer = CSVRenderer(self.dataset)
        
        output = renderer.render()
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert 'AAPL' in output
        assert '2023-01-01' in output
    
    def test_csv_metadata_inclusion(self):
        """Test CSV metadata inclusion/exclusion."""
        renderer = CSVRenderer(self.dataset)
        
        # Test with metadata
        output_with_metadata = renderer.render(include_metadata=True)
        assert '# Financial Data Export' in output_with_metadata
        assert '# Dataset Information:' in output_with_metadata
        
        # Test without metadata
        output_without_metadata = renderer.render(include_metadata=False)
        assert '# Financial Data Export' not in output_without_metadata
    
    def test_csv_summary_inclusion(self):
        """Test CSV summary inclusion/exclusion."""
        renderer = CSVRenderer(self.dataset)
        
        # Test with summary
        output_with_summary = renderer.render(include_summary=True)
        assert '# Summary Statistics:' in output_with_summary
        
        # Test without summary
        output_without_summary = renderer.render(include_summary=False)
        assert '# Summary Statistics:' not in output_without_summary
    
    def test_csv_save(self):
        """Test saving CSV output."""
        renderer = CSVRenderer(self.dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            
            output = renderer.render()
            renderer.save(csv_path)
            
            assert os.path.exists(csv_path)
            
            # Read back and verify content
            with open(csv_path, 'r') as f:
                saved_content = f.read()
            
            assert len(saved_content) > 0
            assert 'AAPL' in saved_content
    
    def test_tsv_format(self):
        """Test TSV (tab-separated values) format."""
        renderer = CSVRenderer(self.dataset)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tsv_path = os.path.join(temp_dir, 'test_data.tsv')
            
            output = renderer.render(separator='\t')
            renderer.save(tsv_path)
            
            assert os.path.exists(tsv_path)
            
            # Read back and verify tab separation
            with open(tsv_path, 'r') as f:
                saved_content = f.read()
            
            # Should contain tabs instead of commas in data section
            lines = saved_content.split('\n')
            data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            if data_lines:
                assert '\t' in data_lines[0]  # Header should have tabs
    
    def test_float_precision_control(self):
        """Test float precision control in CSV output."""
        renderer = CSVRenderer(self.dataset)
        
        # Test different precision levels
        for precision in [2, 4, 6]:
            output = renderer.render(float_precision=precision)
            
            assert isinstance(output, str)
            assert len(output) > 0
    
    def test_analysis_report_generation(self):
        """Test comprehensive analysis report generation."""
        renderer = CSVRenderer(self.dataset)
        
        report = renderer.create_analysis_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert '# Comprehensive Financial Analysis Report' in report
        assert '# Dataset Analysis:' in report


class TestVisualizationIntegration:
    """Test integration between different visualization components."""
    
    def setup_method(self):
        """Set up test data for integration tests."""
        # Create comprehensive test dataset
        self.timestamps = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Multi-symbol dataset
        self.data_arrays = [
            np.random.rand(30, 5) * 100 + 50,  # AAPL OHLCV
            np.random.rand(30) * 200 + 100     # GOOGL single price
        ]
        self.symbols = ['AAPL', 'GOOGL']
        
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps.values, self.symbols)
        
        # Add strategy results
        strategy_results = np.random.uniform(-1, 1, (30, 2))
        self.dataset.add_strategy_results(strategy_results)
        
        # Create emulator with trading history
        self.emulator = WarspiteTradingEmulator(self.dataset, initial_capital=10000)
        
        # Run some simulation steps
        for _ in range(10):
            try:
                self.emulator.step_forward()
            except:
                break
    
    def test_multi_renderer_consistency(self):
        """Test that different renderers handle the same dataset consistently."""
        matplotlib_renderer = MatplotlibRenderer(self.dataset)
        ascii_renderer = ASCIIRenderer(self.dataset)
        csv_renderer = CSVRenderer(self.dataset)
        
        # All renderers should validate the dataset successfully
        assert matplotlib_renderer.validate_dataset_for_rendering()
        assert ascii_renderer.validate_dataset_for_rendering()
        assert csv_renderer.validate_dataset_for_rendering()
        
        # All renderers should generate output
        matplotlib_fig = matplotlib_renderer.render(chart_type='price')
        ascii_output = ascii_renderer.render(chart_type='summary')
        csv_output = csv_renderer.render()
        
        assert isinstance(matplotlib_fig, Figure)
        assert isinstance(ascii_output, str) and len(ascii_output) > 0
        assert isinstance(csv_output, str) and len(csv_output) > 0
        
        # All should contain symbol information
        axes_labels = []
        for ax in matplotlib_fig.get_axes():
            legend = ax.get_legend()
            if legend:
                axes_labels.extend([text.get_text() for text in legend.get_texts()])
        
        # Check that symbols appear in outputs
        for symbol in self.symbols:
            # ASCII and CSV should definitely contain symbols
            assert symbol in ascii_output
            assert symbol in csv_output
        
        plt.close(matplotlib_fig)
    
    def test_emulator_visualization_integration(self):
        """Test visualization integration with emulator data."""
        if len(self.emulator.portfolio_history) < 2:
            pytest.skip("Insufficient emulator history for visualization test")
        
        matplotlib_renderer = MatplotlibRenderer(self.dataset)
        
        # Test portfolio chart rendering
        fig = matplotlib_renderer.render(chart_type='portfolio', emulator=self.emulator)
        
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 2  # Should have multiple axes for portfolio analysis
        
        plt.close(fig)
        
        # Test performance summary
        summary_fig = matplotlib_renderer.create_performance_summary(self.emulator)
        
        assert isinstance(summary_fig, Figure)
        
        plt.close(summary_fig)
    
    def test_save_all_formats(self):
        """Test saving in all supported formats."""
        renderers = {
            'matplotlib': MatplotlibRenderer(self.dataset),
            'ascii': ASCIIRenderer(self.dataset),
            'csv': CSVRenderer(self.dataset)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for renderer_name, renderer in renderers.items():
                for fmt in renderer.get_supported_formats():
                    if fmt == 'default':
                        continue
                    
                    filename = f'test_{renderer_name}.{fmt}'
                    filepath = os.path.join(temp_dir, filename)
                    
                    try:
                        if renderer_name == 'matplotlib':
                            fig = renderer.render(chart_type='price')
                            renderer.save(filepath)
                            plt.close(fig)
                        else:
                            output = renderer.render()
                            renderer.save(filepath)
                        
                        assert os.path.exists(filepath)
                        assert os.path.getsize(filepath) > 0
                        
                    except Exception as e:
                        pytest.fail(f"Failed to save {renderer_name} in {fmt} format: {e}")
    
    def test_error_handling(self):
        """Test error handling in visualization components."""
        # Test with invalid dataset
        invalid_timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        invalid_data_arrays = [np.random.rand(10)]  # Mismatched length
        invalid_symbols = ['TEST']
        
        with pytest.raises(ValueError):
            WarspiteDataset(invalid_data_arrays, invalid_timestamps.values, invalid_symbols)
        
        # Test with valid dataset but invalid operations
        renderer = MatplotlibRenderer(self.dataset)
        
        with pytest.raises(ValueError, match="Unsupported chart type"):
            renderer.render(chart_type='nonexistent_type')
    
    def test_memory_management(self):
        """Test that visualization doesn't cause memory leaks."""
        renderer = MatplotlibRenderer(self.dataset)
        
        # Create and close multiple figures
        for i in range(5):
            fig = renderer.render(chart_type='price')
            assert isinstance(fig, Figure)
            plt.close(fig)
        
        # Close all remaining figures to ensure cleanup
        plt.close('all')
        
        # Should not accumulate figures
        assert len(plt.get_fignums()) == 0
    def test_dimension_mismatch_regression(self):
        """
        Regression test for dimension mismatch between portfolio timestamps and history.
        
        This test reproduces the specific error where portfolio_timestamps has 63 elements
        and portfolio_history has 64 elements, causing a ValueError in matplotlib plotting.
        """
        # Create dataset with specific length that can cause mismatch
        timestamps = pd.date_range('2023-01-01', periods=64, freq='D')
        prices = np.random.uniform(90, 110, 64)
        data_arrays = [prices]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        
        # Create emulator and simulate partial execution that causes mismatch
        emulator = WarspiteTradingEmulator(dataset, initial_capital=10000)
        
        # Simulate scenario where portfolio_history has more entries than timestamp_history
        # This can happen when emulator steps don't perfectly align with dataset timestamps
        emulator._portfolio_history = list(np.random.uniform(9800, 10200, 64))  # 64 entries
        emulator._timestamp_history = timestamps[:63].to_pydatetime().tolist()  # 63 entries (mismatch!)
        
        # This should NOT raise a ValueError anymore
        renderer = MatplotlibRenderer(dataset)
        
        try:
            fig = renderer.render(chart_type='combined', emulator=emulator)
            assert isinstance(fig, Figure)
            
            # Verify that the plot was created successfully
            axes = fig.get_axes()
            assert len(axes) >= 2  # Should have multiple axes for combined chart
            
            # Check that portfolio axis has data (even with mismatched lengths)
            portfolio_ax = None
            for ax in axes:
                if ax.get_ylabel() == 'Portfolio Value':
                    portfolio_ax = ax
                    break
            
            if portfolio_ax is not None:
                # Should have portfolio line plotted (with aligned data)
                lines = portfolio_ax.get_lines()
                if lines:  # Only check if lines exist (might be empty if insufficient data)
                    portfolio_line = lines[0]
                    x_data = portfolio_line.get_xdata()
                    y_data = portfolio_line.get_ydata()
                    assert len(x_data) == len(y_data)  # No dimension mismatch
                    assert len(x_data) <= 64  # Should be truncated to available data
            
            plt.close(fig)
            
        except ValueError as e:
            if "x and y must have same first dimension" in str(e):
                pytest.fail(f"Dimension mismatch regression: {e}")
            else:
                # Re-raise if it's a different ValueError
                raise
    
    def test_empty_portfolio_history_handling(self):
        """Test handling of empty or insufficient portfolio history."""
        timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = np.random.uniform(90, 110, 10)
        data_arrays = [prices]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        
        # Create emulator with empty portfolio history
        emulator = WarspiteTradingEmulator(dataset, initial_capital=10000)
        emulator._portfolio_history = []  # Empty history
        emulator._timestamp_history = []
        
        renderer = MatplotlibRenderer(dataset)
        
        # Should not crash with empty portfolio history
        fig = renderer.render(chart_type='combined', emulator=emulator)
        assert isinstance(fig, Figure)
        plt.close(fig)
        
        # Test with single data point (insufficient for plotting)
        emulator._portfolio_history = [10000]
        emulator._timestamp_history = [timestamps[0].to_pydatetime()]
        
        fig = renderer.render(chart_type='combined', emulator=emulator)
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_mismatched_timestamp_lengths_edge_cases(self):
        """Test various edge cases of timestamp/portfolio history length mismatches."""
        timestamps = pd.date_range('2023-01-01', periods=20, freq='D')
        prices = np.random.uniform(90, 110, 20)
        data_arrays = [prices]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        renderer = MatplotlibRenderer(dataset)
        
        # Test case 1: Portfolio history longer than timestamps
        emulator1 = WarspiteTradingEmulator(dataset, initial_capital=10000)
        emulator1._portfolio_history = list(np.random.uniform(9800, 10200, 25))  # 25 entries
        emulator1._timestamp_history = timestamps[:15].to_pydatetime().tolist()  # 15 entries
        
        fig1 = renderer.render(chart_type='combined', emulator=emulator1)
        assert isinstance(fig1, Figure)
        plt.close(fig1)
        
        # Test case 2: Timestamps longer than portfolio history
        emulator2 = WarspiteTradingEmulator(dataset, initial_capital=10000)
        emulator2._portfolio_history = list(np.random.uniform(9800, 10200, 10))  # 10 entries
        emulator2._timestamp_history = timestamps.to_pydatetime().tolist()  # 20 entries
        
        fig2 = renderer.render(chart_type='combined', emulator=emulator2)
        assert isinstance(fig2, Figure)
        plt.close(fig2)
        
        # Test case 3: No timestamp history (fallback to dataset timestamps)
        emulator3 = WarspiteTradingEmulator(dataset, initial_capital=10000)
        emulator3._portfolio_history = list(np.random.uniform(9800, 10200, 15))  # 15 entries
        emulator3._timestamp_history = None  # No timestamp history
        
        fig3 = renderer.render(chart_type='combined', emulator=emulator3)
        assert isinstance(fig3, Figure)
        plt.close(fig3)
    
    def test_2d_strategy_results_position_chart_regression(self):
        """Regression test: Ensure 2D strategy results are plotted correctly in position chart."""
        timestamps = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.random.uniform(90, 110, 50)
        data_arrays = [prices]
        symbols = ['TEST']
        
        dataset = WarspiteDataset(data_arrays, timestamps.values, symbols)
        
        # Add 2D strategy results (multi-symbol format) - shape (50, 1)
        strategy_results_2d = np.random.uniform(-1, 1, (50, 1))
        dataset.add_strategy_results(strategy_results_2d)
        
        # Verify the strategy results are 2D
        assert dataset.strategy_results.ndim == 2
        assert dataset.strategy_results.shape == (50, 1)
        
        renderer = MatplotlibRenderer(dataset)
        
        # Render combined chart (which includes position chart)
        fig = renderer.render(chart_type='combined')
        assert isinstance(fig, Figure)
        
        # Find the position axis (should be second axis)
        axes = fig.get_axes()
        assert len(axes) >= 2, "Combined chart should have at least 2 axes"
        
        position_ax = None
        for ax in axes:
            if ax.get_ylabel() == 'Position':
                position_ax = ax
                break
        
        assert position_ax is not None, "Position axis should exist"
        
        # Check that position axis has the correct number of data points
        lines = position_ax.get_lines()
        assert len(lines) >= 1, "Position axis should have at least one line"
        
        position_line = lines[0]
        y_data = position_line.get_ydata()
        
        # Regression test: Should have 50 data points, not just 2
        assert len(y_data) == 50, f"Position line should have 50 data points, got {len(y_data)}"
        
        # Verify data range is correct
        assert y_data.min() >= -1.0 and y_data.max() <= 1.0, "Position values should be in [-1, 1] range"
        
        plt.close(fig)