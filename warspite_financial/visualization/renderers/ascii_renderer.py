"""
ASCII renderer for warspite_financial visualization.

This module provides ASCII-based console visualization for datasets and trading results.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from .base import WarspiteDatasetRenderer


class ASCIIRenderer(WarspiteDatasetRenderer):
    """
    ASCII-based renderer for console visualization of WarspiteDataset.
    
    This renderer creates text-based charts and tables suitable for console output
    or text-based interfaces.
    """
    
    def __init__(self, dataset):
        """
        Initialize ASCII renderer.
        
        Args:
            dataset: WarspiteDataset instance to render
        """
        super().__init__(dataset)
        
        # Default style options
        self._style_options = {
            'width': 80,
            'height': 20,
            'chart_chars': {
                'horizontal': '─',
                'vertical': '│',
                'corner_tl': '┌',
                'corner_tr': '┐',
                'corner_bl': '└',
                'corner_br': '┘',
                'cross': '┼',
                'tee_up': '┴',
                'tee_down': '┬',
                'tee_left': '┤',
                'tee_right': '├',
                'price_char': '*',
                'signal_buy': '▲',
                'signal_sell': '▼',
                'position_pos': '+',
                'position_neg': '-',
                'position_zero': '·'
            },
            'show_legend': True,
            'show_grid': True,
            'precision': 2
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return ['txt', 'ascii']
    
    def render(self, **kwargs) -> str:
        """
        Render the dataset as ASCII text.
        
        Args:
            **kwargs: Rendering options including:
                - chart_type: 'price', 'strategy', 'summary', 'table'
                - symbol: Specific symbol to render (default: first symbol)
                - show_values: Whether to show numeric values
                
        Returns:
            ASCII text representation of the dataset
        """
        if not self.validate_dataset_for_rendering():
            raise ValueError("Dataset is not suitable for rendering")
        
        # Merge options
        options = {**self._style_options, **kwargs}
        chart_type = options.get('chart_type', 'summary')
        
        # Render based on chart type
        if chart_type == 'price':
            return self._render_price_chart(options)
        elif chart_type == 'strategy':
            return self._render_strategy_chart(options)
        elif chart_type == 'summary':
            return self._render_summary(options)
        elif chart_type == 'table':
            return self._render_table(options)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _render_price_chart(self, options: Dict[str, Any]) -> str:
        """Render ASCII price chart."""
        symbol = options.get('symbol', self._dataset.symbols[0])
        width = options.get('width', 80)
        height = options.get('height', 20)
        chars = options.get('chart_chars', self._style_options['chart_chars'])
        
        # Get symbol data
        if symbol not in self._dataset.symbols:
            return f"Error: Symbol '{symbol}' not found in dataset"
        
        symbol_idx = self._dataset.symbols.index(symbol)
        data_array = self._dataset.data_arrays[symbol_idx]
        
        # Extract price data
        if data_array.ndim == 1:
            prices = data_array
        else:
            # Use close prices if available
            prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
        
        # Create chart
        chart_lines = []
        
        # Title
        chart_lines.append(f"Price Chart: {symbol}")
        chart_lines.append("=" * len(chart_lines[-1]))
        chart_lines.append("")
        
        # Calculate chart dimensions
        chart_width = width - 15  # Leave space for y-axis labels
        chart_height = height - 5  # Leave space for title and x-axis
        
        # Normalize prices to chart height
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            price_range = 1  # Avoid division by zero
        
        # Sample data points to fit chart width
        data_points = len(prices)
        if data_points > chart_width:
            # Sample evenly across the data
            indices = np.linspace(0, data_points - 1, chart_width, dtype=int)
            sampled_prices = prices[indices]
        else:
            sampled_prices = prices
            indices = np.arange(len(prices))
        
        # Create the chart grid
        chart_grid = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Plot price line
        for i, price in enumerate(sampled_prices):
            if i < chart_width:
                # Calculate y position (inverted for display)
                y_pos = int((max_price - price) / price_range * (chart_height - 1))
                y_pos = max(0, min(chart_height - 1, y_pos))
                chart_grid[y_pos][i] = chars['price_char']
        
        # Add y-axis labels and render chart
        for i, row in enumerate(chart_grid):
            # Calculate price for this row
            row_price = max_price - (i / (chart_height - 1)) * price_range
            price_label = f"{row_price:8.{options.get('precision', 2)}f}"
            
            # Add border and content
            if i == 0:
                line = f"{price_label} {chars['corner_tl']}" + "".join(row) + chars['corner_tr']
            elif i == chart_height - 1:
                line = f"{price_label} {chars['corner_bl']}" + "".join(row) + chars['corner_br']
            else:
                line = f"{price_label} {chars['vertical']}" + "".join(row) + chars['vertical']
            
            chart_lines.append(line)
        
        # Add x-axis
        x_axis = " " * 10 + chars['corner_bl'] + chars['horizontal'] * chart_width + chars['corner_br']
        chart_lines.append(x_axis)
        
        # Add timestamps for x-axis labels
        if len(self._dataset.timestamps) > 0:
            start_date = pd.to_datetime(self._dataset.timestamps[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(self._dataset.timestamps[-1]).strftime('%Y-%m-%d')
            date_line = f" " * 10 + f"{start_date}" + " " * (chart_width - len(start_date) - len(end_date)) + f"{end_date}"
            chart_lines.append(date_line)
        
        # Add legend if requested
        if options.get('show_legend', True):
            chart_lines.append("")
            chart_lines.append("Legend:")
            chart_lines.append(f"  {chars['price_char']} Price")
        
        return "\n".join(chart_lines)
    
    def _render_strategy_chart(self, options: Dict[str, Any]) -> str:
        """Render ASCII strategy signals chart."""
        if self._dataset.strategy_results is None:
            return "No strategy results available for rendering"
        
        width = options.get('width', 80)
        height = options.get('height', 20)
        chars = options.get('chart_chars', self._style_options['chart_chars'])
        
        chart_lines = []
        
        # Title
        chart_lines.append("Strategy Signals")
        chart_lines.append("=" * len(chart_lines[-1]))
        chart_lines.append("")
        
        # Get strategy results
        strategy_results = self._dataset.strategy_results
        
        if strategy_results.ndim == 1:
            # Single strategy
            positions = strategy_results
            symbol = self._dataset.symbols[0] if self._dataset.symbols else "Unknown"
            
            # Create position chart
            chart_width = width - 10
            chart_height = height - 5
            
            # Sample positions to fit chart width
            if len(positions) > chart_width:
                indices = np.linspace(0, len(positions) - 1, chart_width, dtype=int)
                sampled_positions = positions[indices]
            else:
                sampled_positions = positions
            
            # Create chart grid
            chart_grid = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
            
            # Plot positions
            zero_line = chart_height // 2
            
            for i, pos in enumerate(sampled_positions):
                if i < chart_width:
                    # Map position (-1 to 1) to chart height
                    y_offset = int(pos * (chart_height // 2 - 1))
                    y_pos = zero_line - y_offset
                    y_pos = max(0, min(chart_height - 1, y_pos))
                    
                    # Choose character based on position
                    if pos > 0.1:
                        char = chars['position_pos']
                    elif pos < -0.1:
                        char = chars['position_neg']
                    else:
                        char = chars['position_zero']
                    
                    chart_grid[y_pos][i] = char
            
            # Add zero line
            for i in range(chart_width):
                if chart_grid[zero_line][i] == ' ':
                    chart_grid[zero_line][i] = chars['horizontal']
            
            # Render chart with y-axis labels
            for i, row in enumerate(chart_grid):
                # Calculate position value for this row
                pos_value = 1.0 - (i / (chart_height - 1)) * 2.0
                pos_label = f"{pos_value:6.2f}"
                
                # Add border
                if i == 0:
                    line = f"{pos_label} {chars['corner_tl']}" + "".join(row) + chars['corner_tr']
                elif i == chart_height - 1:
                    line = f"{pos_label} {chars['corner_bl']}" + "".join(row) + chars['corner_br']
                else:
                    line = f"{pos_label} {chars['vertical']}" + "".join(row) + chars['vertical']
                
                chart_lines.append(line)
            
        else:
            # Multi-symbol strategy
            chart_lines.append("Multi-symbol strategy positions:")
            chart_lines.append("")
            
            for i, symbol in enumerate(self._dataset.symbols):
                if i < strategy_results.shape[1]:
                    positions = strategy_results[:, i]
                    avg_pos = np.mean(positions)
                    current_pos = positions[-1] if len(positions) > 0 else 0
                    
                    # Create simple bar representation
                    bar_length = 20
                    pos_bar = self._create_position_bar(current_pos, bar_length, chars)
                    
                    chart_lines.append(f"{symbol:>10}: {pos_bar} ({current_pos:+6.2f})")
        
        # Add legend
        if options.get('show_legend', True):
            chart_lines.append("")
            chart_lines.append("Legend:")
            chart_lines.append(f"  {chars['position_pos']} Long position")
            chart_lines.append(f"  {chars['position_neg']} Short position")
            chart_lines.append(f"  {chars['position_zero']} Neutral position")
            chart_lines.append(f"  {chars['horizontal']} Zero line")
        
        return "\n".join(chart_lines)
    
    def _render_summary(self, options: Dict[str, Any]) -> str:
        """Render ASCII summary of the dataset."""
        lines = []
        
        # Title
        lines.append("Dataset Summary")
        lines.append("=" * 50)
        lines.append("")
        
        # Basic information
        lines.append(f"Symbols: {', '.join(self._dataset.symbols)}")
        lines.append(f"Data points: {len(self._dataset)}")
        
        if len(self._dataset) > 0:
            start_date = pd.to_datetime(self._dataset.timestamps[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(self._dataset.timestamps[-1]).strftime('%Y-%m-%d')
            lines.append(f"Date range: {start_date} to {end_date}")
        
        lines.append(f"Has strategy results: {'Yes' if self._dataset.strategy_results is not None else 'No'}")
        lines.append("")
        
        # Price statistics for each symbol
        lines.append("Price Statistics:")
        lines.append("-" * 30)
        
        for i, symbol in enumerate(self._dataset.symbols):
            data_array = self._dataset.data_arrays[i]
            
            # Extract prices
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Calculate statistics
            min_price = np.min(prices)
            max_price = np.max(prices)
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            lines.append(f"{symbol}:")
            lines.append(f"  Min:  {min_price:10.2f}")
            lines.append(f"  Max:  {max_price:10.2f}")
            lines.append(f"  Mean: {mean_price:10.2f}")
            lines.append(f"  Std:  {std_price:10.2f}")
            lines.append("")
        
        # Strategy statistics if available
        if self._dataset.strategy_results is not None:
            lines.append("Strategy Statistics:")
            lines.append("-" * 30)
            
            strategy_results = self._dataset.strategy_results
            
            if strategy_results.ndim == 1:
                # Single strategy
                lines.append("Single strategy results:")
                lines.append(f"  Min position:  {np.min(strategy_results):8.2f}")
                lines.append(f"  Max position:  {np.max(strategy_results):8.2f}")
                lines.append(f"  Mean position: {np.mean(strategy_results):8.2f}")
                
                # Count signals
                buy_signals = np.sum(strategy_results > 0.5)
                sell_signals = np.sum(strategy_results < -0.5)
                neutral = len(strategy_results) - buy_signals - sell_signals
                
                lines.append(f"  Buy signals:   {buy_signals:8d}")
                lines.append(f"  Sell signals:  {sell_signals:8d}")
                lines.append(f"  Neutral:       {neutral:8d}")
            else:
                # Multi-symbol strategy
                lines.append("Multi-symbol strategy results:")
                for i, symbol in enumerate(self._dataset.symbols):
                    if i < strategy_results.shape[1]:
                        positions = strategy_results[:, i]
                        lines.append(f"  {symbol}:")
                        lines.append(f"    Mean position: {np.mean(positions):8.2f}")
                        lines.append(f"    Current:       {positions[-1]:8.2f}")
            
            lines.append("")
        
        # Metadata if available
        if self._dataset.metadata:
            lines.append("Metadata:")
            lines.append("-" * 30)
            for key, value in self._dataset.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_table(self, options: Dict[str, Any]) -> str:
        """Render ASCII table of the dataset."""
        lines = []
        
        # Convert to DataFrame for easier table rendering
        df = self._dataset.to_dataframe()
        
        # Limit rows for display
        max_rows = options.get('max_rows', 20)
        if len(df) > max_rows:
            # Show first and last rows
            half_rows = max_rows // 2
            df_display = pd.concat([df.head(half_rows), df.tail(half_rows)])
            show_ellipsis = True
        else:
            df_display = df
            show_ellipsis = False
        
        # Title
        lines.append("Dataset Table")
        lines.append("=" * 50)
        lines.append("")
        
        # Convert DataFrame to string with limited precision
        pd.set_option('display.float_format', lambda x: f'{x:.{options.get("precision", 2)}f}')
        table_str = str(df_display)
        
        lines.extend(table_str.split('\n'))
        
        if show_ellipsis:
            lines.insert(-len(df.tail(max_rows // 2)), "... (rows omitted) ...")
        
        lines.append("")
        lines.append(f"Total rows: {len(df)}")
        
        return "\n".join(lines)
    
    def _create_position_bar(self, position: float, length: int, chars: Dict[str, str]) -> str:
        """Create ASCII bar representation of position."""
        # Clamp position to [-1, 1]
        position = max(-1, min(1, position))
        
        # Calculate bar components
        zero_pos = length // 2
        bar_chars = [' '] * length
        
        if position > 0:
            # Positive position - fill from center to right
            fill_length = int(position * zero_pos)
            for i in range(zero_pos, min(length, zero_pos + fill_length)):
                bar_chars[i] = chars['position_pos']
        elif position < 0:
            # Negative position - fill from center to left
            fill_length = int(abs(position) * zero_pos)
            for i in range(max(0, zero_pos - fill_length), zero_pos):
                bar_chars[i] = chars['position_neg']
        
        # Mark center
        bar_chars[zero_pos] = '|'
        
        return ''.join(bar_chars)
    
    def _save_output(self, rendered_output: str, filepath: str, **kwargs) -> None:
        """Save ASCII output to text file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(rendered_output)
    
    def render_emulator_summary(self, emulator, **kwargs) -> str:
        """
        Render ASCII summary of emulator results.
        
        Args:
            emulator: WarspiteTradingEmulator instance with completed simulation
            **kwargs: Additional rendering options
            
        Returns:
            ASCII text summary of emulator performance
        """
        lines = []
        
        # Title
        lines.append("Trading Emulation Summary")
        lines.append("=" * 50)
        lines.append("")
        
        # Basic metrics
        if hasattr(emulator, 'portfolio_history') and len(emulator.portfolio_history) >= 2:
            initial_value = emulator.portfolio_history[0]
            final_value = emulator.portfolio_history[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            lines.append(f"Initial Capital:    ${initial_value:12,.2f}")
            lines.append(f"Final Value:        ${final_value:12,.2f}")
            lines.append(f"Total Return:       {total_return:12.2f}%")
            lines.append("")
            
            # Performance metrics
            metrics = emulator.get_performance_metrics()
            lines.append("Performance Metrics:")
            lines.append("-" * 30)
            lines.append(f"Volatility:         {metrics.get('volatility', 0)*100:12.2f}%")
            lines.append(f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):12.2f}")
            lines.append(f"Max Drawdown:       {metrics.get('max_drawdown', 0)*100:12.2f}%")
            lines.append(f"Win Rate:           {metrics.get('win_rate', 0)*100:12.1f}%")
            lines.append(f"Total Trades:       {metrics.get('total_trades', 0):12d}")
            lines.append("")
            
            # Portfolio chart
            chart_width = kwargs.get('width', 60)
            chart_height = kwargs.get('height', 10)
            
            lines.append("Portfolio Value Chart:")
            lines.append("-" * 30)
            
            portfolio_chart = self._create_portfolio_chart(
                emulator.portfolio_history, chart_width, chart_height, kwargs
            )
            lines.extend(portfolio_chart)
            lines.append("")
        
        # Current positions
        if hasattr(emulator, '_positions'):
            lines.append("Current Positions:")
            lines.append("-" * 30)
            
            for symbol, position in emulator.positions.items():
                if abs(position) > 1e-6:  # Only show non-zero positions
                    lines.append(f"{symbol:>10}: {position:+10.4f}")
            
            lines.append("")
        
        # Recent trades
        if hasattr(emulator, 'trade_history') and emulator.trade_history:
            lines.append("Recent Trades (last 10):")
            lines.append("-" * 50)
            
            recent_trades = emulator.trade_history[-10:]
            for trade in recent_trades:
                timestamp = trade.timestamp.strftime('%Y-%m-%d %H:%M')
                action = trade.action.upper()
                lines.append(f"{timestamp} {action:>4} {trade.quantity:8.2f} {trade.symbol:>6} @ {trade.price:8.2f}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_portfolio_chart(self, portfolio_history: List[float], 
                              width: int, height: int, options: Dict[str, Any]) -> List[str]:
        """Create ASCII chart of portfolio performance."""
        if len(portfolio_history) < 2:
            return ["Insufficient data for chart"]
        
        chars = options.get('chart_chars', self._style_options['chart_chars'])
        
        # Sample data to fit width
        if len(portfolio_history) > width:
            indices = np.linspace(0, len(portfolio_history) - 1, width, dtype=int)
            sampled_values = [portfolio_history[i] for i in indices]
        else:
            sampled_values = portfolio_history
        
        # Normalize to chart height
        min_val = min(sampled_values)
        max_val = max(sampled_values)
        val_range = max_val - min_val
        
        if val_range == 0:
            val_range = 1
        
        # Create chart grid
        chart_grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot values
        for i, value in enumerate(sampled_values):
            if i < width:
                y_pos = int((max_val - value) / val_range * (height - 1))
                y_pos = max(0, min(height - 1, y_pos))
                chart_grid[y_pos][i] = chars['price_char']
        
        # Render chart
        chart_lines = []
        for i, row in enumerate(chart_grid):
            # Calculate value for this row
            row_value = max_val - (i / (height - 1)) * val_range
            value_label = f"{row_value:10,.0f}"
            
            # Add border
            if i == 0:
                line = f"{value_label} {chars['corner_tl']}" + "".join(row) + chars['corner_tr']
            elif i == height - 1:
                line = f"{value_label} {chars['corner_bl']}" + "".join(row) + chars['corner_br']
            else:
                line = f"{value_label} {chars['vertical']}" + "".join(row) + chars['vertical']
            
            chart_lines.append(line)
        
        return chart_lines