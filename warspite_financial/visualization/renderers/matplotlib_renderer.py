"""
Matplotlib renderer for warspite_financial visualization.

This module provides matplotlib-based visualization for datasets, strategies, and trading results.
"""

from typing import Any, Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime

from .base import WarspiteDatasetRenderer


class MatplotlibRenderer(WarspiteDatasetRenderer):
    """
    Matplotlib-based renderer for WarspiteDataset visualization.
    
    This renderer creates comprehensive charts showing price data, strategy signals,
    trade points, and portfolio performance using matplotlib.
    """
    
    def __init__(self, dataset):
        """
        Initialize matplotlib renderer.
        
        Args:
            dataset: WarspiteDataset instance to render
        """
        super().__init__(dataset)
        
        # Default style options
        self._style_options = {
            'figsize': (12, 8),
            'dpi': 100,
            'style': 'default',
            'color_scheme': 'default',
            'style_theme': 'default',
            'show_grid': True,
            'show_legend': True,
            'title': None,
            'date_format': '%Y-%m-%d',
            'line_style': '-',
            'marker_style': None,
            'transparency': 1.0,
            'background_color': 'white'
        }
        
        # Color schemes
        self._color_schemes = {
            'default': {
                'price': '#1f77b4',
                'volume': '#ff7f0e', 
                'buy_signal': '#2ca02c',
                'sell_signal': '#d62728',
                'position': '#9467bd',
                'portfolio': '#8c564b'
            },
            'dark': {
                'price': '#00d4ff',
                'volume': '#ff9500',
                'buy_signal': '#00ff41',
                'sell_signal': '#ff0040',
                'position': '#bf00ff',
                'portfolio': '#ffbf00'
            },
            'minimal': {
                'price': '#333333',
                'volume': '#666666',
                'buy_signal': '#4CAF50',
                'sell_signal': '#F44336',
                'position': '#9C27B0',
                'portfolio': '#FF9800'
            },
            'professional': {
                'price': '#2E86AB',
                'volume': '#A23B72',
                'buy_signal': '#F18F01',
                'sell_signal': '#C73E1D',
                'position': '#592E83',
                'portfolio': '#1B998B'
            },
            'high_contrast': {
                'price': '#000000',
                'volume': '#808080',
                'buy_signal': '#008000',
                'sell_signal': '#FF0000',
                'position': '#800080',
                'portfolio': '#000080'
            }
        }
        
        # Style themes
        self._style_themes = {
            'default': {
                'grid_alpha': 0.3,
                'line_width': 1.5,
                'marker_size': 50,
                'font_size': 10,
                'title_size': 14,
                'legend_size': 9
            },
            'presentation': {
                'grid_alpha': 0.2,
                'line_width': 2.0,
                'marker_size': 60,
                'font_size': 12,
                'title_size': 16,
                'legend_size': 11
            },
            'print': {
                'grid_alpha': 0.5,
                'line_width': 1.0,
                'marker_size': 40,
                'font_size': 9,
                'title_size': 12,
                'legend_size': 8
            }
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return ['png', 'pdf', 'svg', 'jpg', 'eps']
    
    def render(self, **kwargs) -> Figure:
        """
        Render the dataset as a matplotlib figure.
        
        Args:
            **kwargs: Rendering options including:
                - chart_type: 'price', 'strategy', 'portfolio', 'combined'
                - symbols: List of symbols to include (default: all)
                - show_volume: Whether to show volume data
                - show_signals: Whether to show strategy signals
                - show_trades: Whether to show trade points
                - emulator: Optional WarspiteTradingEmulator for portfolio data
                
        Returns:
            matplotlib Figure object
        """
        if not self.validate_dataset_for_rendering():
            raise ValueError("Dataset is not suitable for rendering")
        
        # Merge options
        options = {**self._style_options, **kwargs}
        chart_type = options.get('chart_type', 'combined')
        
        # Set matplotlib style
        if options.get('style') != 'default':
            plt.style.use(options['style'])
        
        # Create figure
        fig = plt.figure(figsize=options['figsize'], dpi=options['dpi'])
        
        # Render based on chart type
        if chart_type == 'price':
            self._render_price_chart(fig, options)
        elif chart_type == 'strategy':
            self._render_strategy_chart(fig, options)
        elif chart_type == 'portfolio':
            self._render_portfolio_chart(fig, options)
        elif chart_type == 'combined':
            self._render_combined_chart(fig, options)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Apply common formatting
        self._apply_common_formatting(fig, options)
        
        return fig
    
    def _render_price_chart(self, fig: Figure, options: Dict[str, Any]) -> None:
        """Render price data chart."""
        symbols = options.get('symbols', self._dataset.symbols)
        show_volume = options.get('show_volume', True)
        colors = self._get_color_scheme(options.get('color_scheme', 'default'))
        theme = self._get_style_theme(options.get('style_theme', 'default'))
        
        # Create subplots
        if show_volume:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_price = fig.add_subplot(gs[0])
            ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
        else:
            ax_price = fig.add_subplot(1, 1, 1)
            ax_volume = None
        
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(self._dataset.timestamps)
        
        # Plot price data for each symbol
        for i, symbol in enumerate(symbols):
            if symbol not in self._dataset.symbols:
                continue
            
            symbol_idx = self._dataset.symbols.index(symbol)
            data_array = self._dataset.data_arrays[symbol_idx]
            
            if data_array.ndim == 1:
                # Single price series
                ax_price.plot(timestamps, data_array, 
                            label=f'{symbol} Price', 
                            color=colors['price'],
                            linewidth=theme.get('line_width', 1.5),
                            linestyle=options.get('line_style', '-'),
                            alpha=options.get('transparency', 1.0))
            else:
                # OHLCV data - plot candlestick or line
                if data_array.shape[1] >= 4:  # Has OHLC
                    # Simple line chart using close prices
                    close_prices = data_array[:, 3]
                    ax_price.plot(timestamps, close_prices,
                                label=f'{symbol} Close',
                                color=colors['price'],
                                linewidth=theme.get('line_width', 1.5),
                                linestyle=options.get('line_style', '-'),
                                alpha=options.get('transparency', 1.0))
                    
                    # Plot volume if available and requested
                    if show_volume and ax_volume is not None and data_array.shape[1] >= 5:
                        volume = data_array[:, 4]
                        ax_volume.bar(timestamps, volume,
                                    label=f'{symbol} Volume',
                                    color=colors['volume'],
                                    alpha=0.7,
                                    width=1)
                else:
                    # Plot first column as price
                    ax_price.plot(timestamps, data_array[:, 0],
                                label=f'{symbol} Price',
                                color=colors['price'],
                                linewidth=theme.get('line_width', 1.5),
                                linestyle=options.get('line_style', '-'),
                                alpha=options.get('transparency', 1.0))
        
        # Apply style theme
        self._apply_style_theme(ax_price, theme, options)
        
        # Format price axis
        ax_price.set_ylabel('Price', fontsize=theme.get('font_size', 10))
        
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_price.get_legend_handles_labels()
            if handles and labels:
                ax_price.legend(loc='upper left', fontsize=theme.get('legend_size', 9))
        
        # Format volume axis
        if ax_volume is not None:
            self._apply_style_theme(ax_volume, theme, options)
            ax_volume.set_ylabel('Volume', fontsize=theme.get('font_size', 10))
            if options.get('show_legend', True):
                # Only create legend if there are labeled plots
                handles, labels = ax_volume.get_legend_handles_labels()
                if handles and labels:
                    ax_volume.legend(loc='upper left', fontsize=theme.get('legend_size', 9))
    
    def _render_strategy_chart(self, fig: Figure, options: Dict[str, Any]) -> None:
        """Render strategy signals and positions."""
        colors = self._get_color_scheme(options.get('color_scheme', 'default'))
        
        # Check if we have strategy results to determine layout
        has_strategy_results = self._dataset.strategy_results is not None
        
        if has_strategy_results:
            # Create subplots for price and positions
            gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.2)
            ax_price = fig.add_subplot(gs[0])
            ax_position = fig.add_subplot(gs[1], sharex=ax_price)
        else:
            # Only create price chart if no strategy results
            ax_price = fig.add_subplot(1, 1, 1)
            ax_position = None
        
        timestamps = pd.to_datetime(self._dataset.timestamps)
        
        # Plot price data for all symbols
        symbols = options.get('symbols', self._dataset.symbols)
        for i, symbol in enumerate(symbols):
            if symbol not in self._dataset.symbols:
                continue
                
            symbol_idx = self._dataset.symbols.index(symbol)
            data_array = self._dataset.data_arrays[symbol_idx]
            
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Use different colors for different symbols
            color = colors['price'] if i == 0 else plt.cm.tab10(i % 10)
            
            ax_price.plot(timestamps, prices,
                        label=f'{symbol} Price',
                        color=color,
                        linewidth=1.5)
        
        # Plot strategy results if available
        if has_strategy_results and ax_position is not None:
            strategy_results = self._dataset.strategy_results
            
            if strategy_results.ndim == 1:
                # Single strategy results - plot signals on first symbol's price
                ax_position.plot(timestamps, strategy_results,
                               label='Position',
                               color=colors['position'],
                               linewidth=2)
                
                # Get first symbol's prices for signal plotting
                if self._dataset.symbols:
                    first_symbol_data = self._dataset.data_arrays[0]
                    if first_symbol_data.ndim == 1:
                        first_symbol_prices = first_symbol_data
                    else:
                        first_symbol_prices = first_symbol_data[:, 3] if first_symbol_data.shape[1] > 3 else first_symbol_data[:, -1]
                    
                    # Mark buy/sell signals
                    buy_signals = (strategy_results > 0.5)
                    sell_signals = (strategy_results < -0.5)
                    
                    if np.any(buy_signals):
                        ax_price.scatter(timestamps[buy_signals], first_symbol_prices[buy_signals],
                                       color=colors['buy_signal'], marker='^',
                                       s=50, label='Buy Signal', zorder=5)
                    
                    if np.any(sell_signals):
                        ax_price.scatter(timestamps[sell_signals], first_symbol_prices[sell_signals],
                                       color=colors['sell_signal'], marker='v',
                                       s=50, label='Sell Signal', zorder=5)
            else:
                # Multi-symbol strategy results - use different colors for each symbol
                for i, symbol in enumerate(self._dataset.symbols):
                    if i < strategy_results.shape[1]:
                        # Use different colors for different symbols
                        position_color = colors['position'] if i == 0 else plt.cm.tab10(i % 10)
                        ax_position.plot(timestamps, strategy_results[:, i],
                                       label=f'{symbol} Position',
                                       color=position_color,
                                       linewidth=1.5)
        
        # Format axes
        ax_price.set_ylabel('Price')
        ax_price.grid(options.get('show_grid', True), alpha=0.3)
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_price.get_legend_handles_labels()
            if handles and labels:
                ax_price.legend(loc='upper left')
        
        # Format position axis only if it exists
        if ax_position is not None:
            ax_position.set_ylabel('Position')
            ax_position.set_ylim(-1.1, 1.1)
            ax_position.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_position.grid(options.get('show_grid', True), alpha=0.3)
            if options.get('show_legend', True):
                # Only create legend if there are labeled plots
                handles, labels = ax_position.get_legend_handles_labels()
                if handles and labels:
                    ax_position.legend(loc='upper left')
    
    def _render_portfolio_chart(self, fig: Figure, options: Dict[str, Any]) -> None:
        """Render portfolio performance chart."""
        emulator = options.get('emulator')
        if emulator is None:
            raise ValueError("Portfolio chart requires an emulator instance")
        
        colors = self._get_color_scheme(options.get('color_scheme', 'default'))
        
        # Get portfolio history
        portfolio_history = emulator.portfolio_history
        if len(portfolio_history) < 2:
            raise ValueError("Insufficient portfolio history for rendering")
        
        # Create subplots
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)
        ax_portfolio = fig.add_subplot(gs[0])
        ax_returns = fig.add_subplot(gs[1], sharex=ax_portfolio)
        ax_drawdown = fig.add_subplot(gs[2], sharex=ax_portfolio)
        
        # Get timestamps (should match portfolio history length)
        if hasattr(emulator, '_timestamp_history') and emulator._timestamp_history:
            timestamps = emulator._timestamp_history
        else:
            # Fallback to dataset timestamps
            timestamps = pd.to_datetime(self._dataset.timestamps[:len(portfolio_history)])
        
        # Ensure timestamps match portfolio history
        if len(timestamps) != len(portfolio_history):
            timestamps = timestamps[:len(portfolio_history)]
        
        # Calculate cumulative returns instead of absolute portfolio value
        portfolio_array = np.array(portfolio_history)
        initial_value = portfolio_array[0]
        cumulative_returns = (portfolio_array - initial_value) / initial_value * 100
        
        # Plot cumulative returns
        ax_portfolio.plot(timestamps, cumulative_returns,
                        label='Cumulative Returns (%)',
                        color=colors['portfolio'],
                        linewidth=2)
        
        # Add zero line for reference
        ax_portfolio.axhline(y=0, color='gray', 
                           linestyle='--', alpha=0.7, label='Break-even')
        
        # Calculate and plot returns
        portfolio_array = np.array(portfolio_history)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        returns_timestamps = timestamps[1:]  # One less than portfolio values
        
        ax_returns.plot(returns_timestamps, returns * 100,
                      label='Daily Returns (%)',
                      color=colors['price'],
                      linewidth=1)
        ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Calculate and plot drawdown
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak * 100
        
        ax_drawdown.fill_between(timestamps, drawdown, 0,
                               color=colors['sell_signal'],
                               alpha=0.3, label='Drawdown (%)')
        ax_drawdown.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark trades on cumulative returns chart
        trade_history = emulator.trade_history
        if trade_history:
            buy_trades = [t for t in trade_history if t.action == 'buy']
            sell_trades = [t for t in trade_history if t.action == 'sell']
            
            if buy_trades:
                buy_times = [t.timestamp for t in buy_trades]
                # Convert portfolio values to cumulative returns for trade markers
                buy_returns = [(t.portfolio_value - initial_value) / initial_value * 100 for t in buy_trades]
                ax_portfolio.scatter(buy_times, buy_returns,
                                   color=colors['buy_signal'], marker='^',
                                   s=30, alpha=0.7, label='Buy Trades')
            
            if sell_trades:
                sell_times = [t.timestamp for t in sell_trades]
                # Convert portfolio values to cumulative returns for trade markers
                sell_returns = [(t.portfolio_value - initial_value) / initial_value * 100 for t in sell_trades]
                ax_portfolio.scatter(sell_times, sell_returns,
                                   color=colors['sell_signal'], marker='v',
                                   s=30, alpha=0.7, label='Sell Trades')
        
        # Format axes
        ax_portfolio.set_ylabel('Cumulative Returns (%)')
        ax_portfolio.grid(options.get('show_grid', True), alpha=0.3)
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_portfolio.get_legend_handles_labels()
            if handles and labels:
                ax_portfolio.legend(loc='upper left')
        
        ax_returns.set_ylabel('Returns (%)')
        ax_returns.grid(options.get('show_grid', True), alpha=0.3)
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_returns.get_legend_handles_labels()
            if handles and labels:
                ax_returns.legend(loc='upper left')
        
        ax_drawdown.set_ylabel('Drawdown (%)')
        ax_drawdown.grid(options.get('show_grid', True), alpha=0.3)
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_drawdown.get_legend_handles_labels()
            if handles and labels:
                ax_drawdown.legend(loc='upper left')
    
    def _render_combined_chart(self, fig: Figure, options: Dict[str, Any]) -> None:
        """Render combined chart with price, strategy, and portfolio data."""
        emulator = options.get('emulator')
        has_strategy_results = self._dataset.strategy_results is not None
        
        if emulator is not None:
            # Full combined chart with portfolio
            if has_strategy_results:
                gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.2)
                ax_price = fig.add_subplot(gs[0])
                ax_position = fig.add_subplot(gs[1], sharex=ax_price)
                ax_portfolio = fig.add_subplot(gs[2], sharex=ax_price)
                ax_returns = fig.add_subplot(gs[3], sharex=ax_price)
            else:
                # No strategy results - skip position chart
                gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)
                ax_price = fig.add_subplot(gs[0])
                ax_position = None
                ax_portfolio = fig.add_subplot(gs[1], sharex=ax_price)
                ax_returns = fig.add_subplot(gs[2], sharex=ax_price)
        else:
            # Price and strategy only
            if has_strategy_results:
                gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.2)
                ax_price = fig.add_subplot(gs[0])
                ax_position = fig.add_subplot(gs[1], sharex=ax_price)
            else:
                # Only price chart
                ax_price = fig.add_subplot(1, 1, 1)
                ax_position = None
            ax_portfolio = None
            ax_returns = None
        
        colors = self._get_color_scheme(options.get('color_scheme', 'default'))
        timestamps = pd.to_datetime(self._dataset.timestamps)
        
        # Plot price data for all symbols
        symbols = options.get('symbols', self._dataset.symbols)
        for i, symbol in enumerate(symbols):
            if symbol not in self._dataset.symbols:
                continue
                
            symbol_idx = self._dataset.symbols.index(symbol)
            data_array = self._dataset.data_arrays[symbol_idx]
            
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Use different colors for different symbols
            color = colors['price'] if i == 0 else plt.cm.tab10(i % 10)
            
            ax_price.plot(timestamps, prices,
                        label=f'{symbol} Price',
                        color=color,
                        linewidth=1.5)
        
        # Plot strategy positions
        if has_strategy_results and ax_position is not None:
            strategy_results = self._dataset.strategy_results
            
            if strategy_results.ndim == 1:
                ax_position.plot(timestamps, strategy_results,
                               label='Position',
                               color=colors['position'],
                               linewidth=2)
                
                # Get first symbol's prices for signal plotting
                if self._dataset.symbols:
                    first_symbol_data = self._dataset.data_arrays[0]
                    if first_symbol_data.ndim == 1:
                        first_symbol_prices = first_symbol_data
                    else:
                        first_symbol_prices = first_symbol_data[:, 3] if first_symbol_data.shape[1] > 3 else first_symbol_data[:, -1]
                    
                    # Mark signals on price chart
                    buy_signals = (strategy_results > 0.5)
                    sell_signals = (strategy_results < -0.5)
                    
                    if np.any(buy_signals):
                        ax_price.scatter(timestamps[buy_signals], first_symbol_prices[buy_signals],
                                       color=colors['buy_signal'], marker='^',
                                       s=50, label='Buy Signal', zorder=5)
                    
                    if np.any(sell_signals):
                        ax_price.scatter(timestamps[sell_signals], first_symbol_prices[sell_signals],
                                       color=colors['sell_signal'], marker='v',
                                       s=50, label='Sell Signal', zorder=5)
            else:
                # Multi-symbol strategy results - use different colors for each symbol
                for i, symbol in enumerate(self._dataset.symbols):
                    if i < strategy_results.shape[1]:
                        # Use different colors for different symbols
                        position_color = colors['position'] if i == 0 else plt.cm.tab10(i % 10)
                        ax_position.plot(timestamps, strategy_results[:, i],
                                       label=f'{symbol} Position',
                                       color=position_color,
                                       linewidth=2)
                        
                        # Mark signals on price chart for first symbol only
                        if i == 0:
                            first_symbol_data = self._dataset.data_arrays[0]
                            if first_symbol_data.ndim == 1:
                                first_symbol_prices = first_symbol_data
                            else:
                                first_symbol_prices = first_symbol_data[:, 3] if first_symbol_data.shape[1] > 3 else first_symbol_data[:, -1]
                            
                            buy_signals = (strategy_results[:, i] > 0.5)
                            sell_signals = (strategy_results[:, i] < -0.5)
                            
                            if np.any(buy_signals):
                                ax_price.scatter(timestamps[buy_signals], first_symbol_prices[buy_signals],
                                               color=colors['buy_signal'], marker='^',
                                               s=50, label='Buy Signal', zorder=5)
                            
                            if np.any(sell_signals):
                                ax_price.scatter(timestamps[sell_signals], first_symbol_prices[sell_signals],
                                               color=colors['sell_signal'], marker='v',
                                               s=50, label='Sell Signal', zorder=5)
        
        # Plot portfolio data if available
        if emulator is not None and ax_portfolio is not None:
            portfolio_history = emulator.portfolio_history
            
            if len(portfolio_history) >= 2:
                # Get matching timestamps with robust alignment
                if hasattr(emulator, '_timestamp_history') and emulator._timestamp_history:
                    portfolio_timestamps = emulator._timestamp_history
                else:
                    portfolio_timestamps = timestamps[:len(portfolio_history)]
                
                # Robust length alignment - handle both cases where timestamps or portfolio_history is longer
                min_length = min(len(portfolio_timestamps), len(portfolio_history))
                if min_length >= 2:  # Need at least 2 points to plot
                    portfolio_timestamps_aligned = portfolio_timestamps[:min_length]
                    portfolio_history_aligned = portfolio_history[:min_length]
                    
                    # Calculate cumulative returns instead of absolute portfolio value
                    portfolio_array = np.array(portfolio_history_aligned)
                    initial_value = portfolio_array[0]
                    cumulative_returns = (portfolio_array - initial_value) / initial_value * 100
                    
                    ax_portfolio.plot(portfolio_timestamps_aligned, cumulative_returns,
                                    label='Cumulative Returns (%)',
                                    color=colors['portfolio'],
                                    linewidth=2)
                    
                    # Add zero line for reference
                    ax_portfolio.axhline(y=0, color='gray', 
                                       linestyle='--', alpha=0.7, label='Break-even')
                else:
                    # Not enough data points to plot portfolio
                    portfolio_timestamps_aligned = []
                    portfolio_history_aligned = []
                
                # Calculate returns for bottom panel
                if ax_returns is not None and len(portfolio_history_aligned) >= 2:
                    portfolio_array = np.array(portfolio_history_aligned)
                    returns = np.diff(portfolio_array) / portfolio_array[:-1]
                    returns_timestamps = portfolio_timestamps_aligned[1:]
                    
                    # Ensure returns and timestamps have matching lengths
                    min_returns_length = min(len(returns_timestamps), len(returns))
                    if min_returns_length > 0:
                        returns_timestamps_aligned = returns_timestamps[:min_returns_length]
                        returns_aligned = returns[:min_returns_length]
                        
                        ax_returns.plot(returns_timestamps_aligned, returns_aligned * 100,
                                      label='Daily Returns (%)',
                                      color=colors['price'],
                                      linewidth=1)
                        ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format all axes
        ax_price.set_ylabel('Price')
        ax_price.grid(options.get('show_grid', True), alpha=0.3)
        if options.get('show_legend', True):
            # Only create legend if there are labeled plots
            handles, labels = ax_price.get_legend_handles_labels()
            if handles and labels:
                ax_price.legend(loc='upper left')
        
        if ax_position is not None:
            ax_position.set_ylabel('Position')
            ax_position.set_ylim(-1.1, 1.1)
            ax_position.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_position.grid(options.get('show_grid', True), alpha=0.3)
            if options.get('show_legend', True):
                # Only create legend if there are labeled plots
                handles, labels = ax_position.get_legend_handles_labels()
                if handles and labels:
                    ax_position.legend(loc='upper left')
        
        if ax_portfolio is not None:
            ax_portfolio.set_ylabel('Cumulative Returns (%)')
            ax_portfolio.grid(options.get('show_grid', True), alpha=0.3)
            if options.get('show_legend', True):
                # Only create legend if there are labeled plots
                handles, labels = ax_portfolio.get_legend_handles_labels()
                if handles and labels:
                    ax_portfolio.legend(loc='upper left')
        
        if ax_returns is not None:
            ax_returns.set_ylabel('Returns (%)')
            ax_returns.grid(options.get('show_grid', True), alpha=0.3)
            if options.get('show_legend', True):
                # Only create legend if there are labeled plots
                handles, labels = ax_returns.get_legend_handles_labels()
                if handles and labels:
                    ax_returns.legend(loc='upper left')
    
    def _apply_common_formatting(self, fig: Figure, options: Dict[str, Any]) -> None:
        """Apply common formatting to the figure."""
        # Set title
        title = options.get('title')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Format x-axis (dates) for all subplots
        for ax in fig.get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter(options.get('date_format', '%Y-%m-%d')))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            
            # Rotate date labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Add timestamp
        fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='right', va='bottom', fontsize=8, alpha=0.7)
    
    def _get_color_scheme(self, scheme_name: str) -> Dict[str, str]:
        """Get color scheme by name."""
        return self._color_schemes.get(scheme_name, self._color_schemes['default'])
    
    def _get_style_theme(self, theme_name: str) -> Dict[str, Any]:
        """Get style theme by name."""
        return self._style_themes.get(theme_name, self._style_themes['default'])
    
    def _apply_style_theme(self, ax: Axes, theme: Dict[str, Any], options: Dict[str, Any]) -> None:
        """Apply style theme to axes."""
        # Apply grid styling
        if options.get('show_grid', True):
            ax.grid(True, alpha=theme.get('grid_alpha', 0.3))
        
        # Apply font sizes
        ax.tick_params(labelsize=theme.get('font_size', 10))
        
        # Apply background color
        bg_color = options.get('background_color', 'white')
        ax.set_facecolor(bg_color)
    
    def set_custom_color_scheme(self, name: str, colors: Dict[str, str]) -> None:
        """
        Add a custom color scheme.
        
        Args:
            name: Name for the custom color scheme
            colors: Dictionary mapping color roles to hex colors
        """
        required_colors = ['price', 'volume', 'buy_signal', 'sell_signal', 'position', 'portfolio']
        
        # Validate that all required colors are provided
        for color_role in required_colors:
            if color_role not in colors:
                raise ValueError(f"Color scheme must include '{color_role}' color")
        
        self._color_schemes[name] = colors
    
    def set_custom_style_theme(self, name: str, theme: Dict[str, Any]) -> None:
        """
        Add a custom style theme.
        
        Args:
            name: Name for the custom style theme
            theme: Dictionary with style parameters
        """
        self._style_themes[name] = theme
    
    def _save_output(self, rendered_output: Figure, filepath: str, **kwargs) -> None:
        """Save matplotlib figure to file."""
        # Determine format from file extension
        format_ext = filepath.split('.')[-1].lower()
        if format_ext not in self.get_supported_formats():
            format_ext = 'png'  # Default format
        
        # Save with appropriate options
        save_options = {
            'dpi': kwargs.get('dpi', self._style_options.get('dpi', 100)),
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        rendered_output.savefig(filepath, format=format_ext, **save_options)
        
    def create_performance_summary(self, emulator, **kwargs) -> Figure:
        """
        Create a comprehensive performance summary chart.
        
        Args:
            emulator: WarspiteTradingEmulator instance with completed simulation
            **kwargs: Additional rendering options
            
        Returns:
            matplotlib Figure with performance summary
        """
        if not hasattr(emulator, 'portfolio_history') or len(emulator.portfolio_history) < 2:
            raise ValueError("Emulator must have completed simulation with portfolio history")
        
        # Get performance metrics
        metrics = emulator.get_performance_metrics()
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        colors = self._get_color_scheme(kwargs.get('color_scheme', 'default'))
        
        # Main portfolio chart (top left, spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        
        # Get timestamps and portfolio values with robust alignment
        portfolio_history = emulator.portfolio_history
        if hasattr(emulator, '_timestamp_history') and emulator._timestamp_history:
            timestamps = emulator._timestamp_history
        else:
            timestamps = pd.to_datetime(self._dataset.timestamps[:len(portfolio_history)])
        
        # Robust length alignment - handle both cases where timestamps or portfolio_history is longer
        min_length = min(len(timestamps), len(portfolio_history))
        if min_length >= 2:  # Need at least 2 points to plot
            timestamps_aligned = timestamps[:min_length]
            portfolio_history_aligned = portfolio_history[:min_length]
            
            ax_main.plot(timestamps_aligned, portfolio_history_aligned, 
                        color=colors['portfolio'], linewidth=2, label='Portfolio Value')
            ax_main.axhline(y=portfolio_history_aligned[0], color='gray', 
                           linestyle='--', alpha=0.7, label='Initial Capital')
        else:
            # Not enough data points - create empty plot with message
            ax_main.text(0.5, 0.5, 'Insufficient data for portfolio visualization', 
                        ha='center', va='center', transform=ax_main.transAxes)
        
        ax_main.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Portfolio Value')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Performance metrics table (top right)
        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.axis('off')
        
        metrics_text = [
            f"Total Return: {metrics.get('total_return', 0)*100:.2f}%",
            f"Volatility: {metrics.get('volatility', 0)*100:.2f}%",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%",
            f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%",
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Final Value: ${metrics.get('final_portfolio_value', 0):,.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            ax_metrics.text(0.05, 0.9 - i*0.12, text, fontsize=11, 
                          transform=ax_metrics.transAxes, fontweight='bold')
        
        ax_metrics.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        # Returns distribution (bottom left)
        ax_returns = fig.add_subplot(gs[1, 0])
        
        if min_length >= 2:  # Only calculate returns if we have enough data
            portfolio_array = np.array(portfolio_history_aligned)
            returns = np.diff(portfolio_array) / portfolio_array[:-1]
            
            ax_returns.hist(returns * 100, bins=30, color=colors['price'], alpha=0.7, edgecolor='black')
            ax_returns.axvline(x=np.mean(returns) * 100, color=colors['sell_signal'], 
                              linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns)*100:.2f}%')
        else:
            ax_returns.text(0.5, 0.5, 'Insufficient data for returns analysis', 
                           ha='center', va='center', transform=ax_returns.transAxes)
        ax_returns.set_title('Returns Distribution')
        ax_returns.set_xlabel('Daily Returns (%)')
        ax_returns.set_ylabel('Frequency')
        ax_returns.legend()
        ax_returns.grid(True, alpha=0.3)
        
        # Drawdown chart (bottom middle)
        ax_drawdown = fig.add_subplot(gs[1, 1])
        
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak * 100
        
        ax_drawdown.fill_between(timestamps, drawdown, 0, 
                               color=colors['sell_signal'], alpha=0.3)
        ax_drawdown.plot(timestamps, drawdown, color=colors['sell_signal'], linewidth=1)
        ax_drawdown.set_title('Drawdown')
        ax_drawdown.set_ylabel('Drawdown (%)')
        ax_drawdown.grid(True, alpha=0.3)
        
        # Trade analysis (bottom right)
        ax_trades = fig.add_subplot(gs[1, 2])
        
        if emulator.trade_history:
            # Analyze trade profitability
            trade_pnl = []
            buy_trades = {}
            
            for trade in emulator.trade_history:
                if trade.action == 'buy':
                    if trade.symbol not in buy_trades:
                        buy_trades[trade.symbol] = []
                    buy_trades[trade.symbol].append(trade)
                elif trade.action == 'sell':
                    if trade.symbol in buy_trades and buy_trades[trade.symbol]:
                        buy_trade = buy_trades[trade.symbol].pop(0)
                        pnl = (trade.price - buy_trade.price) * trade.quantity - trade.fee - buy_trade.fee
                        trade_pnl.append(pnl)
            
            if trade_pnl:
                winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnl if pnl <= 0]
                
                ax_trades.bar(['Winning', 'Losing'], 
                            [len(winning_trades), len(losing_trades)],
                            color=[colors['buy_signal'], colors['sell_signal']], alpha=0.7)
                ax_trades.set_title('Trade Analysis')
                ax_trades.set_ylabel('Number of Trades')
            else:
                ax_trades.text(0.5, 0.5, 'No completed\ntrades', 
                             ha='center', va='center', transform=ax_trades.transAxes)
                ax_trades.set_title('Trade Analysis')
        else:
            ax_trades.text(0.5, 0.5, 'No trades\nexecuted', 
                         ha='center', va='center', transform=ax_trades.transAxes)
            ax_trades.set_title('Trade Analysis')
        
        # Monthly returns heatmap (bottom row, spans all columns)
        ax_monthly = fig.add_subplot(gs[2, :])
        
        # Calculate monthly returns
        df_portfolio = pd.DataFrame({'value': portfolio_history}, index=timestamps)
        monthly_returns = df_portfolio.resample('M')['value'].last().pct_change().dropna()
        
        if len(monthly_returns) > 1:
            # Create a simple monthly returns chart
            monthly_returns_pct = monthly_returns * 100
            colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns_pct]
            
            ax_monthly.bar(range(len(monthly_returns_pct)), monthly_returns_pct, 
                          color=colors_monthly, alpha=0.7)
            ax_monthly.set_title('Monthly Returns')
            ax_monthly.set_ylabel('Monthly Return (%)')
            ax_monthly.set_xlabel('Month')
            ax_monthly.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_monthly.grid(True, alpha=0.3)
            
            # Set x-axis labels to month names
            month_labels = [ts.strftime('%Y-%m') for ts in monthly_returns.index]
            ax_monthly.set_xticks(range(len(month_labels)))
            ax_monthly.set_xticklabels(month_labels, rotation=45, ha='right')
        else:
            ax_monthly.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                          ha='center', va='center', transform=ax_monthly.transAxes)
            ax_monthly.set_title('Monthly Returns')
        
        # Apply final formatting
        fig.suptitle('Trading Performance Summary', fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout()
        
        return fig