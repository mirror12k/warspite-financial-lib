"""
CSV renderer for warspite_financial visualization.

This module provides CSV export functionality for financial data visualization.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import io

from .base import WarspiteDatasetRenderer


class CSVRenderer(WarspiteDatasetRenderer):
    """
    CSV export renderer for WarspiteDataset visualization.
    
    This renderer exports financial data and analysis results to CSV format
    with customizable formatting and structure options.
    """
    
    def __init__(self, dataset):
        """
        Initialize CSV renderer.
        
        Args:
            dataset: WarspiteDataset instance to render
        """
        super().__init__(dataset)
        
        # CSV-specific style options
        self._style_options = {
            'include_metadata': True,
            'include_strategy_results': True,
            'date_format': '%Y-%m-%d',
            'float_precision': 6,
            'separator': ',',
            'include_headers': True,
            'include_summary': True,
            'multi_sheet': False,  # For Excel-like output
            'flatten_columns': True  # Flatten MultiIndex columns
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return ['csv', 'tsv', 'excel']
    
    def render(self, **kwargs) -> str:
        """
        Render the dataset as CSV text.
        
        Args:
            **kwargs: Rendering options including:
                - include_metadata: Whether to include metadata in output
                - include_strategy_results: Whether to include strategy results
                - date_format: Format for date columns
                - float_precision: Number of decimal places for floats
                - separator: Column separator character
                
        Returns:
            CSV text representation of the dataset
        """
        if not self.validate_dataset_for_rendering():
            raise ValueError("Dataset is not suitable for rendering")
        
        # Merge options
        options = {**self._style_options, **kwargs}
        
        # Convert dataset to DataFrame
        df = self._dataset.to_dataframe()
        
        # Format the DataFrame for CSV output
        formatted_df = self._format_dataframe(df, options)
        
        # Create CSV output
        output_parts = []
        
        # Add metadata header if requested
        if options.get('include_metadata', True):
            metadata_header = self._create_metadata_header(options)
            output_parts.append(metadata_header)
        
        # Add summary if requested
        if options.get('include_summary', True):
            summary = self._create_summary_section(options)
            output_parts.append(summary)
        
        # Add main data
        csv_data = self._dataframe_to_csv(formatted_df, options)
        output_parts.append(csv_data)
        
        return '\n'.join(output_parts)
    
    def _format_dataframe(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Format DataFrame for CSV output."""
        formatted_df = df.copy()
        
        # Flatten MultiIndex columns if requested
        if options.get('flatten_columns', True) and isinstance(df.columns, pd.MultiIndex):
            formatted_df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
        
        # Format date index
        if isinstance(formatted_df.index, pd.DatetimeIndex):
            date_format = options.get('date_format', '%Y-%m-%d')
            formatted_df.index = formatted_df.index.strftime(date_format)
            formatted_df.index.name = 'Date'
        
        # Round float columns to specified precision
        float_precision = options.get('float_precision', 6)
        float_columns = formatted_df.select_dtypes(include=[np.float64, np.float32]).columns
        formatted_df[float_columns] = formatted_df[float_columns].round(float_precision)
        
        return formatted_df
    
    def _create_metadata_header(self, options: Dict[str, Any]) -> str:
        """Create metadata header section."""
        lines = []
        lines.append("# Financial Data Export")
        lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Source: warspite_financial library")
        lines.append("#")
        
        # Dataset information
        lines.append(f"# Dataset Information:")
        lines.append(f"# Symbols: {', '.join(self._dataset.symbols)}")
        lines.append(f"# Data Points: {len(self._dataset)}")
        
        if len(self._dataset) > 0:
            start_date = pd.to_datetime(self._dataset.timestamps[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(self._dataset.timestamps[-1]).strftime('%Y-%m-%d')
            lines.append(f"# Date Range: {start_date} to {end_date}")
        
        lines.append(f"# Has Strategy Results: {'Yes' if self._dataset.strategy_results is not None else 'No'}")
        
        # Custom metadata
        if self._dataset.metadata:
            lines.append("#")
            lines.append("# Custom Metadata:")
            for key, value in self._dataset.metadata.items():
                lines.append(f"# {key}: {value}")
        
        lines.append("#")
        return '\n'.join(lines)
    
    def _create_summary_section(self, options: Dict[str, Any]) -> str:
        """Create summary statistics section."""
        lines = []
        lines.append("# Summary Statistics:")
        lines.append("#")
        
        # Calculate statistics for each symbol
        for i, symbol in enumerate(self._dataset.symbols):
            data_array = self._dataset.data_arrays[i]
            
            # Extract prices (use close prices for OHLCV data)
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Calculate statistics
            stats = {
                'Min': np.min(prices),
                'Max': np.max(prices),
                'Mean': np.mean(prices),
                'Std': np.std(prices),
                'First': prices[0] if len(prices) > 0 else np.nan,
                'Last': prices[-1] if len(prices) > 0 else np.nan
            }
            
            lines.append(f"# {symbol}:")
            for stat_name, stat_value in stats.items():
                lines.append(f"#   {stat_name}: {stat_value:.6f}")
            lines.append("#")
        
        # Strategy statistics if available
        if self._dataset.strategy_results is not None:
            lines.append("# Strategy Results:")
            strategy_results = self._dataset.strategy_results
            
            if strategy_results.ndim == 1:
                lines.append(f"#   Min Position: {np.min(strategy_results):.6f}")
                lines.append(f"#   Max Position: {np.max(strategy_results):.6f}")
                lines.append(f"#   Mean Position: {np.mean(strategy_results):.6f}")
                
                # Count signals
                buy_signals = np.sum(strategy_results > 0.5)
                sell_signals = np.sum(strategy_results < -0.5)
                neutral = len(strategy_results) - buy_signals - sell_signals
                
                lines.append(f"#   Buy Signals: {buy_signals}")
                lines.append(f"#   Sell Signals: {sell_signals}")
                lines.append(f"#   Neutral: {neutral}")
            else:
                lines.append("#   Multi-symbol strategy results:")
                for i, symbol in enumerate(self._dataset.symbols):
                    if i < strategy_results.shape[1]:
                        positions = strategy_results[:, i]
                        lines.append(f"#     {symbol} Mean Position: {np.mean(positions):.6f}")
            
            lines.append("#")
        
        return '\n'.join(lines)
    
    def _dataframe_to_csv(self, df: pd.DataFrame, options: Dict[str, Any]) -> str:
        """Convert DataFrame to CSV string."""
        separator = options.get('separator', ',')
        include_headers = options.get('include_headers', True)
        
        # Use StringIO to capture CSV output
        output = io.StringIO()
        
        df.to_csv(
            output,
            sep=separator,
            header=include_headers,
            index=True,
            float_format=f"%.{options.get('float_precision', 6)}f"
        )
        
        return output.getvalue().strip()
    
    def _save_output(self, rendered_output: str, filepath: str, **kwargs) -> None:
        """Save CSV output to file."""
        # Determine format from file extension
        format_ext = filepath.split('.')[-1].lower()
        
        if format_ext == 'tsv':
            # Replace commas with tabs for TSV format
            rendered_output = rendered_output.replace(',', '\t')
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(rendered_output)
    
    def create_analysis_report(self, emulator=None, **kwargs) -> str:
        """
        Create a comprehensive CSV analysis report.
        
        Args:
            emulator: Optional WarspiteTradingEmulator for portfolio analysis
            **kwargs: Additional rendering options
            
        Returns:
            CSV text with comprehensive analysis
        """
        options = {**self._style_options, **kwargs}
        
        output_parts = []
        
        # Header
        output_parts.append("# Comprehensive Financial Analysis Report")
        output_parts.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_parts.append("#")
        
        # Dataset summary
        dataset_summary = self._create_dataset_analysis_section(options)
        output_parts.append(dataset_summary)
        
        # Main data
        main_data = self.render(**options)
        output_parts.append(main_data)
        
        # Portfolio analysis if emulator provided
        if emulator is not None:
            portfolio_analysis = self._create_portfolio_analysis_section(emulator, options)
            output_parts.append(portfolio_analysis)
        
        return '\n'.join(output_parts)
    
    def _create_dataset_analysis_section(self, options: Dict[str, Any]) -> str:
        """Create detailed dataset analysis section."""
        lines = []
        lines.append("# Dataset Analysis:")
        lines.append("#")
        
        # Price correlation analysis (if multiple symbols)
        if len(self._dataset.symbols) > 1:
            lines.append("# Price Correlations:")
            
            # Extract close prices for all symbols
            price_data = {}
            for i, symbol in enumerate(self._dataset.symbols):
                data_array = self._dataset.data_arrays[i]
                if data_array.ndim == 1:
                    prices = data_array
                else:
                    prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
                price_data[symbol] = prices
            
            # Calculate correlations
            price_df = pd.DataFrame(price_data)
            correlations = price_df.corr()
            
            for i, symbol1 in enumerate(self._dataset.symbols):
                for j, symbol2 in enumerate(self._dataset.symbols):
                    if i < j:  # Only show upper triangle
                        corr_value = correlations.loc[symbol1, symbol2]
                        lines.append(f"#   {symbol1} vs {symbol2}: {corr_value:.4f}")
            
            lines.append("#")
        
        # Volatility analysis
        lines.append("# Volatility Analysis (Annualized):")
        for i, symbol in enumerate(self._dataset.symbols):
            data_array = self._dataset.data_arrays[i]
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Calculate daily returns
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            lines.append(f"#   {symbol}: {volatility*100:.2f}%")
        
        lines.append("#")
        return '\n'.join(lines)
    
    def _create_portfolio_analysis_section(self, emulator, options: Dict[str, Any]) -> str:
        """Create portfolio analysis section."""
        lines = []
        lines.append("#")
        lines.append("# Portfolio Analysis:")
        lines.append("#")
        
        # Performance metrics
        if hasattr(emulator, 'get_performance_metrics'):
            metrics = emulator.get_performance_metrics()
            
            lines.append("# Performance Metrics:")
            lines.append(f"#   Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            lines.append(f"#   Volatility: {metrics.get('volatility', 0)*100:.2f}%")
            lines.append(f"#   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            lines.append(f"#   Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            lines.append(f"#   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            lines.append(f"#   Total Trades: {metrics.get('total_trades', 0)}")
            lines.append("#")
        
        # Portfolio history
        if hasattr(emulator, 'portfolio_history') and len(emulator.portfolio_history) > 1:
            lines.append("# Portfolio History:")
            lines.append("# Date,Portfolio_Value,Daily_Return")
            
            portfolio_values = emulator.portfolio_history
            timestamps = getattr(emulator, '_timestamp_history', None)
            
            if timestamps and len(timestamps) == len(portfolio_values):
                for i, (timestamp, value) in enumerate(zip(timestamps, portfolio_values)):
                    if i == 0:
                        daily_return = 0.0
                    else:
                        daily_return = (value - portfolio_values[i-1]) / portfolio_values[i-1]
                    
                    date_str = timestamp.strftime(options.get('date_format', '%Y-%m-%d'))
                    lines.append(f"# {date_str},{value:.2f},{daily_return:.6f}")
            
            lines.append("#")
        
        # Trade history
        if hasattr(emulator, 'trade_history') and emulator.trade_history:
            lines.append("# Trade History:")
            lines.append("# Timestamp,Symbol,Action,Quantity,Price,Fee,Portfolio_Value")
            
            for trade in emulator.trade_history[-20:]:  # Last 20 trades
                timestamp_str = trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                lines.append(f"# {timestamp_str},{trade.symbol},{trade.action},"
                           f"{trade.quantity:.4f},{trade.price:.6f},{trade.fee:.2f},"
                           f"{trade.portfolio_value:.2f}")
            
            lines.append("#")
        
        return '\n'.join(lines)
    
    def export_for_excel(self, filepath: str, emulator=None, **kwargs) -> None:
        """
        Export data in Excel-compatible format with multiple sheets.
        
        Args:
            filepath: Path to save Excel file
            emulator: Optional WarspiteTradingEmulator for portfolio data
            **kwargs: Additional export options
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main data sheet
            df = self._dataset.to_dataframe()
            formatted_df = self._format_dataframe(df, kwargs)
            formatted_df.to_excel(writer, sheet_name='Data', index=True)
            
            # Summary sheet
            summary_data = self._create_summary_dataframe()
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Portfolio sheet (if emulator provided)
            if emulator is not None and hasattr(emulator, 'portfolio_history'):
                portfolio_df = self._create_portfolio_dataframe(emulator)
                if not portfolio_df.empty:
                    portfolio_df.to_excel(writer, sheet_name='Portfolio', index=True)
                
                # Trades sheet
                if hasattr(emulator, 'trade_history') and emulator.trade_history:
                    trades_df = self._create_trades_dataframe(emulator)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary statistics DataFrame."""
        summary_data = []
        
        for i, symbol in enumerate(self._dataset.symbols):
            data_array = self._dataset.data_arrays[i]
            
            # Extract prices
            if data_array.ndim == 1:
                prices = data_array
            else:
                prices = data_array[:, 3] if data_array.shape[1] > 3 else data_array[:, -1]
            
            # Calculate statistics
            summary_data.append({
                'Symbol': symbol,
                'Min': np.min(prices),
                'Max': np.max(prices),
                'Mean': np.mean(prices),
                'Std': np.std(prices),
                'First': prices[0] if len(prices) > 0 else np.nan,
                'Last': prices[-1] if len(prices) > 0 else np.nan,
                'Count': len(prices)
            })
        
        return pd.DataFrame(summary_data)
    
    def _create_portfolio_dataframe(self, emulator) -> pd.DataFrame:
        """Create portfolio history DataFrame."""
        if not hasattr(emulator, 'portfolio_history') or len(emulator.portfolio_history) < 2:
            return pd.DataFrame()
        
        portfolio_values = emulator.portfolio_history
        timestamps = getattr(emulator, '_timestamp_history', None)
        
        if timestamps and len(timestamps) == len(portfolio_values):
            data = {
                'Portfolio_Value': portfolio_values,
                'Daily_Return': [0.0] + [
                    (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    for i in range(1, len(portfolio_values))
                ]
            }
            
            return pd.DataFrame(data, index=pd.to_datetime(timestamps))
        
        return pd.DataFrame()
    
    def _create_trades_dataframe(self, emulator) -> pd.DataFrame:
        """Create trades history DataFrame."""
        if not hasattr(emulator, 'trade_history') or not emulator.trade_history:
            return pd.DataFrame()
        
        trades_data = []
        for trade in emulator.trade_history:
            trades_data.append({
                'Timestamp': trade.timestamp,
                'Symbol': trade.symbol,
                'Action': trade.action,
                'Quantity': trade.quantity,
                'Price': trade.price,
                'Fee': trade.fee,
                'Portfolio_Value': trade.portfolio_value
            })
        
        return pd.DataFrame(trades_data)