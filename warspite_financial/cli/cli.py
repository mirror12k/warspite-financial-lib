"""
Command line interface implementation for warspite_financial library.

This module contains the WarspiteCLI class for interactive trading and dataset visualization.
"""

import sys
import cmd
from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
import pandas as pd

from ..providers.base import TradingProvider, Position
from ..datasets.dataset import WarspiteDataset


class WarspiteCLI:
    """
    Command line interface for warspite_financial library.
    
    Provides interactive trading capabilities and dataset visualization
    through console-based commands and ASCII rendering.
    """
    
    def __init__(self, trading_provider: Optional[TradingProvider] = None):
        """
        Initialize CLI with optional trading provider.
        
        Args:
            trading_provider: Optional TradingProvider for live trading operations
        """
        self._trading_provider = trading_provider
        self._current_dataset: Optional[WarspiteDataset] = None
        
        # Validate trading provider if provided
        if self._trading_provider is not None:
            self._validate_trading_provider()
    
    def _validate_trading_provider(self) -> None:
        """
        Validate the trading provider connection and permissions.
        
        Raises:
            ValueError: If provider validation fails
            ConnectionError: If provider is unreachable
        """
        try:
            # Test basic connectivity
            account_info = self._trading_provider.get_account_info()
            if not account_info:
                raise ValueError("Provider failed to return account information")
            
            # Test position access
            positions = self._trading_provider.get_positions()
            if positions is None:
                raise ValueError("Provider failed to return position information")
            
            print(f"✓ Trading provider connected: {type(self._trading_provider).__name__}")
            print(f"  Account ID: {account_info.account_id}")
            print(f"  Balance: {account_info.balance} {account_info.currency}")
            
        except Exception as e:
            raise ConnectionError(f"Trading provider validation failed: {e}")
    
    def get_positions(self) -> List[Position]:
        """
        Get current trading positions and display them.
        
        Returns:
            List of Position objects, or empty list if no provider connected
        
        Raises:
            ConnectionError: If provider is unreachable
        """
        if self._trading_provider is None:
            print("❌ No trading provider connected")
            return []
        
        try:
            positions = self._trading_provider.get_positions()
            account_info = self._trading_provider.get_account_info()
            
            print("\n" + "="*70)
            print("CURRENT POSITIONS")
            print("="*70)
            print(f"Account: {account_info.account_id}")
            print(f"Balance: {account_info.balance:.2f} {account_info.currency}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-"*70)
            
            if not positions:
                print("No open positions")
            else:
                # Header
                print(f"{'Position ID':<15} {'Symbol':<10} {'Quantity':<12} {'Price':<10} {'P&L':<12} {'%':<8}")
                print("-"*70)
                
                total_pnl = 0.0
                total_value = 0.0
                
                for pos in positions:
                    position_value = abs(pos.quantity) * pos.current_price
                    pnl_percent = (pos.unrealized_pnl / position_value * 100) if position_value > 0 else 0.0
                    
                    print(f"{pos.position_id:<15} {pos.symbol:<10} {pos.quantity:<12.4f} "
                          f"{pos.current_price:<10.4f} {pos.unrealized_pnl:<12.2f} {pnl_percent:<8.2f}%")
                    
                    total_pnl += pos.unrealized_pnl
                    total_value += position_value
                
                print("-"*70)
                total_pnl_percent = (total_pnl / account_info.balance * 100) if account_info.balance > 0 else 0.0
                print(f"{'Total P&L:':<48} {total_pnl:<12.2f} {total_pnl_percent:<8.2f}%")
                print(f"{'Total Position Value:':<48} {total_value:<12.2f}")
            
            print("="*70)
            return positions
            
        except Exception as e:
            print(f"❌ Error retrieving positions: {e}")
            return []
    
    def open_position(self, symbol: str, quantity: float, order_type: str = 'market') -> bool:
        """
        Open a new trading position with enhanced validation and feedback.
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade (positive for buy, negative for sell)
            order_type: Type of order ('market', 'limit')
        
        Returns:
            True if position was opened successfully, False otherwise
        
        Raises:
            ValueError: If no trading provider is connected or parameters are invalid
            ConnectionError: If provider is unreachable
        """
        if self._trading_provider is None:
            print("❌ No trading provider connected")
            return False
        
        if not symbol or quantity == 0:
            print("❌ Invalid symbol or quantity")
            return False
        
        try:
            # Validate symbol
            if not self._trading_provider.validate_symbol(symbol):
                print(f"❌ Invalid symbol: {symbol}")
                return False
            
            # Get account info for validation
            account_info = self._trading_provider.get_account_info()
            
            # Determine order type and action
            action = "BUY" if quantity > 0 else "SELL"
            abs_quantity = abs(quantity)
            
            # Basic risk check - don't allow orders larger than account balance
            estimated_cost = abs_quantity * 100  # Rough estimate, actual price may vary
            if estimated_cost > account_info.balance:
                print(f"⚠️  Warning: Estimated order cost ({estimated_cost:.2f}) may exceed account balance ({account_info.balance:.2f})")
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("Order cancelled")
                    return False
            
            print(f"\n🔄 Placing {action} order for {abs_quantity} of {symbol}...")
            print(f"   Order type: {order_type.upper()}")
            print(f"   Account balance: {account_info.balance:.2f} {account_info.currency}")
            
            # Place the order
            order_result = self._trading_provider.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type
            )
            
            if order_result.success:
                print(f"✅ Order executed successfully!")
                print(f"   Order ID: {order_result.order_id}")
                print(f"   Action: {action}")
                print(f"   Symbol: {symbol}")
                print(f"   Quantity: {abs_quantity}")
                print(f"   Type: {order_type.upper()}")
                
                # Show updated positions after a brief pause
                print("\n📊 Updated positions:")
                self.get_positions()
                return True
            else:
                print(f"❌ Order failed: {order_result.message}")
                return False
                
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return False
    
    def close_position(self, position_id: str) -> bool:
        """
        Close a specific trading position with confirmation.
        
        Args:
            position_id: ID of the position to close
        
        Returns:
            True if position was closed successfully, False otherwise
        
        Raises:
            ValueError: If no trading provider is connected or position_id is invalid
            ConnectionError: If provider is unreachable
        """
        if self._trading_provider is None:
            print("❌ No trading provider connected")
            return False
        
        if not position_id:
            print("❌ Invalid position ID")
            return False
        
        try:
            # Get current positions to show what we're closing
            positions = self._trading_provider.get_positions()
            target_position = None
            
            for pos in positions:
                if pos.position_id == position_id:
                    target_position = pos
                    break
            
            if target_position is None:
                print(f"❌ Position {position_id} not found")
                return False
            
            # Show position details before closing
            print(f"\n🔄 Closing position:")
            print(f"   Position ID: {target_position.position_id}")
            print(f"   Symbol: {target_position.symbol}")
            print(f"   Quantity: {target_position.quantity}")
            print(f"   Current Price: {target_position.current_price:.4f}")
            print(f"   Unrealized P&L: {target_position.unrealized_pnl:.2f}")
            
            # Confirmation for significant losses
            if target_position.unrealized_pnl < -100:  # Arbitrary threshold
                response = input(f"⚠️  This position has a loss of {target_position.unrealized_pnl:.2f}. Continue? (y/N): ").strip().lower()
                if response != 'y':
                    print("Position close cancelled")
                    return False
            
            success = self._trading_provider.close_position(position_id)
            
            if success:
                print(f"✅ Position {position_id} closed successfully!")
                
                # Show updated positions
                print("\n📊 Updated positions:")
                self.get_positions()
                return True
            else:
                print(f"❌ Failed to close position {position_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open trading positions with confirmation.
        
        Returns:
            True if all positions were closed successfully, False otherwise
        
        Raises:
            ValueError: If no trading provider is connected
            ConnectionError: If provider is unreachable
        """
        if self._trading_provider is None:
            print("❌ No trading provider connected")
            return False
        
        try:
            # Get current positions first
            positions = self._trading_provider.get_positions()
            
            if not positions:
                print("ℹ️  No open positions to close")
                return True
            
            # Show summary of positions to be closed
            print(f"\n⚠️  About to close {len(positions)} open positions:")
            print("-" * 50)
            
            total_pnl = 0.0
            for pos in positions:
                print(f"   {pos.symbol}: {pos.quantity:+.4f} (P&L: {pos.unrealized_pnl:+.2f})")
                total_pnl += pos.unrealized_pnl
            
            print("-" * 50)
            print(f"   Total P&L: {total_pnl:+.2f}")
            
            # Confirmation prompt
            response = input("\nAre you sure you want to close ALL positions? (y/N): ").strip().lower()
            if response != 'y':
                print("Operation cancelled")
                return False
            
            print(f"\n🔄 Closing {len(positions)} open positions...")
            
            success = self._trading_provider.close_all_positions()
            
            if success:
                print("✅ All positions closed successfully!")
                
                # Show updated positions
                print("\n📊 Updated positions:")
                self.get_positions()
                return True
            else:
                print("❌ Failed to close some positions")
                
                # Show remaining positions
                print("\n📊 Remaining positions:")
                self.get_positions()
                return False
                
        except Exception as e:
            print(f"❌ Error closing positions: {e}")
            return False
    
    def render_dataset_ascii(self, dataset: WarspiteDataset, width: int = 80, height: int = 20) -> None:
        """
        Render dataset as ASCII chart for console visualization.
        
        Args:
            dataset: WarspiteDataset to render
            width: Width of the ASCII chart in characters
            height: Height of the ASCII chart in characters
        
        Raises:
            ValueError: If dataset is invalid or empty
        """
        if dataset is None or len(dataset) == 0:
            print("❌ No dataset to render")
            return
        
        try:
            print(f"\n📊 ASCII Chart - Dataset Visualization")
            print("=" * width)
            
            # Get price data for the first symbol (or all symbols if multiple)
            symbols = dataset.symbols
            timestamps = pd.to_datetime(dataset.timestamps)
            
            # Determine what to plot
            if len(symbols) == 1:
                # Single symbol - plot OHLC or Close
                symbol = symbols[0]
                data_array = dataset.data_arrays[0]
                
                if data_array.ndim == 2 and data_array.shape[1] >= 4:
                    # OHLC data available
                    prices = data_array[:, 3]  # Close prices
                    price_type = "Close"
                else:
                    # Single price series
                    prices = data_array
                    price_type = "Price"
                
                self._render_single_series_ascii(
                    prices, timestamps, f"{symbol} {price_type}", width, height
                )
                
            else:
                # Multiple symbols - plot close prices for each
                print(f"Multiple symbols detected: {', '.join(symbols[:3])}")
                if len(symbols) > 3:
                    print(f"... and {len(symbols) - 3} more (showing first 3)")
                
                for i, symbol in enumerate(symbols[:3]):  # Limit to first 3 symbols
                    data_array = dataset.data_arrays[i]
                    
                    if data_array.ndim == 2 and data_array.shape[1] >= 4:
                        prices = data_array[:, 3]  # Close prices
                    else:
                        prices = data_array
                    
                    print(f"\n{symbol}:")
                    self._render_single_series_ascii(
                        prices, timestamps, f"{symbol} Close", width, height // 3
                    )
            
            # Show strategy results if available
            if dataset.strategy_results is not None:
                print(f"\n📈 Strategy Results:")
                strategy_results = dataset.strategy_results
                
                if strategy_results.ndim == 1:
                    # Single strategy result
                    self._render_strategy_results_ascii(
                        strategy_results, timestamps, "Strategy Positions", width, height // 4
                    )
                else:
                    # Multi-symbol strategy results
                    print("Multi-symbol strategy results (showing first symbol):")
                    self._render_strategy_results_ascii(
                        strategy_results[:, 0], timestamps, f"Strategy - {symbols[0]}", width, height // 4
                    )
            
            print("=" * width)
            
        except Exception as e:
            print(f"❌ Error rendering dataset: {e}")
    
    def _render_single_series_ascii(self, prices: np.ndarray, timestamps: pd.DatetimeIndex, 
                                   title: str, width: int, height: int) -> None:
        """
        Render a single price series as ASCII chart.
        
        Args:
            prices: Array of price values
            timestamps: Corresponding timestamps
            title: Chart title
            width: Chart width in characters
            height: Chart height in characters
        """
        if len(prices) == 0:
            print("No data to plot")
            return
        
        # Calculate chart dimensions
        chart_width = width - 15  # Leave space for price labels
        chart_height = height - 3  # Leave space for title and axis
        
        # Normalize prices to chart height
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            print(f"{title}: Constant price {min_price:.4f}")
            return
        
        # Create the chart grid
        chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Sample data points to fit chart width
        if len(prices) > chart_width:
            # Downsample data
            indices = np.linspace(0, len(prices) - 1, chart_width, dtype=int)
            sampled_prices = prices[indices]
            sampled_timestamps = timestamps[indices]
        else:
            sampled_prices = prices
            sampled_timestamps = timestamps
        
        # Plot the price line
        for i, price in enumerate(sampled_prices):
            if i < chart_width:
                # Calculate y position (inverted for display)
                y = int((max_price - price) / price_range * (chart_height - 1))
                y = max(0, min(chart_height - 1, y))
                
                # Use different characters for different parts of the line
                if i == 0:
                    char = '●'  # Start point
                elif i == len(sampled_prices) - 1:
                    char = '●'  # End point
                else:
                    char = '─'  # Line
                
                chart[y][i] = char
        
        # Print the chart
        print(f"\n{title}")
        print(f"Range: {min_price:.4f} - {max_price:.4f}")
        
        # Print chart with price labels
        for y in range(chart_height):
            # Calculate price for this row
            row_price = max_price - (y / (chart_height - 1)) * price_range
            price_label = f"{row_price:8.4f} │"
            
            # Print price label and chart row
            row_chars = ''.join(chart[y])
            print(f"{price_label}{row_chars}")
        
        # Print time axis
        time_axis = " " * 10 + "└" + "─" * chart_width
        print(time_axis)
        
        # Print time labels
        if len(sampled_timestamps) > 0:
            start_time = sampled_timestamps[0].strftime('%m/%d')
            end_time = sampled_timestamps[-1].strftime('%m/%d')
            time_labels = " " * 11 + start_time + " " * (chart_width - len(start_time) - len(end_time)) + end_time
            print(time_labels)
    
    def _render_strategy_results_ascii(self, positions: np.ndarray, timestamps: pd.DatetimeIndex,
                                     title: str, width: int, height: int) -> None:
        """
        Render strategy positions as ASCII chart.
        
        Args:
            positions: Array of position values (-1 to 1)
            timestamps: Corresponding timestamps
            title: Chart title
            width: Chart width in characters
            height: Chart height in characters
        """
        if len(positions) == 0:
            print("No strategy data to plot")
            return
        
        # Calculate chart dimensions
        chart_width = width - 15  # Leave space for position labels
        chart_height = height - 2  # Leave space for title and axis
        
        # Create the chart grid
        chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Sample data points to fit chart width
        if len(positions) > chart_width:
            indices = np.linspace(0, len(positions) - 1, chart_width, dtype=int)
            sampled_positions = positions[indices]
        else:
            sampled_positions = positions
        
        # Plot the position line
        zero_line = chart_height // 2  # Middle line for zero position
        
        for i, position in enumerate(sampled_positions):
            if i < chart_width and not np.isnan(position):
                # Calculate y position (position ranges from -1 to 1)
                # Map to chart height with zero in the middle
                y = int(zero_line - (position * (chart_height // 2)))
                y = max(0, min(chart_height - 1, y))
                
                # Use different characters for different position types
                if position > 0.1:
                    char = '▲'  # Long position
                elif position < -0.1:
                    char = '▼'  # Short position
                else:
                    char = '─'  # Neutral/small position
                
                chart[y][i] = char
        
        # Add zero line
        for i in range(chart_width):
            if chart[zero_line][i] == ' ':
                chart[zero_line][i] = '·'
        
        # Print the chart
        print(f"\n{title}")
        print("Range: -1.0 (Short) to +1.0 (Long)")
        
        # Print chart with position labels
        for y in range(chart_height):
            # Calculate position for this row
            row_position = 1.0 - (y / (chart_height - 1)) * 2.0
            
            if y == zero_line:
                position_label = f"   0.00 ┼"
            else:
                position_label = f"{row_position:7.2f} │"
            
            # Print position label and chart row
            row_chars = ''.join(chart[y])
            print(f"{position_label}{row_chars}")
        
        # Print legend
        print(" " * 10 + "Legend: ▲ Long  ▼ Short  · Zero  ─ Neutral")
    
    def run_interactive_mode(self) -> None:
        """
        Run the CLI in interactive mode with command loop.
        
        Provides a command-line interface for real-time trading operations
        with help system and command validation. Supports Ctrl+D for exit.
        """
        print("🚀 Starting Warspite Financial CLI Interactive Mode...")
        
        # Create and run the interactive command processor
        interactive_cli = WarspiteCLIInteractive(self)
        
        try:
            interactive_cli.cmdloop()
        except KeyboardInterrupt:
            print("\n\n👋 Interactive mode interrupted. Goodbye!")
        except EOFError:
            # Handle Ctrl+D gracefully
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Interactive mode error: {e}")
            print("Exiting interactive mode...")
    
    def set_current_dataset(self, dataset: WarspiteDataset) -> None:
        """
        Set the current dataset for CLI operations.
        
        Args:
            dataset: WarspiteDataset to use for CLI operations
        """
        self._current_dataset = dataset
        print(f"✅ Dataset loaded: {len(dataset.symbols)} symbols, {len(dataset)} data points")
    
    def get_current_dataset(self) -> Optional[WarspiteDataset]:
        """
        Get the current dataset.
        
        Returns:
            Current WarspiteDataset or None if no dataset is loaded
        """
        return self._current_dataset


class WarspiteCLIInteractive(cmd.Cmd):
    """
    Interactive command processor for WarspiteCLI.
    
    Provides a command-line interface with help system and command validation.
    """
    
    intro = '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Warspite Financial CLI                              ║
║                     Interactive Trading Interface                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Type 'help' or '?' to list commands.
Type 'help <command>' for detailed help on a specific command.
Type 'quit', 'exit', or press Ctrl+D to leave the interactive mode.

'''
    
    prompt = 'warspite> '
    
    def __init__(self, cli_instance: WarspiteCLI):
        """
        Initialize interactive CLI with WarspiteCLI instance.
        
        Args:
            cli_instance: WarspiteCLI instance to use for operations
        """
        super().__init__()
        self.cli = cli_instance
    
    def do_help(self, arg):
        """Override help command to ensure it always prints something."""
        if arg:
            # Call parent's help for specific commands
            super().do_help(arg)
        else:
            # Print general help when no argument is given
            super().do_help(arg)
            # If no output was generated, print the documented commands
            print("\nDocumented commands (type help <topic>):")
            print("=" * 40)
            commands = [name[3:] for name in dir(self) if name.startswith('do_')]
            commands.sort()
            for i in range(0, len(commands), 8):
                print("  ".join(f"{cmd:<8}" for cmd in commands[i:i+8]))
    
    def do_positions(self, arg):
        """Show current trading positions."""
        self.cli.get_positions()
    
    def help_positions(self):
        """Help for positions command."""
        print("""
positions - Show current trading positions

Usage: positions

Displays all open positions with their details including:
- Position ID
- Symbol
- Quantity
- Current price
- Unrealized P&L
- Account balance
""")
    
    def do_buy(self, arg):
        """Buy a specified quantity of a symbol."""
        try:
            parts = arg.split()
            if len(parts) < 2:
                print("Usage: buy <symbol> <quantity> [order_type]")
                print("Example: buy EURUSD 1000")
                print("         buy AAPL 10 limit")
                return
            
            symbol = parts[0].upper()
            quantity = float(parts[1])
            order_type = parts[2].lower() if len(parts) > 2 else 'market'
            
            if quantity <= 0:
                print("❌ Quantity must be positive for buy orders")
                return
            
            self.cli.open_position(symbol, quantity, order_type)
            
        except ValueError:
            print("❌ Invalid quantity. Must be a number.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def help_buy(self):
        """Help for buy command."""
        print("""
buy - Place a buy order for a symbol

Usage: buy <symbol> <quantity> [order_type]

Arguments:
  symbol      - Trading symbol (e.g., EURUSD, AAPL)
  quantity    - Quantity to buy (must be positive)
  order_type  - Optional order type (market, limit) - default: market

Examples:
  buy EURUSD 1000
  buy AAPL 10
  buy BTCUSD 0.1 limit
""")
    
    def do_sell(self, arg):
        """Sell a specified quantity of a symbol."""
        try:
            parts = arg.split()
            if len(parts) < 2:
                print("Usage: sell <symbol> <quantity> [order_type]")
                print("Example: sell EURUSD 1000")
                print("         sell AAPL 10 limit")
                return
            
            symbol = parts[0].upper()
            quantity = float(parts[1])
            order_type = parts[2].lower() if len(parts) > 2 else 'market'
            
            if quantity <= 0:
                print("❌ Quantity must be positive for sell orders")
                return
            
            # Convert to negative for sell
            self.cli.open_position(symbol, -quantity, order_type)
            
        except ValueError:
            print("❌ Invalid quantity. Must be a number.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def help_sell(self):
        """Help for sell command."""
        print("""
sell - Place a sell order for a symbol

Usage: sell <symbol> <quantity> [order_type]

Arguments:
  symbol      - Trading symbol (e.g., EURUSD, AAPL)
  quantity    - Quantity to sell (must be positive)
  order_type  - Optional order type (market, limit) - default: market

Examples:
  sell EURUSD 1000
  sell AAPL 10
  sell BTCUSD 0.1 limit
""")
    
    def do_close(self, arg):
        """Close a specific position or all positions."""
        if not arg:
            print("Usage: close <position_id> or close all")
            print("Use 'positions' command to see available position IDs")
            return
        
        if arg.lower() == 'all':
            self.cli.close_all_positions()
        else:
            self.cli.close_position(arg)
    
    def help_close(self):
        """Help for close command."""
        print("""
close - Close trading positions

Usage: 
  close <position_id>  - Close a specific position
  close all           - Close all open positions

Arguments:
  position_id - ID of the position to close (from positions command)

Examples:
  close 12345
  close all
""")
    
    def do_status(self, arg):
        """Show trading provider status."""
        if self.cli._trading_provider is None:
            print("❌ No trading provider connected")
        else:
            try:
                account_info = self.cli._trading_provider.get_account_info()
                print(f"\n✅ Trading provider: {type(self.cli._trading_provider).__name__}")
                print(f"   Account ID: {account_info.account_id}")
                print(f"   Balance: {account_info.balance} {account_info.currency}")
                print(f"   Status: Connected")
            except Exception as e:
                print(f"❌ Provider error: {e}")
    
    def help_status(self):
        """Help for status command."""
        print("""
status - Show trading provider connection status

Usage: status

Displays:
- Trading provider type
- Account information
- Connection status
""")
    
    def do_dataset(self, arg):
        """Show current dataset information or render dataset."""
        if not arg:
            # Show current dataset info
            if self.cli._current_dataset is None:
                print("ℹ️  No dataset loaded")
            else:
                dataset = self.cli._current_dataset
                print(f"\n📊 Current Dataset:")
                print(f"   Symbols: {', '.join(dataset.symbols)}")
                print(f"   Length: {len(dataset)} data points")
                print(f"   Date range: {dataset.timestamps[0]} to {dataset.timestamps[-1]}")
                if dataset.strategy_results is not None:
                    print(f"   Strategy results: Available")
        elif arg.lower() == 'render':
            # Render current dataset
            if self.cli._current_dataset is None:
                print("❌ No dataset loaded")
            else:
                self.cli.render_dataset_ascii(self.cli._current_dataset)
        else:
            print("Usage: dataset [render]")
    
    def help_dataset(self):
        """Help for dataset command."""
        print("""
dataset - Show dataset information or render visualization

Usage: 
  dataset        - Show current dataset information
  dataset render - Render ASCII visualization of current dataset

The dataset command helps you work with loaded financial datasets.
Use 'dataset render' to see an ASCII chart of the price data.
""")
    
    def do_quit(self, arg):
        """Exit the interactive CLI."""
        print("\nGoodbye! 👋")
        return True
    
    def do_exit(self, arg):
        """Exit the interactive CLI."""
        return self.do_quit(arg)
    
    def help_quit(self):
        """Help for quit command."""
        print("""
quit/exit - Exit the interactive CLI

Usage: quit, exit, or press Ctrl+D

Exits the interactive command interface and returns to the Python shell.
""")
    
    def help_exit(self):
        """Help for exit command."""
        self.help_quit()
    
    def emptyline(self):
        """Handle empty line input."""
        pass
    
    def do_EOF(self, arg):
        """Handle Ctrl+D (EOF) to exit gracefully."""
        print("\n👋 Goodbye!")
        return True
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"❌ Unknown command: {line}")
        print("Type 'help' to see available commands.")
    
    def do_load(self, arg):
        """Load a dataset for visualization."""
        if not arg:
            print("Usage: load <dataset_path>")
            print("Example: load /path/to/dataset.csv")
            return
        
        try:
            # This is a placeholder - in real implementation, you'd load from file
            print(f"📂 Loading dataset from: {arg}")
            print("⚠️  Dataset loading from file not implemented in this demo")
            print("Use Python API to create and set dataset with cli.set_current_dataset()")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
    
    def help_load(self):
        """Help for load command."""
        print("""
load - Load a dataset for visualization

Usage: load <dataset_path>

Arguments:
  dataset_path - Path to dataset file (CSV, pickle, etc.)

Note: In this demo version, use the Python API to load datasets:
  from warspite_financial.datasets import WarspiteDataset
  dataset = WarspiteDataset.from_provider(...)
  cli.set_current_dataset(dataset)
""")
    
    def do_symbols(self, arg):
        """Show available symbols from trading provider."""
        if self.cli._trading_provider is None:
            print("❌ No trading provider connected")
            return
        
        try:
            print("🔍 Fetching available symbols...")
            symbols = self.cli._trading_provider.get_available_symbols()
            
            if not symbols:
                print("No symbols available")
                return
            
            print(f"\n📋 Available Symbols ({len(symbols)} total):")
            print("=" * 50)
            
            # Display symbols in columns
            cols = 4
            for i in range(0, len(symbols), cols):
                row_symbols = symbols[i:i+cols]
                print("  ".join(f"{sym:<12}" for sym in row_symbols))
            
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
    
    def help_symbols(self):
        """Help for symbols command."""
        print("""
symbols - Show available trading symbols

Usage: symbols

Displays all symbols available from the connected trading provider.
Use these symbols with buy/sell commands.
""")
    
    def do_account(self, arg):
        """Show detailed account information."""
        if self.cli._trading_provider is None:
            print("❌ No trading provider connected")
            return
        
        try:
            account_info = self.cli._trading_provider.get_account_info()
            positions = self.cli._trading_provider.get_positions()
            
            print("\n" + "="*60)
            print("ACCOUNT INFORMATION")
            print("="*60)
            print(f"Account ID: {account_info.account_id}")
            print(f"Currency: {account_info.currency}")
            print(f"Balance: {account_info.balance:.2f}")
            
            if positions:
                total_pnl = sum(pos.unrealized_pnl for pos in positions)
                total_value = sum(abs(pos.quantity) * pos.current_price for pos in positions)
                
                print(f"Open Positions: {len(positions)}")
                print(f"Total Position Value: {total_value:.2f}")
                print(f"Total Unrealized P&L: {total_pnl:.2f}")
                print(f"Equity: {account_info.balance + total_pnl:.2f}")
            else:
                print("Open Positions: 0")
                print("Total Position Value: 0.00")
                print("Total Unrealized P&L: 0.00")
                print(f"Equity: {account_info.balance:.2f}")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error fetching account info: {e}")
    
    def help_account(self):
        """Help for account command."""
        print("""
account - Show detailed account information

Usage: account

Displays comprehensive account information including:
- Account ID and currency
- Current balance
- Open positions summary
- Total unrealized P&L
- Account equity
""")
    
    def do_clear(self, arg):
        """Clear the screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def help_clear(self):
        """Help for clear command."""
        print("""
clear - Clear the terminal screen

Usage: clear

Clears the terminal screen for better readability.
""")
    
    def do_history(self, arg):
        """Show command history (placeholder)."""
        print("📜 Command history feature not implemented in this demo")
        print("Your shell's history (up/down arrows) should work for command recall")
    
    def help_history(self):
        """Help for history command."""
        print("""
history - Show command history

Usage: history

Note: This is a placeholder. Use your shell's built-in history
(up/down arrow keys) to recall previous commands.
""")