"""
Unit tests for CLI functionality.

Tests position management commands, ASCII rendering output, and interactive mode command processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import io
import sys
from datetime import datetime
import numpy as np
import pandas as pd

from warspite_financial.cli.cli import WarspiteCLI, WarspiteCLIInteractive
from warspite_financial.providers.base import TradingProvider, Position, AccountInfo, OrderResult
from warspite_financial.datasets.dataset import WarspiteDataset


class MockTradingProvider(TradingProvider):
    """Mock trading provider for testing."""
    
    def __init__(self):
        super().__init__()
        self.account_info = AccountInfo("TEST123", 10000.0, "USD")
        self.positions = []
        self.orders = []
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AAPL", "GOOGL"]
    
    def get_data(self, symbol, start_date, end_date, interval='1d'):
        # Mock implementation - return empty DataFrame
        return pd.DataFrame()
    
    def get_available_symbols(self):
        return self.symbols
    
    def validate_symbol(self, symbol):
        return symbol in self.symbols
    
    def place_order(self, symbol, quantity, order_type):
        order_id = f"ORDER_{len(self.orders) + 1}"
        self.orders.append({
            'id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type
        })
        
        # Add to positions
        existing_pos = None
        for pos in self.positions:
            if pos.symbol == symbol:
                existing_pos = pos
                break
        
        if existing_pos:
            existing_pos.quantity += quantity
        else:
            new_pos = Position(
                position_id=f"POS_{len(self.positions) + 1}",
                symbol=symbol,
                quantity=quantity,
                current_price=1.2000,  # Mock price
                unrealized_pnl=0.0
            )
            self.positions.append(new_pos)
        
        return OrderResult(order_id, True, "Order executed")
    
    def get_account_info(self):
        return self.account_info
    
    def get_positions(self):
        return self.positions
    
    def close_position(self, position_id):
        for i, pos in enumerate(self.positions):
            if pos.position_id == position_id:
                del self.positions[i]
                return True
        return False
    
    def close_all_positions(self):
        self.positions.clear()
        return True


class TestWarspiteCLI(unittest.TestCase):
    """Test cases for WarspiteCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_provider = MockTradingProvider()
        self.cli = WarspiteCLI(self.mock_provider)
        
        # Create mock dataset
        timestamps = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        data_arrays = [prices]  # 1D array, not reshaped
        symbols = ['AAPL']
        
        self.mock_dataset = WarspiteDataset(
            data_arrays=data_arrays,
            timestamps=timestamps.values,
            symbols=symbols
        )
    
    def test_cli_initialization_with_provider(self):
        """Test CLI initialization with trading provider."""
        cli = WarspiteCLI(self.mock_provider)
        self.assertEqual(cli._trading_provider, self.mock_provider)
        self.assertIsNone(cli._current_dataset)
    
    def test_cli_initialization_without_provider(self):
        """Test CLI initialization without trading provider."""
        cli = WarspiteCLI()
        self.assertIsNone(cli._trading_provider)
        self.assertIsNone(cli._current_dataset)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_positions_with_provider(self, mock_stdout):
        """Test get_positions method with connected provider."""
        # Add a test position
        test_position = Position("POS1", "EURUSD", 1000.0, 1.2000, 50.0)
        self.mock_provider.positions = [test_position]
        
        positions = self.cli.get_positions()
        
        output = mock_stdout.getvalue()
        self.assertIn("CURRENT POSITIONS", output)
        self.assertIn("EURUSD", output)
        self.assertIn("1000.0000", output)
        self.assertIn("50.00", output)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "EURUSD")
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_positions_without_provider(self, mock_stdout):
        """Test get_positions method without provider."""
        cli = WarspiteCLI()
        positions = cli.get_positions()
        
        output = mock_stdout.getvalue()
        self.assertIn("No trading provider connected", output)
        self.assertEqual(positions, [])
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_open_position_success(self, mock_input, mock_stdout):
        """Test successful position opening."""
        result = self.cli.open_position("EURUSD", 100.0)  # Smaller amount to avoid warning
        
        output = mock_stdout.getvalue()
        self.assertTrue(result)
        self.assertIn("Order executed successfully", output)
        self.assertIn("EURUSD", output)
        self.assertIn("BUY", output)
        self.assertEqual(len(self.mock_provider.positions), 1)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_open_position_invalid_symbol(self, mock_stdout):
        """Test position opening with invalid symbol."""
        result = self.cli.open_position("INVALID", 1000.0)
        
        output = mock_stdout.getvalue()
        self.assertFalse(result)
        self.assertIn("Invalid symbol", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_open_position_without_provider(self, mock_stdout):
        """Test position opening without provider."""
        cli = WarspiteCLI()
        result = cli.open_position("EURUSD", 1000.0)
        
        output = mock_stdout.getvalue()
        self.assertFalse(result)
        self.assertIn("No trading provider connected", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_close_position_success(self, mock_input, mock_stdout):
        """Test successful position closing."""
        # First open a position
        self.cli.open_position("EURUSD", 100.0)  # Smaller amount
        position_id = self.mock_provider.positions[0].position_id
        
        result = self.cli.close_position(position_id)
        
        output = mock_stdout.getvalue()
        self.assertTrue(result)
        self.assertIn("closed successfully", output)
        self.assertEqual(len(self.mock_provider.positions), 0)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_close_position_not_found(self, mock_stdout):
        """Test closing non-existent position."""
        result = self.cli.close_position("INVALID_ID")
        
        output = mock_stdout.getvalue()
        self.assertFalse(result)
        self.assertIn("not found", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_close_all_positions_success(self, mock_input, mock_stdout):
        """Test closing all positions successfully."""
        # Open multiple positions
        self.cli.open_position("EURUSD", 100.0)  # Smaller amounts
        self.cli.open_position("GBPUSD", 50.0)
        
        result = self.cli.close_all_positions()
        
        output = mock_stdout.getvalue()
        self.assertTrue(result)
        self.assertIn("All positions closed successfully", output)
        self.assertEqual(len(self.mock_provider.positions), 0)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='n')  # Mock user input
    def test_close_all_positions_cancelled(self, mock_input, mock_stdout):
        """Test cancelling close all positions."""
        # Open a position
        self.cli.open_position("EURUSD", 100.0)  # Smaller amount
        
        result = self.cli.close_all_positions()
        
        output = mock_stdout.getvalue()
        self.assertFalse(result)
        self.assertIn("Operation cancelled", output)
        self.assertEqual(len(self.mock_provider.positions), 1)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_render_dataset_ascii_single_symbol(self, mock_stdout):
        """Test ASCII rendering of single symbol dataset."""
        self.cli.render_dataset_ascii(self.mock_dataset)
        
        output = mock_stdout.getvalue()
        self.assertIn("ASCII Chart", output)
        self.assertIn("AAPL", output)
        self.assertIn("Range:", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_render_dataset_ascii_empty_dataset(self, mock_stdout):
        """Test ASCII rendering with empty dataset."""
        self.cli.render_dataset_ascii(None)
        
        output = mock_stdout.getvalue()
        self.assertIn("No dataset to render", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_render_dataset_ascii_with_strategy_results(self, mock_stdout):
        """Test ASCII rendering with strategy results."""
        # Add strategy results to dataset
        strategy_results = np.random.uniform(-1, 1, len(self.mock_dataset))
        self.mock_dataset.add_strategy_results(strategy_results)
        
        self.cli.render_dataset_ascii(self.mock_dataset)
        
        output = mock_stdout.getvalue()
        self.assertIn("Strategy Results", output)
        self.assertIn("Legend:", output)
    
    def test_set_current_dataset(self):
        """Test setting current dataset."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            self.cli.set_current_dataset(self.mock_dataset)
            
            output = mock_stdout.getvalue()
            self.assertIn("Dataset loaded", output)
            self.assertEqual(self.cli.get_current_dataset(), self.mock_dataset)
    
    def test_get_current_dataset_none(self):
        """Test getting current dataset when none is set."""
        self.assertIsNone(self.cli.get_current_dataset())


class TestWarspiteCLIInteractive(unittest.TestCase):
    """Test cases for WarspiteCLIInteractive class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_provider = MockTradingProvider()
        self.cli = WarspiteCLI(self.mock_provider)
        self.interactive_cli = WarspiteCLIInteractive(self.cli)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_positions(self, mock_stdout):
        """Test positions command in interactive mode."""
        self.interactive_cli.do_positions("")
        
        output = mock_stdout.getvalue()
        self.assertIn("CURRENT POSITIONS", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_do_buy_valid(self, mock_input, mock_stdout):
        """Test buy command with valid parameters."""
        self.interactive_cli.do_buy("EURUSD 100")  # Smaller amount
        
        output = mock_stdout.getvalue()
        self.assertIn("Order executed successfully", output)
        self.assertEqual(len(self.mock_provider.positions), 1)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_buy_invalid_usage(self, mock_stdout):
        """Test buy command with invalid usage."""
        self.interactive_cli.do_buy("EURUSD")
        
        output = mock_stdout.getvalue()
        self.assertIn("Usage:", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_buy_invalid_quantity(self, mock_stdout):
        """Test buy command with invalid quantity."""
        self.interactive_cli.do_buy("EURUSD abc")
        
        output = mock_stdout.getvalue()
        self.assertIn("Invalid quantity", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_do_sell_valid(self, mock_input, mock_stdout):
        """Test sell command with valid parameters."""
        self.interactive_cli.do_sell("EURUSD 100")  # Smaller amount
        
        output = mock_stdout.getvalue()
        self.assertIn("Order executed successfully", output)
        # Should create a short position (negative quantity)
        self.assertEqual(len(self.mock_provider.positions), 1)
        self.assertEqual(self.mock_provider.positions[0].quantity, -100.0)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_do_close_specific_position(self, mock_input, mock_stdout):
        """Test close command for specific position."""
        # First create a position
        self.interactive_cli.do_buy("EURUSD 100")  # Smaller amount
        position_id = self.mock_provider.positions[0].position_id
        
        self.interactive_cli.do_close(position_id)
        
        output = mock_stdout.getvalue()
        self.assertIn("closed successfully", output)
        self.assertEqual(len(self.mock_provider.positions), 0)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('builtins.input', return_value='y')  # Mock user input
    def test_do_close_all_positions(self, mock_input, mock_stdout):
        """Test close all command."""
        # Create multiple positions
        self.interactive_cli.do_buy("EURUSD 100")  # Smaller amounts
        self.interactive_cli.do_buy("GBPUSD 50")
        
        self.interactive_cli.do_close("all")
        
        output = mock_stdout.getvalue()
        self.assertIn("All positions closed successfully", output)
        self.assertEqual(len(self.mock_provider.positions), 0)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_status_with_provider(self, mock_stdout):
        """Test status command with connected provider."""
        self.interactive_cli.do_status("")
        
        output = mock_stdout.getvalue()
        self.assertIn("Trading provider:", output)
        self.assertIn("Connected", output)
        self.assertIn("TEST123", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_status_without_provider(self, mock_stdout):
        """Test status command without provider."""
        cli_no_provider = WarspiteCLI()
        interactive_no_provider = WarspiteCLIInteractive(cli_no_provider)
        
        interactive_no_provider.do_status("")
        
        output = mock_stdout.getvalue()
        self.assertIn("No trading provider connected", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_dataset_no_dataset(self, mock_stdout):
        """Test dataset command with no dataset loaded."""
        self.interactive_cli.do_dataset("")
        
        output = mock_stdout.getvalue()
        self.assertIn("No dataset loaded", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_dataset_with_dataset(self, mock_stdout):
        """Test dataset command with dataset loaded."""
        # Create and set a mock dataset
        timestamps = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = np.random.randn(10).cumsum() + 100
        data_arrays = [prices]  # 1D array
        symbols = ['AAPL']
        
        mock_dataset = WarspiteDataset(
            data_arrays=data_arrays,
            timestamps=timestamps.values,
            symbols=symbols
        )
        
        self.cli.set_current_dataset(mock_dataset)
        self.interactive_cli.do_dataset("")
        
        output = mock_stdout.getvalue()
        self.assertIn("Current Dataset", output)
        self.assertIn("AAPL", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_symbols(self, mock_stdout):
        """Test symbols command."""
        self.interactive_cli.do_symbols("")
        
        output = mock_stdout.getvalue()
        self.assertIn("Available Symbols", output)
        self.assertIn("EURUSD", output)
        self.assertIn("AAPL", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_do_account(self, mock_stdout):
        """Test account command."""
        self.interactive_cli.do_account("")
        
        output = mock_stdout.getvalue()
        self.assertIn("ACCOUNT INFORMATION", output)
        self.assertIn("TEST123", output)
        self.assertIn("10000.00", output)
    
    def test_do_quit(self):
        """Test quit command."""
        result = self.interactive_cli.do_quit("")
        self.assertTrue(result)
    
    def test_do_exit(self):
        """Test exit command."""
        result = self.interactive_cli.do_exit("")
        self.assertTrue(result)
    
    def test_do_EOF(self):
        """Test Ctrl+D (EOF) handling."""
        result = self.interactive_cli.do_EOF("")
        self.assertTrue(result)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_default_unknown_command(self, mock_stdout):
        """Test handling of unknown commands."""
        self.interactive_cli.default("unknown_command")
        
        output = mock_stdout.getvalue()
        self.assertIn("Unknown command", output)
        self.assertIn("unknown_command", output)
    
    def test_emptyline(self):
        """Test empty line handling."""
        # Should not raise any exception
        result = self.interactive_cli.emptyline()
        self.assertIsNone(result)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_help_commands(self, mock_stdout):
        """Test help system for various commands."""
        # Test help for buy command
        self.interactive_cli.help_buy()
        output = mock_stdout.getvalue()
        self.assertIn("buy - Place a buy order", output)
        
        # Clear output and test help for sell command
        mock_stdout.truncate(0)
        mock_stdout.seek(0)
        
        self.interactive_cli.help_sell()
        output = mock_stdout.getvalue()
        self.assertIn("sell - Place a sell order", output)


class TestCLIErrorHandling(unittest.TestCase):
    """Test error handling in CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_provider = Mock(spec=TradingProvider)
        self.cli = WarspiteCLI(self.mock_provider)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_provider_validation_failure(self, mock_stdout):
        """Test CLI behavior when provider validation fails."""
        # Mock provider that fails validation
        self.mock_provider.get_account_info.side_effect = Exception("Connection failed")
        
        with self.assertRaises(ConnectionError):
            WarspiteCLI(self.mock_provider)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_positions_provider_error(self, mock_stdout):
        """Test get_positions when provider raises exception."""
        self.mock_provider.get_positions.side_effect = Exception("API Error")
        self.mock_provider.get_account_info.return_value = AccountInfo("TEST", 1000.0, "USD")
        
        positions = self.cli.get_positions()
        
        output = mock_stdout.getvalue()
        self.assertIn("Error retrieving positions", output)
        self.assertEqual(positions, [])
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_open_position_provider_error(self, mock_stdout):
        """Test open_position when provider raises exception."""
        self.mock_provider.validate_symbol.return_value = True
        self.mock_provider.get_account_info.return_value = AccountInfo("TEST", 1000.0, "USD")
        self.mock_provider.place_order.side_effect = Exception("Order failed")
        
        result = self.cli.open_position("EURUSD", 1000.0)
        
        output = mock_stdout.getvalue()
        self.assertFalse(result)
        self.assertIn("Error placing order", output)


if __name__ == '__main__':
    unittest.main()