"""
CLI integration tests for warspite_financial library.

These tests verify CLI integration with trading providers and end-to-end workflows.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from datetime import datetime, timedelta

from warspite_financial import (
    BrownianMotionProvider,
    WarspiteDataset,
    WarspiteCLI,
    create_dataset_from_provider
)
from warspite_financial.cli.cli import WarspiteCLIInteractive
from warspite_financial.cli.main import (
    main,
    create_argument_parser,
    setup_trading_provider,
    print_welcome_banner,
    print_quick_start_guide
)
from warspite_financial.utils.exceptions import WarspiteError

# Try to import optional providers
try:
    from warspite_financial import OANDAProvider
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False


class TestCLIIntegration:
    """Test CLI integration with various components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BrownianMotionProvider()
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
        # Create test dataset
        self.dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL', 'GOOGL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
    def test_cli_initialization_without_provider(self):
        """Test CLI initialization without trading provider."""
        cli = WarspiteCLI()
        
        # Should initialize successfully
        assert cli is not None
        assert cli._trading_provider is None
        assert hasattr(cli, '_current_dataset')
        
    def test_cli_initialization_with_provider(self):
        """Test CLI initialization with trading provider."""
        mock_provider = Mock()
        mock_provider.get_account_info.return_value = Mock(
            account_id='test_account',
            balance=10000.0,
            currency='USD'
        )
        mock_provider.get_positions.return_value = []
        
        cli = WarspiteCLI(trading_provider=mock_provider)
        
        assert cli._trading_provider is mock_provider
        assert hasattr(cli, '_current_dataset')
        
    def test_cli_dataset_integration(self):
        """Test CLI integration with dataset operations."""
        cli = WarspiteCLI()
        
        # Load dataset using the correct method
        cli.set_current_dataset(self.dataset)
        
        # Verify dataset is loaded
        current_dataset = cli.get_current_dataset()
        assert current_dataset is not None
        assert len(current_dataset.symbols) == 2
        assert 'AAPL' in current_dataset.symbols
        assert 'GOOGL' in current_dataset.symbols
        
    def test_cli_ascii_rendering_integration(self):
        """Test CLI ASCII rendering with real dataset."""
        cli = WarspiteCLI()
        cli.set_current_dataset(self.dataset)
        
        # Test ASCII rendering - the method doesn't return a string, it prints
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.render_dataset_ascii(self.dataset)
            output = mock_stdout.getvalue()
            
        assert isinstance(output, str)
        assert len(output) > 0
        
    def test_cli_position_management_without_provider(self):
        """Test position management commands without trading provider."""
        cli = WarspiteCLI()
        
        # Should handle gracefully when no provider
        positions = cli.get_positions()
        assert positions == []
        
        # Should return False for trading operations
        result = cli.open_position('AAPL', 100, 'market')
        assert result is False
        
    def test_cli_position_management_with_mock_provider(self):
        """Test position management with mock trading provider."""
        mock_provider = Mock()
        mock_provider.get_account_info.return_value = Mock(
            account_id='test_account',
            balance=10000.0,
            currency='USD'
        )
        mock_provider.get_positions.return_value = [
            Mock(position_id='1', symbol='AAPL', quantity=100, side='buy', 
                 current_price=150.0, unrealized_pnl=500.0),
            Mock(position_id='2', symbol='GOOGL', quantity=50, side='sell',
                 current_price=2800.0, unrealized_pnl=-200.0)
        ]
        mock_provider.place_order.return_value = Mock(order_id='12345', status='filled')
        
        cli = WarspiteCLI(trading_provider=mock_provider)
        
        # Test get positions - this prints to stdout, so we need to capture it
        with patch('sys.stdout', new_callable=StringIO):
            positions = cli.get_positions()
            # The method returns the actual positions from the provider
            assert len(positions) == 2
        
        # Test open position
        result = cli.open_position('MSFT', 75, 'market')
        mock_provider.place_order.assert_called_once()
        assert result is True  # Should return True on success
        
    def test_cli_interactive_commands(self):
        """Test CLI interactive command processing."""
        cli = WarspiteCLI()
        cli.set_current_dataset(self.dataset)
        
        # Create interactive CLI instance
        interactive_cli = WarspiteCLIInteractive(cli)
        
        # Test help command
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.do_help('')
            help_output = mock_stdout.getvalue()
            assert len(help_output) > 0
            
        # Test dataset command
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.do_dataset('')
            dataset_output = mock_stdout.getvalue()
            assert 'AAPL' in dataset_output
            assert 'GOOGL' in dataset_output
            
    def test_cli_command_validation(self):
        """Test CLI command validation and error handling."""
        cli = WarspiteCLI()
        interactive_cli = WarspiteCLIInteractive(cli)
        
        # Test invalid commands are handled gracefully
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.default('invalid_command_xyz')
            output = mock_stdout.getvalue()
            assert 'unknown' in output.lower() or 'invalid' in output.lower()


class TestCLIMainIntegration:
    """Test CLI main entry point integration."""
    
    def test_argument_parser_creation(self):
        """Test argument parser creation and configuration."""
        parser = create_argument_parser()
        
        # Test parser exists and has expected arguments
        assert parser is not None
        assert parser.prog == 'warspite-financial-cli'
        
        # Test parsing valid arguments (avoid --version as it exits)
        args = parser.parse_args(['--provider', 'demo'])
        assert args.provider == 'demo'
        
        args = parser.parse_args(['--no-interactive'])
        assert args.no_interactive is True
        
    def test_demo_provider_setup(self):
        """Test demo provider setup."""
        provider = setup_trading_provider('demo')
        
        # Demo provider should return None (CLI handles demo mode internally)
        assert provider is None
        
    @patch.dict(os.environ, {'OANDA_API_TOKEN': 'test_token', 'OANDA_ENVIRONMENT': 'practice'})
    def test_oanda_provider_setup_success(self):
        """Test successful OANDA provider setup."""
        # Mock the import and provider creation
        with patch('warspite_financial.providers.oanda.OANDAProvider') as mock_oanda_class:
            mock_provider = Mock()
            mock_oanda_class.return_value = mock_provider
            
            provider = setup_trading_provider('oanda', 'test_account_id')
            
            assert provider is mock_provider
            mock_oanda_class.assert_called_once_with(
                api_token='test_token',
                account_id='test_account_id',
                practice=True
            )
        
    def test_oanda_provider_setup_missing_token(self):
        """Test OANDA provider setup with missing API token."""
        # Ensure no token in environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                setup_trading_provider('oanda', 'test_account_id')
                
    def test_oanda_provider_setup_missing_account_id(self):
        """Test OANDA provider setup with missing account ID."""
        with patch.dict(os.environ, {'OANDA_API_TOKEN': 'test_token'}):
            with pytest.raises(SystemExit):
                setup_trading_provider('oanda', None)
                
    @patch('warspite_financial.cli.main.print_welcome_banner')
    @patch('warspite_financial.cli.main.print_quick_start_guide')
    @patch('warspite_financial.cli.main.WarspiteCLI')
    def test_main_function_demo_mode(self, mock_cli_class, mock_quick_start, mock_banner):
        """Test main function in demo mode."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['warspite-financial-cli', '--provider', 'demo']):
            main()
            
        # Verify CLI was created and started
        mock_cli_class.assert_called_once()
        mock_cli.run_interactive_mode.assert_called_once()
        mock_banner.assert_called_once()
        mock_quick_start.assert_called_once()
        
    @patch('warspite_financial.cli.main.print_welcome_banner')
    @patch('warspite_financial.cli.main.WarspiteCLI')
    def test_main_function_no_interactive(self, mock_cli_class, mock_banner):
        """Test main function in non-interactive mode."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['warspite-financial-cli', '--no-interactive']):
            main()
            
        # Verify CLI was created but interactive mode not started
        mock_cli_class.assert_called_once()
        mock_cli.run_interactive_mode.assert_not_called()
        mock_banner.assert_called_once()
        
    def test_welcome_banner_output(self):
        """Test welcome banner output."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_welcome_banner()
            output = mock_stdout.getvalue()
            
        assert 'Warspite Financial CLI' in output
        assert 'help' in output.lower()
        assert 'quit' in output.lower()
        
    def test_quick_start_guide_output(self):
        """Test quick start guide output."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_quick_start_guide()
            output = mock_stdout.getvalue()
            
        assert 'Quick Start' in output
        assert 'status' in output
        assert 'positions' in output
        assert 'account' in output


class TestCLIEndToEndWorkflows:
    """Test complete CLI workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BrownianMotionProvider()
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=20)
        
    def test_complete_cli_workflow_demo_mode(self):
        """Test complete CLI workflow in demo mode."""
        # Create dataset
        dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        # Initialize CLI
        cli = WarspiteCLI()
        cli.set_current_dataset(dataset)
        interactive_cli = WarspiteCLIInteractive(cli)
        
        # Test workflow steps
        # 1. Check dataset info
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.do_dataset('')
            output = mock_stdout.getvalue()
            assert 'AAPL' in output
            
        # 2. Render ASCII chart
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.render_dataset_ascii(dataset)
            output = mock_stdout.getvalue()
            assert isinstance(output, str)
        
        # 3. Check positions (should be empty in demo)
        positions = cli.get_positions()
        assert positions == []
        
        # 4. Test help system
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.do_help('')
            help_output = mock_stdout.getvalue()
            assert len(help_output) > 0
            
    def test_cli_workflow_with_mock_trading(self):
        """Test CLI workflow with mock trading provider."""
        # Create mock provider
        mock_provider = Mock()
        mock_provider.get_account_info.return_value = Mock(
            account_id='test_account',
            balance=10000.0,
            currency='USD'
        )
        mock_provider.get_positions.return_value = []
        mock_provider.place_order.return_value = Mock(
            order_id='12345',
            status='filled',
            symbol='AAPL',
            quantity=100
        )
        
        # Initialize CLI with provider
        cli = WarspiteCLI(trading_provider=mock_provider)
        
        # Test trading workflow
        # 1. Check account info - this is done during initialization
        
        # 2. Check initial positions
        with patch('sys.stdout', new_callable=StringIO):
            positions = cli.get_positions()
            assert len(positions) == 0
        
        # 3. Place order
        result = cli.open_position('AAPL', 100, 'market')
        mock_provider.place_order.assert_called_once()
        assert result is True
        
        # 4. Verify provider methods were called
        mock_provider.get_account_info.assert_called()
        mock_provider.get_positions.assert_called()
        
    def test_cli_error_handling_workflow(self):
        """Test CLI error handling in various scenarios."""
        cli = WarspiteCLI()
        interactive_cli = WarspiteCLIInteractive(cli)
        
        # Test operations without dataset
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            interactive_cli.do_dataset('')
            output = mock_stdout.getvalue()
            assert 'no dataset' in output.lower() or 'not loaded' in output.lower()
            
        # Test ASCII rendering without dataset
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            try:
                cli.render_dataset_ascii(None)
            except Exception:
                pass  # Expected to fail
            
        # Test trading operations without provider
        result = cli.open_position('AAPL', 100, 'market')
        assert result is False
        
    def test_cli_command_line_integration(self):
        """Test CLI command line argument integration."""
        # Test argument parsing
        parser = create_argument_parser()
        
        # Test various argument combinations
        test_cases = [
            ['--provider', 'demo'],
            ['--provider', 'demo', '--no-interactive'],
            ['--account-id', 'test123'],
            ['--help']  # This would normally exit
        ]
        
        for args in test_cases[:-1]:  # Skip --help as it exits
            parsed_args = parser.parse_args(args)
            assert parsed_args is not None
            
    @patch('warspite_financial.cli.main.WarspiteCLI')
    def test_main_function_keyboard_interrupt(self, mock_cli_class):
        """Test main function handling keyboard interrupt."""
        mock_cli = Mock()
        mock_cli.run_interactive_mode.side_effect = KeyboardInterrupt()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['warspite-financial-cli']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0  # Should exit gracefully
            
    @patch('warspite_financial.cli.main.WarspiteCLI')
    def test_main_function_exception_handling(self, mock_cli_class):
        """Test main function handling unexpected exceptions."""
        mock_cli_class.side_effect = Exception("Test error")
        
        with patch('sys.argv', ['warspite-financial-cli']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1  # Should exit with error


@pytest.mark.skipif(not OANDA_AVAILABLE, reason="OANDA provider not available")
class TestCLIOANDAIntegration:
    """Test CLI integration with OANDA provider."""
    
    def test_cli_oanda_provider_interface(self):
        """Test CLI interface with OANDA provider."""
        # This test verifies the interface without requiring real credentials
        try:
            # Create provider with dummy credentials
            from warspite_financial import OANDAProvider
            provider = OANDAProvider(api_token="dummy", account_id="dummy")
            
            # Initialize CLI
            cli = WarspiteCLI(trading_provider=provider)
            
            # Verify CLI accepts OANDA provider
            assert cli.trading_provider is provider
            assert hasattr(cli.trading_provider, 'place_order')
            assert hasattr(cli.trading_provider, 'get_positions')
            assert hasattr(cli.trading_provider, 'get_account_info')
            
        except Exception:
            # Skip if provider can't be created without real credentials
            pytest.skip("OANDA provider requires valid credentials")
            
    def test_cli_oanda_setup_integration(self):
        """Test CLI setup with OANDA provider integration."""
        with patch('warspite_financial.providers.oanda.OANDAProvider') as mock_oanda_class:
            mock_provider = Mock()
            mock_oanda_class.return_value = mock_provider
            
            # Test provider setup
            provider = setup_trading_provider('oanda', 'test_account')
            assert provider is mock_provider
            
            # Test CLI initialization with provider - need to mock account info
            mock_provider.get_account_info.return_value = Mock(
                account_id='test_account',
                balance=10000.0,
                currency='USD'
            )
            mock_provider.get_positions.return_value = []
            
            cli = WarspiteCLI(trading_provider=provider)
            assert cli._trading_provider is provider


if __name__ == '__main__':
    pytest.main([__file__])