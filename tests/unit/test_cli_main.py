"""
Unit tests for CLI main entry point.

Tests the console script functionality and argument parsing.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import io

from warspite_financial.cli.main import (
    create_argument_parser,
    setup_trading_provider,
    main
)


class TestCLIMain(unittest.TestCase):
    """Test cases for CLI main entry point."""
    
    def test_argument_parser_creation(self):
        """Test that argument parser is created correctly."""
        parser = create_argument_parser()
        
        # Test that parser exists and has expected attributes
        self.assertIsNotNone(parser)
        self.assertEqual(parser.prog, 'warspite-financial-cli')
        
        # Test parsing help (should not raise exception)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--help'])
    
    def test_argument_parser_version(self):
        """Test version argument parsing."""
        parser = create_argument_parser()
        
        with self.assertRaises(SystemExit):
            parser.parse_args(['--version'])
    
    def test_argument_parser_valid_args(self):
        """Test parsing of valid arguments."""
        parser = create_argument_parser()
        
        # Test default arguments
        args = parser.parse_args([])
        self.assertFalse(args.no_interactive)
        self.assertIsNone(args.provider)
        self.assertIsNone(args.account_id)
        
        # Test with provider arguments
        args = parser.parse_args(['--provider', 'demo', '--no-interactive'])
        self.assertTrue(args.no_interactive)
        self.assertEqual(args.provider, 'demo')
        
        # Test OANDA arguments
        args = parser.parse_args([
            '--provider', 'oanda',
            '--account-id', 'test_account'
        ])
        self.assertEqual(args.provider, 'oanda')
        self.assertEqual(args.account_id, 'test_account')
    
    def test_setup_trading_provider_demo(self):
        """Test demo provider setup."""
        provider = setup_trading_provider('demo')
        self.assertIsNone(provider)  # Demo returns None (no real provider)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_setup_trading_provider_oanda_missing_credentials(self, mock_stdout):
        """Test OANDA provider setup with missing environment variable."""
        with patch.dict('os.environ', {}, clear=True):  # Clear environment
            with self.assertRaises(SystemExit) as cm:
                setup_trading_provider('oanda', 'test_account')
        
        self.assertEqual(cm.exception.code, 1)
        output = mock_stdout.getvalue()
        self.assertIn("OANDA_API_TOKEN environment variable is required", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_setup_trading_provider_oanda_missing_account_id(self, mock_stdout):
        """Test OANDA provider setup with missing account ID."""
        with patch.dict('os.environ', {'OANDA_API_TOKEN': 'test_token'}):
            with self.assertRaises(SystemExit) as cm:
                setup_trading_provider('oanda', None)
        
        self.assertEqual(cm.exception.code, 1)
        output = mock_stdout.getvalue()
        self.assertIn("--account-id is required", output)
    
    @patch('warspite_financial.providers.oanda.OANDAProvider')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_setup_trading_provider_oanda_success(self, mock_stdout, mock_oanda_class):
        """Test successful OANDA provider setup."""
        # Mock the OANDA provider
        mock_provider = MagicMock()
        mock_oanda_class.return_value = mock_provider
        
        # Set environment variables
        with patch.dict('os.environ', {
            'OANDA_API_TOKEN': 'test_token',
            'OANDA_ENVIRONMENT': 'practice'
        }):
            provider = setup_trading_provider('oanda', 'test_account')
        
        self.assertEqual(provider, mock_provider)
        mock_oanda_class.assert_called_once_with(
            api_token='test_token',
            account_id='test_account',
            practice=True  # practice mode for 'practice' environment
        )
        
        output = mock_stdout.getvalue()
        self.assertIn("OANDA provider connected successfully", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_setup_trading_provider_oanda_import_error(self, mock_stdout):
        """Test OANDA provider setup with import error."""
        # Set environment variables but simulate import error
        with patch.dict('os.environ', {'OANDA_API_TOKEN': 'test_token'}):
            # Mock the import to fail at the provider level
            import sys
            original_modules = sys.modules.copy()
            if 'warspite_financial.providers.oanda' in sys.modules:
                del sys.modules['warspite_financial.providers.oanda']
            
            try:
                with patch.dict('sys.modules', {'warspite_financial.providers.oanda': None}):
                    with self.assertRaises(SystemExit) as cm:
                        setup_trading_provider('oanda', 'test_account')
            finally:
                sys.modules.update(original_modules)
        
        self.assertEqual(cm.exception.code, 1)
        output = mock_stdout.getvalue()
        self.assertIn("OANDA provider not available", output)
    
    @patch('warspite_financial.cli.main.WarspiteCLI')
    @patch('warspite_financial.cli.main.setup_trading_provider')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_no_interactive(self, mock_stdout, mock_setup_provider, mock_cli_class):
        """Test main function in non-interactive mode."""
        # Mock CLI instance
        mock_cli = MagicMock()
        mock_cli_class.return_value = mock_cli
        mock_setup_provider.return_value = None
        
        # Mock sys.argv
        test_args = ['warspite-financial-cli', '--no-interactive']
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Verify CLI was created and interactive mode was not called
        mock_cli_class.assert_called_once_with(trading_provider=None)
        mock_cli.run_interactive_mode.assert_not_called()
        
        output = mock_stdout.getvalue()
        self.assertIn("CLI initialized successfully", output)
    
    @patch('warspite_financial.cli.main.WarspiteCLI')
    @patch('warspite_financial.cli.main.setup_trading_provider')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_interactive_mode(self, mock_stdout, mock_setup_provider, mock_cli_class):
        """Test main function in interactive mode."""
        # Mock CLI instance
        mock_cli = MagicMock()
        mock_cli_class.return_value = mock_cli
        mock_setup_provider.return_value = None
        
        # Mock sys.argv
        test_args = ['warspite-financial-cli']
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Verify CLI was created and interactive mode was called
        mock_cli_class.assert_called_once_with(trading_provider=None)
        mock_cli.run_interactive_mode.assert_called_once()
        
        output = mock_stdout.getvalue()
        self.assertIn("Starting interactive mode", output)
    
    @patch('warspite_financial.cli.main.WarspiteCLI')
    @patch('warspite_financial.cli.main.setup_trading_provider')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_with_provider(self, mock_stdout, mock_setup_provider, mock_cli_class):
        """Test main function with trading provider."""
        # Mock provider and CLI
        mock_provider = MagicMock()
        mock_cli = MagicMock()
        mock_setup_provider.return_value = mock_provider
        mock_cli_class.return_value = mock_cli
        
        # Mock sys.argv
        test_args = ['warspite-financial-cli', '--provider', 'demo', '--no-interactive']
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Verify provider was set up and CLI was created with it
        mock_setup_provider.assert_called_once_with('demo', None)
        mock_cli_class.assert_called_once_with(trading_provider=mock_provider)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_keyboard_interrupt(self, mock_stdout):
        """Test main function handles keyboard interrupt gracefully."""
        # Mock sys.argv and simulate KeyboardInterrupt
        test_args = ['warspite-financial-cli', '--no-interactive']
        with patch.object(sys, 'argv', test_args):
            with patch('warspite_financial.cli.main.WarspiteCLI', side_effect=KeyboardInterrupt):
                with self.assertRaises(SystemExit) as cm:
                    main()
                
                self.assertEqual(cm.exception.code, 0)
        
        output = mock_stdout.getvalue()
        self.assertIn("CLI interrupted by user", output)
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_general_exception(self, mock_stdout):
        """Test main function handles general exceptions."""
        # Mock sys.argv and simulate general exception
        test_args = ['warspite-financial-cli', '--no-interactive']
        with patch.object(sys, 'argv', test_args):
            with patch('warspite_financial.cli.main.WarspiteCLI', side_effect=Exception("Test error")):
                with self.assertRaises(SystemExit) as cm:
                    main()
                
                self.assertEqual(cm.exception.code, 1)
        
        output = mock_stdout.getvalue()
        self.assertIn("Fatal error: Test error", output)


if __name__ == '__main__':
    unittest.main()