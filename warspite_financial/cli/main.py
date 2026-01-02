#!/usr/bin/env python3
"""
Main entry point for the warspite-financial CLI.

This module provides the console script entry point for the warspite-financial-cli command.
"""

import sys
import argparse
from typing import Optional
from .cli import WarspiteCLI


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='warspite-financial-cli',
        description='Warspite Financial CLI - Interactive trading and dataset visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  warspite-financial-cli                    # Start interactive mode
  warspite-financial-cli --version          # Show version information
  warspite-financial-cli --help             # Show this help message

Interactive Commands:
  positions     - View current trading positions
  buy <symbol> <quantity>   - Place buy orders
  sell <symbol> <quantity>  - Place sell orders
  close <id>    - Close specific position
  close all     - Close all positions
  account       - Show account information
  symbols       - List available symbols
  dataset       - Show dataset information
  dataset render - Render ASCII chart
  help          - Show command help
  quit          - Exit interactive mode

Environment Variables for OANDA:
  OANDA_API_TOKEN    - Your OANDA API token (required for live trading)
  OANDA_ENVIRONMENT  - 'practice' or 'live' (default: practice)

Security Note:
  API tokens should be set as environment variables, not passed as command line
  arguments. This prevents credentials from appearing in shell history or
  process lists.

For more information, visit: https://github.com/warspite-financial/warspite-financial
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Start CLI without entering interactive mode (for scripting)'
    )
    
    parser.add_argument(
        '--provider',
        choices=['oanda', 'demo'],
        help='Trading provider to use (oanda for live trading, demo for simulation)'
    )
    
    parser.add_argument(
        '--account-id',
        help='Account ID for trading provider (required for OANDA)'
    )
    
    return parser


def setup_trading_provider(provider_type: str, account_id: Optional[str] = None):
    """
    Set up trading provider based on command line arguments and environment variables.
    
    Args:
        provider_type: Type of provider ('oanda' or 'demo')
        account_id: Account ID for the provider
        
    Returns:
        Configured trading provider instance or None
    """
    if provider_type == 'demo':
        # Create a demo/mock provider for testing
        print("🔧 Setting up demo trading provider...")
        print("⚠️  Demo mode: No real trades will be executed")
        return None  # CLI will work without provider for demo
    
    elif provider_type == 'oanda':
        import os
        
        # Get API token from environment variable
        api_token = os.getenv('OANDA_API_TOKEN')
        environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if not api_token:
            print("❌ Error: OANDA_API_TOKEN environment variable is required")
            print("   Set your API token with: export OANDA_API_TOKEN=your_token_here")
            print("   Get your token from: https://developer.oanda.com/")
            sys.exit(1)
        
        if not account_id:
            print("❌ Error: --account-id is required for OANDA provider")
            print("   Usage: warspite-financial-cli --provider oanda --account-id YOUR_ACCOUNT_ID")
            sys.exit(1)
        
        try:
            from ..providers.oanda import OANDAProvider
            print("🔧 Setting up OANDA trading provider...")
            print(f"   Environment: {environment}")
            print(f"   Account ID: {account_id}")
            
            # Convert environment string to practice boolean
            practice_mode = environment.lower() != 'live'
            
            provider = OANDAProvider(
                api_token=api_token, 
                account_id=account_id,
                practice=practice_mode
            )
            print("✅ OANDA provider connected successfully")
            return provider
        except ImportError:
            print("❌ Error: OANDA provider not available")
            print("   Install with: pip install warspite-financial[providers]")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error connecting to OANDA: {e}")
            print("   Check your API token and account ID")
            print("   Ensure OANDA_API_TOKEN environment variable is set correctly")
            sys.exit(1)
    
    return None


def print_welcome_banner():
    """Print the welcome banner for the CLI."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Warspite Financial CLI                              ║
║                     Interactive Trading Interface                            ║
║                                                                              ║
║  Type 'help' for available commands                                          ║
║  Type 'quit' to exit                                                         ║
║                                                                              ║
║  ⚠️  WARNING: Live trading uses real money. Use demo mode for testing.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def print_quick_start_guide():
    """Print a quick start guide for new users."""
    print("""
🚀 Quick Start Guide:

1. Check connection status:
   warspite> status

2. View available symbols:
   warspite> symbols

3. Check account balance:
   warspite> account

4. View current positions:
   warspite> positions

5. Place a demo trade:
   warspite> buy EURUSD 100

6. Get help on any command:
   warspite> help buy

7. Exit when done:
   warspite> quit

For dataset visualization, load a dataset first using the Python API.
""")


def main():
    """
    Main entry point for the warspite-financial-cli command.
    
    This function handles command line arguments, sets up the trading provider,
    and launches the interactive CLI.
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Print welcome banner
        print_welcome_banner()
        
        # Set up trading provider if specified
        trading_provider = None
        if args.provider:
            trading_provider = setup_trading_provider(
                args.provider, 
                args.account_id
            )
        else:
            print("ℹ️  No trading provider specified. Starting in visualization-only mode.")
            print("   Use --provider oanda or --provider demo for trading functionality.")
        
        # Create CLI instance
        cli = WarspiteCLI(trading_provider=trading_provider)
        
        # Show quick start guide for new users
        if not args.no_interactive:
            print_quick_start_guide()
        
        # Start interactive mode unless disabled
        if not args.no_interactive:
            print("🎯 Starting interactive mode...")
            cli.run_interactive_mode()
        else:
            print("✅ CLI initialized successfully (non-interactive mode)")
            print("   Use the CLI instance programmatically or call run_interactive_mode()")
    
    except KeyboardInterrupt:
        print("\n\n👋 CLI interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("   Please report this issue at: https://github.com/warspite-financial/warspite-financial/issues")
        sys.exit(1)


if __name__ == '__main__':
    main()