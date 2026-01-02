# warspite-financial

A comprehensive Python library for financial data processing, trading strategies, and emulation.

## Overview

The warspite_financial library provides a modular framework for:

- Loading financial data from various providers (Yahoo Finance, OANDA, synthetic data)
- Applying trading strategies to market data
- Simulating trading scenarios with historical data
- Executing real trades through supported trading providers
- Visualizing trading results and performance metrics

## Installation

### From PyPI

```bash
pip install warspite-financial
```

After installation, the CLI is available as a console command:

```bash
warspite-financial-cli --help
```

### With Optional Dependencies

```bash
# For additional data providers
pip install warspite-financial[providers]

# For development
pip install warspite-financial[dev]

# For everything
pip install warspite-financial[dev,providers]
```

### Development Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd warspite-financial
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e .[dev,providers]
```

## Quick Start

### Using the Console Script (Easiest)

After installation, you can immediately start the CLI:

```bash
# Start interactive CLI
warspite-financial-cli

# Or with demo trading
warspite-financial-cli --provider demo
```

### Using Python API

#### Basic Backtesting Example

```python
from warspite_financial import (
    BrownianMotionProvider,
    SMAStrategy,
    WarspiteTradingEmulator,
    create_dataset_from_provider,
    run_strategy_backtest
)
from datetime import datetime, timedelta

# Create synthetic data for testing
provider = BrownianMotionProvider()
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

# Load data
dataset = create_dataset_from_provider(
    provider=provider,
    symbols=['BM-AAPL'],  # BM- prefix for synthetic data
    start_date=start_date,
    end_date=end_date
)

# Create and test a strategy
strategy = SMAStrategy(period=20)
result = run_strategy_backtest(
    dataset=dataset,
    strategy=strategy,
    initial_capital=10000
)

print(f"Final portfolio value: ${result.final_portfolio_value:.2f}")
print(f"Total return: {((result.final_portfolio_value - 10000) / 10000) * 100:.2f}%")
```

#### Using Real Data (with yfinance)

```python
from warspite_financial import YFinanceProvider
from datetime import datetime, timedelta

# Create provider for real data
provider = YFinanceProvider()

# Load real market data
dataset = create_dataset_from_provider(
    provider=provider,
    symbols=['AAPL'],
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# Run the same strategy on real data
result = run_strategy_backtest(dataset, SMAStrategy(period=10))
```

#### Visualization

```python
from warspite_financial import create_visualization

# Create matplotlib chart
chart = create_visualization(
    dataset=dataset,
    renderer_type='matplotlib',
    title='Stock Price Analysis'
)
chart.show()

# Create ASCII chart for terminal
ascii_chart = create_visualization(
    dataset=dataset,
    renderer_type='ascii'
)
print(ascii_chart)
```

## Forecasting

The warspite_financial library includes heuristic forecasting capabilities for predicting future price movements based on historical patterns.

### Basic Forecasting

```python
from warspite_financial import (
    WarspiteHeuristicForecaster,
    create_dataset_from_provider,
    BrownianMotionProvider
)
from datetime import datetime, timedelta

# Create historical dataset
provider = BrownianMotionProvider()
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

dataset = create_dataset_from_provider(
    provider=provider,
    symbols=['BM-AAPL'],
    start_date=start_date,
    end_date=end_date
)

# Create forecaster
forecaster = WarspiteHeuristicForecaster(dataset)

# Generate 10-day forecast using linear extrapolation
forecast = forecaster.forecast(periods=10, method='linear')

print(f"Forecast generated for {len(forecast.timestamps)} periods")
print(f"Forecast symbols: {forecast.symbols}")
```

### Forecasting Methods

The library supports multiple forecasting methods:

```python
# Linear extrapolation (trend-based)
linear_forecast = forecaster.forecast(periods=5, method='linear')

# Exponential smoothing (weighted recent data)
exponential_forecast = forecaster.forecast(periods=5, method='exponential')

# Seasonal pattern recognition
seasonal_forecast = forecaster.forecast(periods=5, method='seasonal')
```

### Confidence Intervals

Get confidence intervals for forecast reliability:

```python
# Generate forecast
forecast = forecaster.forecast(periods=7, method='linear')

# Get confidence intervals for the forecast
confidence_intervals = forecaster.get_confidence_intervals()

print(f"Confidence intervals shape: {confidence_intervals.shape}")
print("Forecast includes uncertainty bounds for risk assessment")
```

### Forecasting with Trading Strategies

Combine forecasting with trading strategies for forward-looking analysis:

```python
from warspite_financial import SMAStrategy, WarspiteTradingEmulator

# Generate forecast
forecast = forecaster.forecast(periods=15, method='exponential')

# Apply strategy to forecast data
strategy = SMAStrategy(period=5)  # Use shorter period for forecast data
emulator = WarspiteTradingEmulator(
    dataset=forecast,
    initial_capital=10000,
    trading_fee=0.001,
    spread=0.0001
)
emulator.add_strategy(strategy)

# Run strategy on forecasted data
forecast_result = emulator.run_to_completion()

print(f"Forecasted strategy performance: {forecast_result.total_return:.2%}")
print(f"Projected final value: ${forecast_result.final_portfolio_value:.2f}")
```

### Combined Historical and Forecast Analysis

Analyze both historical performance and future projections:

```python
# Step 1: Backtest on historical data
historical_strategy = SMAStrategy(period=10)
historical_result = run_strategy_backtest(
    dataset=dataset,
    strategy=historical_strategy,
    initial_capital=10000
)

# Step 2: Generate forecast
forecast = forecaster.forecast(periods=10, method='linear')

# Step 3: Apply strategy to forecast using historical result as starting capital
forecast_emulator = WarspiteTradingEmulator(
    dataset=forecast,
    initial_capital=historical_result.final_portfolio_value,
    trading_fee=0.001,
    spread=0.0001
)
forecast_emulator.add_strategy(SMAStrategy(period=5))
forecast_result = forecast_emulator.run_to_completion()

print("=== Combined Analysis ===")
print(f"Historical return: {historical_result.total_return:.2%}")
print(f"Forecasted return: {forecast_result.total_return:.2%}")
print(f"Total projected value: ${forecast_result.final_portfolio_value:.2f}")
```

### Forecasting Visualization

Visualize forecasts alongside historical data:

```python
from warspite_financial import create_visualization

# Create visualization of forecast
forecast_chart = create_visualization(
    dataset=forecast,
    renderer_type='matplotlib',
    title='Price Forecast - Next 10 Days'
)
forecast_chart.show()

# ASCII visualization for terminal
ascii_forecast = create_visualization(
    dataset=forecast,
    renderer_type='ascii'
)
print("=== Forecast Visualization ===")
print(ascii_forecast)
```

### Forecasting Best Practices

1. **Use Sufficient Historical Data**: Ensure your dataset has enough historical data for reliable pattern recognition
2. **Choose Appropriate Methods**: Linear for trending markets, exponential for volatile markets, seasonal for cyclical patterns
3. **Consider Confidence Intervals**: Always assess forecast uncertainty
4. **Validate with Backtesting**: Test forecasting methods on historical data first
5. **Combine with Risk Management**: Use forecasts as one input among many for trading decisions

### Forecasting Limitations

- **Heuristic Methods**: These are pattern-based forecasts, not predictive models
- **Market Uncertainty**: Financial markets are inherently unpredictable
- **Historical Bias**: Past patterns may not continue in the future
- **Use for Analysis**: Forecasts should inform analysis, not drive automatic trading decisions
```

#### Complete Workflow Example

```python
from warspite_financial import basic_backtest_example

# Run a complete backtest with visualization
result = basic_backtest_example(
    symbols=['BM-AAPL'],
    days=30,
    initial_capital=5000,
    sma_period=15
)

print(f"Strategy performance: {result['total_return']:.2%}")
print(f"Number of trades: {result['num_trades']}")

# Display the chart
result['chart'].show()
```

## Command Line Interface (CLI)

The warspite_financial library includes a powerful command-line interface for interactive trading and dataset visualization. The CLI provides both programmatic access and an interactive terminal mode.

### Getting Started with CLI

```python
from warspite_financial import WarspiteCLI
from warspite_financial.providers import OANDAProvider

# Create CLI without trading provider (visualization only)
cli = WarspiteCLI()

# Or with trading provider for live trading
provider = OANDAProvider(api_token="your_token", account_id="your_account")
cli = WarspiteCLI(trading_provider=provider)
```

### Interactive Mode

Launch the interactive CLI for real-time trading operations:

```python
# Start interactive mode programmatically
cli.run_interactive_mode()
```

**Or use the console script directly from terminal:**

```bash
# Start CLI in interactive mode
warspite-financial-cli

# Start with demo trading provider
warspite-financial-cli --provider demo

# Start with OANDA live trading (requires environment variables)
export OANDA_API_TOKEN=your_token_here
export OANDA_ENVIRONMENT=practice  # or 'live'
warspite-financial-cli --provider oanda --account-id YOUR_ACCOUNT

# Show help and available options
warspite-financial-cli --help

# Show version
warspite-financial-cli --version
```

**Security Note:** For OANDA live trading, API tokens should be set as environment variables rather than command line arguments to prevent credentials from appearing in shell history or process lists.

This opens an interactive terminal with the following commands:

#### Trading Commands

```bash
# View current positions
warspite> positions

# Place buy orders
warspite> buy EURUSD 1000
warspite> buy AAPL 10 limit

# Place sell orders  
warspite> sell EURUSD 500
warspite> sell AAPL 5 market

# Close specific positions
warspite> close POS_12345

# Close all positions
warspite> close all

# Check account status
warspite> account
warspite> status
```

#### Dataset Visualization

```bash
# Show current dataset info
warspite> dataset

# Render ASCII chart of loaded dataset
warspite> dataset render
```

#### Information Commands

```bash
# List available trading symbols
warspite> symbols

# Get help on commands
warspite> help
warspite> help buy

# Clear screen
warspite> clear

# Exit interactive mode
warspite> quit
```

### ASCII Dataset Rendering

The CLI can render datasets as ASCII charts directly in the terminal:

```python
from warspite_financial import create_dataset_from_provider, BrownianMotionProvider
from datetime import datetime, timedelta

# Create sample dataset
provider = BrownianMotionProvider()
dataset = create_dataset_from_provider(
    provider=provider,
    symbols=['BM-AAPL'],
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# Load dataset into CLI
cli.set_current_dataset(dataset)

# Render ASCII chart
cli.render_dataset_ascii(dataset, width=80, height=20)
```

**Example ASCII Output:**
```
📊 ASCII Chart - Dataset Visualization
================================================================================

BM-AAPL Price
Range: 95.2341 - 104.7829
104.7829 │●──                                                               ●
103.1234 │   ──                                                         ──── 
101.4639 │     ──                                               ────────     
 99.8044 │       ───                                     ───────             
 98.1449 │          ────                           ──────                    
 96.4854 │              ─────               ──────                          
 94.8259 │                   ──────────────                                 
          └─────────────────────────────────────────────────────────────────
           11/26                                                      12/26
================================================================================
```

### Position Management

The CLI provides comprehensive position management with safety features:

```python
# Open positions with validation
success = cli.open_position("EURUSD", 1000.0, order_type="market")

# Close specific position with confirmation
success = cli.close_position("POS_12345")

# Close all positions with safety prompt
success = cli.close_all_positions()

# Get detailed position information
positions = cli.get_positions()
```

### CLI Safety Features

The CLI includes several safety mechanisms for live trading:

- **Risk Validation**: Warns when order size exceeds account balance
- **Confirmation Prompts**: Requires confirmation for large trades or losses
- **Position Limits**: Prevents excessive position sizes
- **Error Handling**: Graceful handling of connection and API errors
- **Real-time Feedback**: Immediate confirmation of trade execution

### Example CLI Session

```python
from warspite_financial import WarspiteCLI
from warspite_financial.providers import OANDAProvider
import os

# Set up environment variables for security
os.environ['OANDA_API_TOKEN'] = 'your_demo_token'
os.environ['OANDA_ENVIRONMENT'] = 'practice'

# Set up CLI with trading provider
provider = OANDAProvider(
    api_token=os.environ['OANDA_API_TOKEN'],
    account_id="demo_account",
    environment=os.environ['OANDA_ENVIRONMENT']
)
cli = WarspiteCLI(trading_provider=provider)

# Start interactive session
cli.run_interactive_mode()
```

**Or using the console script:**

```bash
export OANDA_API_TOKEN=your_demo_token
export OANDA_ENVIRONMENT=practice
warspite-financial-cli --provider oanda --account-id demo_account
```

**Interactive Session:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Warspite Financial CLI                              ║
║                     Interactive Trading Interface                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Type 'help' or '?' to list commands.

warspite> status
✅ Trading provider: OANDAProvider
   Account ID: demo_account
   Balance: 10000.00 USD
   Status: Connected

warspite> buy EURUSD 1000
🔄 Placing BUY order for 1000.0 of EURUSD...
✅ Order executed successfully!
   Order ID: ORDER_001
   Action: BUY
   Symbol: EURUSD
   Quantity: 1000.0

warspite> positions
======================================================================
CURRENT POSITIONS
======================================================================
Account: demo_account
Balance: 9998.50 USD
Timestamp: 2023-12-26 14:30:15
----------------------------------------------------------------------
Position ID     Symbol     Quantity     Price      P&L        %
----------------------------------------------------------------------
POS_001         EURUSD     1000.0000    1.2000     25.00      2.08%
----------------------------------------------------------------------
Total P&L:                                         25.00      0.25%
======================================================================

warspite> quit
👋 Goodbye!
```

### Programmatic CLI Usage

You can also use CLI methods programmatically:

```python
# Check if trading provider is connected
if cli._trading_provider:
    # Get current positions
    positions = cli.get_positions()
    
    # Open a position
    if cli.open_position("EURUSD", 500.0):
        print("Position opened successfully")
    
    # Render current dataset if loaded
    if cli.get_current_dataset():
        cli.render_dataset_ascii(cli.get_current_dataset())
```

### CLI Configuration

The CLI can be customized for different use cases:

```python
# CLI for visualization only (no trading)
viz_cli = WarspiteCLI()
viz_cli.set_current_dataset(your_dataset)
viz_cli.render_dataset_ascii(your_dataset, width=120, height=30)

# CLI with different trading providers
from warspite_financial.providers import YFinanceProvider

# Note: YFinanceProvider doesn't support trading, only data
data_provider = YFinanceProvider()
data_cli = WarspiteCLI()  # No trading functionality

# For live trading, use TradingProvider implementations
trading_cli = WarspiteCLI(trading_provider=oanda_provider)
```

## Features

### Core Components

- **Minimal Dependencies**: Core functionality requires only numpy, pandas, and matplotlib
- **Extensible Provider System**: Support for multiple data sources with a common interface
- **Strategy Framework**: Implement custom trading strategies with a simple interface
- **Trading Emulation**: Backtest strategies with realistic trading costs and constraints
- **Live Trading**: Execute strategies with real money through supported brokers (OANDA)
- **Command Line Interface**: Interactive terminal interface for trading and visualization
- **Visualization**: Generate charts and performance reports in multiple formats

### Available Providers

- **BrownianMotionProvider**: Synthetic data generation for testing (always available)
- **YFinanceProvider**: Yahoo Finance data (requires `yfinance` package)
- **OANDAProvider**: OANDA forex data and live trading (requires API credentials)

### Built-in Strategies

- **SMAStrategy**: Simple Moving Average crossover strategy
- **RandomStrategy**: Random trading for baseline comparison
- **PerfectStrategy**: Optimal strategy with future knowledge (for benchmarking)

### Visualization Options

- **MatplotlibRenderer**: Interactive charts with matplotlib
- **ASCIIRenderer**: Terminal-based charts
- **PDFRenderer**: Export charts to PDF
- **CSVRenderer**: Export data to CSV format

## Architecture

The library follows a modular architecture with clear separation between:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│         (Examples, Convenience Functions)                   │
├─────────────────────────────────────────────────────────────┤
│                   Strategy Layer                            │
│     (SMAStrategy, RandomStrategy, PerfectStrategy)          │
├─────────────────────────────────────────────────────────────┤
│                   Emulation Layer                           │
│              (WarspiteTradingEmulator)                      │
├─────────────────────────────────────────────────────────────┤
│                    Dataset Layer                            │
│              (WarspiteDataset)                              │
├─────────────────────────────────────────────────────────────┤
│                   Provider Layer                            │
│    (BaseProvider, YFinanceProvider, OANDAProvider)          │
└─────────────────────────────────────────────────────────────┘
```

## Advanced Usage

### Custom Strategy Implementation

```python
from warspite_financial import BaseStrategy
import numpy as np

class CustomStrategy(BaseStrategy):
    def __init__(self, threshold=0.02):
        self.threshold = threshold
    
    def generate_positions(self, dataset):
        # Implement your trading logic
        prices = dataset.data[:, 3]  # Close prices
        returns = np.diff(prices) / prices[:-1]
        
        positions = np.zeros(len(dataset.timestamps))
        for i in range(1, len(positions)):
            if returns[i-1] > self.threshold:
                positions[i] = 1.0  # Long position
            elif returns[i-1] < -self.threshold:
                positions[i] = -1.0  # Short position
        
        return positions
    
    def get_parameters(self):
        return {'threshold': self.threshold}
```

### Live Trading (OANDA)

```python
from warspite_financial import OANDAProvider, live_trading_example

# Set up OANDA provider with credentials
provider = OANDAProvider(
    api_token="your_api_token",
    account_id="your_account_id"
)

# Run live trading example (dry run by default)
result = live_trading_example(
    trading_provider=provider,
    symbols=['EUR_USD'],
    strategy_type='sma',
    dry_run=True  # Set to False for real trading
)
```

### Multi-Strategy Comparison

```python
from warspite_financial import multi_strategy_comparison

# Compare multiple strategies
comparison = multi_strategy_comparison(
    symbols=['BM-AAPL', 'BM-GOOGL'],
    days=60,
    initial_capital=10000
)

print(f"Best strategy: {comparison['best_strategy']}")
for name, result in comparison['results'].items():
    print(f"{name}: {result['total_return']:.2%}")
```

## Development

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run specific test categories
python3 -m pytest -m unit
python3 -m pytest -m property
python3 -m pytest -m integration

# Run with coverage
python3 -m pytest --cov=warspite_financial
```

### Test Categories

- **Unit Tests**: Test individual components and functions
- **Property Tests**: Property-based testing with Hypothesis for comprehensive input coverage
- **Integration Tests**: End-to-end workflow testing

## API Reference

### Core Classes

- `WarspiteDataset`: Time-series financial data container
- `WarspiteTradingEmulator`: Trading simulation engine
- `WarspiteCLI`: Command-line interface for interactive trading and visualization
- `BaseProvider`: Abstract base class for data providers
- `BaseStrategy`: Abstract base class for trading strategies

### Convenience Functions

- `create_dataset_from_provider()`: Create datasets from providers
- `run_strategy_backtest()`: Run complete strategy backtests
- `create_visualization()`: Generate charts and visualizations

### Example Workflows

- `basic_backtest_example()`: Complete backtesting workflow
- `multi_strategy_comparison()`: Compare multiple strategies
- `visualization_example()`: Demonstrate visualization capabilities
- `live_trading_example()`: Live trading integration example

## Error Handling

The library provides comprehensive error handling with custom exception types:

- `WarspiteError`: Base exception for all library errors
- `ProviderError`: Data provider related errors
- `DatasetError`: Dataset operation errors
- `StrategyError`: Strategy execution errors
- `EmulatorError`: Trading emulation errors
- `TradingError`: Live trading errors
- `VisualizationError`: Chart generation errors

## Performance Considerations

- Core data operations use numpy arrays for efficiency
- Datasets support time-based slicing for large data analysis
- Property-based tests run with 100+ iterations for thorough validation
- Visualization renderers support multiple output formats

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Changelog

### Version 0.1.0

- Initial release with core functionality
- Provider system with BrownianMotion, YFinance, and OANDA support
- Strategy framework with SMA, Random, and Perfect strategies
- Trading emulation with cost modeling
- Command-line interface with interactive trading and ASCII visualization
- Visualization system with multiple renderers
- Comprehensive test suite with property-based testing
- End-to-end workflow examples

## Support

- Documentation: [Read the Docs](https://warspite-financial.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/warspite-financial/warspite-financial/issues)
- Repository: [GitHub](https://github.com/warspite-financial/warspite-financial)