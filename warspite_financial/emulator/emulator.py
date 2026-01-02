"""
Trading emulation for warspite_financial library.

This module contains the WarspiteTradingEmulator class for simulating trading scenarios.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a completed trade."""
    timestamp: datetime
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    portfolio_value: float


@dataclass
class EmulationStep:
    """Represents the result of a single emulation step."""
    timestamp: datetime
    portfolio_value: float
    positions: Dict[str, float]
    trades: List[Trade]
    cash: float


@dataclass
class EmulationResult:
    """Represents the complete result of an emulation run."""
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    total_trades: int
    trade_history: List[Trade]
    portfolio_history: List[float]
    timestamps: List[datetime]
    final_positions: Dict[str, float]
    
    @property
    def trades(self) -> List[Trade]:
        """Get trades (backward compatibility alias for trade_history)."""
        return self.trade_history


class WarspiteTradingEmulator:
    """
    Trading emulator for simulating trading strategies against historical data.
    
    This class provides comprehensive trading simulation capabilities including
    portfolio tracking, trade execution with costs, and performance analysis.
    """
    
    def __init__(self, dataset, initial_capital: float = 10000,
                 trading_fee: float = 0.0, spread: float = 0.0001,
                 strategies: Optional[List] = None):
        """
        Initialize trading emulator.
        
        Args:
            dataset: WarspiteDataset for emulation
            initial_capital: Starting capital amount
            trading_fee: Fee per trade (absolute amount)
            spread: Bid-ask spread (as fraction of price)
            strategies: Optional list of BaseStrategy instances
            
        Raises:
            ValueError: If parameters are invalid
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if trading_fee < 0:
            raise ValueError("Trading fee cannot be negative")
        if spread < 0 or spread >= 1:
            raise ValueError("Spread must be between 0 and 1")
        
        self._dataset = dataset
        self._initial_capital = initial_capital
        self._trading_fee = trading_fee
        self._spread = spread
        self._strategies = strategies or []
        
        # Trading provider for live trading
        self._trading_provider = None
        
        # Initialize emulation state
        self._reset_state()
    
    def _reset_state(self):
        """Reset emulator to initial state."""
        self._current_step = 0
        self._cash = self._initial_capital
        self._positions = {symbol: 0.0 for symbol in self._dataset.symbols}
        self._trade_history = []
        self._portfolio_history = [self._initial_capital]
        self._timestamp_history = []
        
        # Add first timestamp if dataset has data
        if len(self._dataset) > 0:
            self._timestamp_history.append(pd.to_datetime(self._dataset.timestamps[0]).to_pydatetime())
    
    @property
    def current_step(self) -> int:
        """Get current emulation step."""
        return self._current_step
    
    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash
    
    @property
    def positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self._positions.copy()
    
    @property
    def trade_history(self) -> List[Trade]:
        """Get complete trade history."""
        return self._trade_history.copy()
    
    @property
    def portfolio_history(self) -> List[float]:
        """Get portfolio value history."""
        return self._portfolio_history.copy()
    
    def add_strategy(self, strategy) -> None:
        """
        Add a trading strategy to the emulator.
        
        Args:
            strategy: BaseStrategy instance
        """
        self._strategies.append(strategy)
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all symbols at current step.
        
        Returns:
            Dictionary mapping symbols to current prices
        """
        if self._current_step >= len(self._dataset):
            raise ValueError("Current step exceeds dataset length")
        
        prices = {}
        for i, symbol in enumerate(self._dataset.symbols):
            data_array = self._dataset.data_arrays[i]
            if data_array.ndim == 1:
                # Single price series
                prices[symbol] = float(data_array[self._current_step])
            else:
                # Multi-column data, use Close prices (column 3)
                if data_array.shape[1] > 3:
                    prices[symbol] = float(data_array[self._current_step, 3])
                else:
                    prices[symbol] = float(data_array[self._current_step, -1])
        
        return prices
    
    def get_portfolio_value(self) -> float:
        """
        Calculate current portfolio value (cash + positions).
        
        Returns:
            Total portfolio value at current prices
        """
        if self._current_step >= len(self._dataset):
            return self._cash  # No position values if beyond dataset
        
        current_prices = self.get_current_prices()
        position_value = sum(
            self._positions[symbol] * current_prices[symbol]
            for symbol in self._positions
        )
        
        return self._cash + position_value
    
    def buy(self, symbol: str, quantity: float) -> bool:
        """
        Execute a buy order for the specified symbol and quantity.
        
        Args:
            symbol: Symbol to buy
            quantity: Quantity to buy (positive number)
            
        Returns:
            True if trade was executed, False otherwise
        """
        if quantity <= 0:
            return False
        
        if symbol not in self._positions:
            return False
        
        if self._current_step >= len(self._dataset):
            return False
        
        current_prices = self.get_current_prices()
        price = current_prices[symbol]
        
        # Apply spread (buy at ask price)
        ask_price = price * (1 + self._spread / 2)
        
        # Calculate total cost including fees
        total_cost = quantity * ask_price + self._trading_fee
        
        # Check if we have enough cash
        if total_cost > self._cash:
            return False
        
        # Execute trade
        self._cash -= total_cost
        self._positions[symbol] += quantity
        
        # Record trade
        current_timestamp = pd.to_datetime(self._dataset.timestamps[self._current_step]).to_pydatetime()
        trade = Trade(
            timestamp=current_timestamp,
            symbol=symbol,
            action='buy',
            quantity=quantity,
            price=ask_price,
            fee=self._trading_fee,
            portfolio_value=self.get_portfolio_value()
        )
        self._trade_history.append(trade)
        
        return True
    
    def sell(self, symbol: str, quantity: float) -> bool:
        """
        Execute a sell order for the specified symbol and quantity.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell (positive number)
            
        Returns:
            True if trade was executed, False otherwise
        """
        if quantity <= 0:
            return False
        
        if symbol not in self._positions:
            return False
        
        if self._current_step >= len(self._dataset):
            return False
        
        # Check if we have enough position to sell
        if quantity > self._positions[symbol]:
            return False
        
        current_prices = self.get_current_prices()
        price = current_prices[symbol]
        
        # Apply spread (sell at bid price)
        bid_price = price * (1 - self._spread / 2)
        
        # Calculate proceeds minus fees
        proceeds = quantity * bid_price - self._trading_fee
        
        # Execute trade
        self._cash += proceeds
        self._positions[symbol] -= quantity
        
        # Record trade
        current_timestamp = pd.to_datetime(self._dataset.timestamps[self._current_step]).to_pydatetime()
        trade = Trade(
            timestamp=current_timestamp,
            symbol=symbol,
            action='sell',
            quantity=quantity,
            price=bid_price,
            fee=self._trading_fee,
            portfolio_value=self.get_portfolio_value()
        )
        self._trade_history.append(trade)
        
        return True
    
    def close(self, symbol: str) -> bool:
        """
        Close all positions in the specified symbol.
        
        Args:
            symbol: Symbol to close
            
        Returns:
            True if position was closed, False otherwise
        """
        if symbol not in self._positions:
            return False
        
        position = self._positions[symbol]
        if position == 0:
            return True  # Already closed
        
        if position > 0:
            return self.sell(symbol, position)
        else:
            # Short position - buy to cover
            return self.buy(symbol, abs(position))
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            True if all positions were closed successfully
        """
        success = True
        for symbol in list(self._positions.keys()):
            if self._positions[symbol] != 0:
                if not self.close(symbol):
                    success = False
        return success
    
    def apply_positions(self, positions: Dict[str, float]) -> bool:
        """
        Apply target positions by executing necessary trades.
        
        Args:
            positions: Dictionary mapping symbols to target position sizes
            
        Returns:
            True if all positions were successfully applied
        """
        success = True
        
        for symbol, target_position in positions.items():
            if symbol not in self._positions:
                continue
            
            current_position = self._positions[symbol]
            position_diff = target_position - current_position
            
            if abs(position_diff) < 1e-8:  # Essentially zero
                continue
            
            if position_diff > 0:
                # Need to buy more
                if not self.buy(symbol, position_diff):
                    success = False
            else:
                # Need to sell
                if not self.sell(symbol, abs(position_diff)):
                    success = False
        
        return success
    
    def step_forward(self) -> EmulationStep:
        """
        Execute one step of the emulation.
        
        Returns:
            EmulationStep containing the results of this step
            
        Raises:
            ValueError: If emulation has reached the end of the dataset
        """
        if self._current_step >= len(self._dataset):
            raise ValueError("Emulation has reached the end of the dataset")
        
        # Get current timestamp
        current_timestamp = pd.to_datetime(self._dataset.timestamps[self._current_step]).to_pydatetime()
        
        # Store trades executed in this step
        trades_before = len(self._trade_history)
        
        # Apply strategies if any are configured
        if self._strategies:
            # Generate positions from all strategies
            for strategy in self._strategies:
                try:
                    # Get strategy positions for current step
                    strategy_positions = strategy.generate_positions(self._dataset)
                    
                    # Extract positions for current timestamp
                    if strategy_positions.ndim == 1:
                        # Legacy single-symbol format
                        if len(self._dataset.symbols) == 1:
                            target_positions = {
                                self._dataset.symbols[0]: strategy_positions[self._current_step]
                            }
                        else:
                            # Skip if format doesn't match
                            continue
                    else:
                        # Multi-symbol format
                        target_positions = {}
                        for i, symbol in enumerate(self._dataset.symbols):
                            target_positions[symbol] = strategy_positions[self._current_step, i]
                    
                    # Apply the positions
                    self.apply_positions(target_positions)
                    
                except Exception as e:
                    # Log strategy error but continue
                    print(f"Warning: Strategy {type(strategy).__name__} failed at step {self._current_step}: {e}")
        
        # Calculate portfolio value and update history
        portfolio_value = self.get_portfolio_value()
        self._portfolio_history.append(portfolio_value)
        self._timestamp_history.append(current_timestamp)
        
        # Get trades executed in this step
        step_trades = self._trade_history[trades_before:]
        
        # Create step result
        step_result = EmulationStep(
            timestamp=current_timestamp,
            portfolio_value=portfolio_value,
            positions=self._positions.copy(),
            trades=step_trades,
            cash=self._cash
        )
        
        # Move to next step
        self._current_step += 1
        
        return step_result
    
    def run_to_completion(self) -> EmulationResult:
        """
        Run the emulation to completion.
        
        Returns:
            EmulationResult containing complete simulation results
        """
        # Reset to beginning if needed
        if self._current_step > 0:
            self._reset_state()
        
        # Run through all steps
        while self._current_step < len(self._dataset):
            self.step_forward()
        
        # Calculate final metrics
        final_portfolio_value = self._portfolio_history[-1] if self._portfolio_history else self._initial_capital
        total_return = (final_portfolio_value - self._initial_capital) / self._initial_capital
        
        return EmulationResult(
            initial_capital=self._initial_capital,
            final_portfolio_value=final_portfolio_value,
            total_return=total_return,
            total_trades=len(self._trade_history),
            trade_history=self._trade_history.copy(),
            portfolio_history=self._portfolio_history.copy(),
            timestamps=self._timestamp_history.copy(),
            final_positions=self._positions.copy()
        )
    
    def connect_trading_provider(self, provider) -> None:
        """
        Connect a trading provider for live trading.
        
        Args:
            provider: TradingProvider instance for live trading
            
        Raises:
            ValueError: If provider is invalid
        """
        # Import here to avoid circular imports
        from ..providers.base import TradingProvider
        
        if not isinstance(provider, TradingProvider):
            raise ValueError("Provider must be a TradingProvider instance")
        
        # Validate provider connectivity and permissions
        try:
            # Test connection by getting account info
            account_info = provider.get_account_info()
            if not account_info or not hasattr(account_info, 'account_id'):
                raise ValueError("Provider failed to return valid account information")
            
            # Test permissions by getting positions (should not fail)
            positions = provider.get_positions()
            if positions is None:
                raise ValueError("Provider failed to return position information")
            
        except Exception as e:
            raise ValueError(f"Provider validation failed: {e}")
        
        self._trading_provider = provider
        
        # Log connection for safety
        print(f"Trading provider connected: {type(provider).__name__}")
        print(f"Account ID: {account_info.account_id}")
        print(f"Current balance: {account_info.balance} {account_info.currency}")
        print("WARNING: Live trading is now enabled. All trades will use real money.")
    
    def apply_live_trade(self) -> bool:
        """
        Apply current positions to the connected trading provider.
        
        Returns:
            True if live trades were successfully executed
            
        Raises:
            ValueError: If no trading provider is connected
            ConnectionError: If trading provider is unreachable
            PermissionError: If trading permissions are insufficient
        """
        if self._trading_provider is None:
            raise ValueError("No trading provider connected")
        
        # Safety check: Ensure we're not in a dangerous state
        total_position_value = sum(
            abs(position) * self.get_current_prices().get(symbol, 0)
            for symbol, position in self._positions.items()
        )
        
        # Safety limit: Don't allow positions worth more than 10x initial capital
        if total_position_value > self._initial_capital * 10:
            raise PermissionError(
                f"Position value ({total_position_value:.2f}) exceeds safety limit "
                f"({self._initial_capital * 10:.2f}). Trade rejected for safety."
            )
        
        try:
            # Get current positions from provider
            current_positions = self._trading_provider.get_positions()
            current_position_dict = {pos.symbol: pos.quantity for pos in current_positions}
            
            # Get account info to check available balance
            account_info = self._trading_provider.get_account_info()
            
            # Calculate position differences and required capital
            trades_to_execute = []
            total_required_capital = 0.0
            
            for symbol, target_quantity in self._positions.items():
                current_quantity = current_position_dict.get(symbol, 0.0)
                quantity_diff = target_quantity - current_quantity
                
                if abs(quantity_diff) < 1e-8:  # Essentially zero
                    continue
                
                # Estimate required capital for this trade
                if quantity_diff > 0:  # Buying
                    current_price = self.get_current_prices().get(symbol, 0)
                    if current_price > 0:
                        required_capital = quantity_diff * current_price
                        total_required_capital += required_capital
                
                trades_to_execute.append((symbol, quantity_diff))
            
            # Safety check: Ensure we have sufficient balance
            if total_required_capital > account_info.balance * 0.9:  # Use max 90% of balance
                raise PermissionError(
                    f"Required capital ({total_required_capital:.2f}) exceeds available balance "
                    f"({account_info.balance:.2f}). Trade rejected for safety."
                )
            
            # Execute trades with confirmation
            if trades_to_execute:
                print(f"WARNING: About to execute {len(trades_to_execute)} live trades:")
                for symbol, quantity in trades_to_execute:
                    action = "BUY" if quantity > 0 else "SELL"
                    print(f"  {action} {abs(quantity):.4f} of {symbol}")
                
                print(f"Total estimated capital required: {total_required_capital:.2f}")
                print("This will use REAL MONEY. Proceeding with trade execution...")
            
            # Execute the trades
            success = True
            executed_trades = []
            
            for symbol, quantity_diff in trades_to_execute:
                try:
                    # Place order for the difference
                    order_result = self._trading_provider.place_order(
                        symbol=symbol,
                        quantity=quantity_diff,
                        order_type='market'
                    )
                    
                    if order_result.success:
                        executed_trades.append((symbol, quantity_diff, order_result.order_id))
                        print(f"✓ Executed: {quantity_diff:+.4f} {symbol} (Order ID: {order_result.order_id})")
                    else:
                        success = False
                        print(f"✗ Failed: {symbol} - {order_result.message}")
                        
                except Exception as e:
                    success = False
                    print(f"✗ Error executing trade for {symbol}: {e}")
            
            if executed_trades:
                print(f"Successfully executed {len(executed_trades)} out of {len(trades_to_execute)} trades")
            
            return success
            
        except PermissionError:
            # Re-raise permission errors as-is
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to execute live trades: {e}")
    
    def validate_live_trading_safety(self) -> Dict[str, Any]:
        """
        Validate current state for live trading safety.
        
        Returns:
            Dictionary containing safety validation results
        """
        safety_report = {
            'is_safe': True,
            'warnings': [],
            'errors': [],
            'position_value': 0.0,
            'cash_ratio': 0.0,
            'max_position_size': 0.0
        }
        
        try:
            # Calculate position metrics
            current_prices = self.get_current_prices()
            total_position_value = 0.0
            max_position_value = 0.0
            
            for symbol, position in self._positions.items():
                if symbol in current_prices:
                    position_value = abs(position) * current_prices[symbol]
                    total_position_value += position_value
                    max_position_value = max(max_position_value, position_value)
            
            safety_report['position_value'] = total_position_value
            safety_report['cash_ratio'] = self._cash / self._initial_capital
            safety_report['max_position_size'] = max_position_value / self._initial_capital
            
            # Safety checks
            
            # 1. Position size limits
            if total_position_value > self._initial_capital * 5:
                safety_report['errors'].append(
                    f"Total position value ({total_position_value:.2f}) exceeds 5x initial capital"
                )
                safety_report['is_safe'] = False
            elif total_position_value > self._initial_capital * 2:
                safety_report['warnings'].append(
                    f"Total position value ({total_position_value:.2f}) exceeds 2x initial capital"
                )
            
            # 2. Cash ratio checks
            if self._cash < 0:
                safety_report['errors'].append(f"Negative cash balance: {self._cash:.2f}")
                safety_report['is_safe'] = False
            elif self._cash < self._initial_capital * 0.1:
                safety_report['warnings'].append(
                    f"Low cash reserves: {self._cash:.2f} ({self._cash/self._initial_capital*100:.1f}% of initial capital)"
                )
            
            # 3. Individual position size checks
            if max_position_value > self._initial_capital * 2:
                safety_report['warnings'].append(
                    f"Large individual position: {max_position_value:.2f} ({max_position_value/self._initial_capital*100:.1f}% of initial capital)"
                )
            
            # 4. Trading provider connection check
            if self._trading_provider is None:
                safety_report['warnings'].append("No trading provider connected")
            
        except Exception as e:
            safety_report['errors'].append(f"Safety validation failed: {e}")
            safety_report['is_safe'] = False
        
        return safety_report
    
    def enable_live_trading_mode(self, confirmation_code: str = None) -> bool:
        """
        Enable live trading mode with additional safety confirmation.
        
        Args:
            confirmation_code: Required confirmation code for live trading
            
        Returns:
            True if live trading mode was enabled
            
        Raises:
            PermissionError: If confirmation is invalid or safety checks fail
        """
        # Require explicit confirmation code
        expected_code = "ENABLE_LIVE_TRADING_WITH_REAL_MONEY"
        if confirmation_code != expected_code:
            raise PermissionError(
                f"Invalid confirmation code. Required: '{expected_code}'"
            )
        
        # Run safety validation
        safety_report = self.validate_live_trading_safety()
        
        if not safety_report['is_safe']:
            error_msg = "Live trading safety validation failed:\n"
            for error in safety_report['errors']:
                error_msg += f"  - {error}\n"
            raise PermissionError(error_msg.strip())
        
        # Show warnings if any
        if safety_report['warnings']:
            print("Live trading safety warnings:")
            for warning in safety_report['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Final confirmation
        print("🚨 LIVE TRADING MODE ENABLED 🚨")
        print("All subsequent trades will use REAL MONEY")
        print("Position value:", f"{safety_report['position_value']:.2f}")
        print("Cash ratio:", f"{safety_report['cash_ratio']*100:.1f}%")
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the emulation.
        
        Returns:
            Dictionary containing various performance metrics
        """
        if len(self._portfolio_history) < 2:
            return {}
        
        # Convert to numpy array for calculations
        portfolio_values = np.array(self._portfolio_history)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Volatility (annualized, assuming daily data)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for trade in self._trade_history 
                           if trade.action == 'sell' and 
                           any(t.symbol == trade.symbol and t.action == 'buy' and t.timestamp < trade.timestamp 
                               for t in self._trade_history))
        total_trades = len([t for t in self._trade_history if t.action == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self._trade_history),
            'final_portfolio_value': portfolio_values[-1]
        }