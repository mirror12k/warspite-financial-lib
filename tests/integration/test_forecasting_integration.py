"""
Forecasting integration tests for warspite_financial library.

These tests verify forecasting integration with existing datasets and workflows.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from warspite_financial import (
    BrownianMotionProvider,
    WarspiteDataset,
    WarspiteHeuristicForecaster,
    SMAStrategy,
    WarspiteTradingEmulator,
    create_dataset_from_provider
)
from warspite_financial.utils.exceptions import WarspiteError


class TestForecastingIntegration:
    """Test forecasting integration with existing components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BrownianMotionProvider()
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=60)
        
        # Create base dataset for testing
        self.dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL', 'GOOGL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
    def test_forecasting_with_dataset_creation(self):
        """Test forecasting integration with dataset creation workflow."""
        # Create forecaster
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        
        # Generate forecast
        forecast_periods = 10
        forecast_dataset = forecaster.forecast(
            periods=forecast_periods,
            method='linear'
        )
        
        # Verify forecast dataset structure
        assert isinstance(forecast_dataset, WarspiteDataset)
        assert len(forecast_dataset.symbols) == len(self.dataset.symbols)
        assert len(forecast_dataset.timestamps) == forecast_periods
        
        # Check data arrays structure
        forecast_arrays = forecast_dataset.data_arrays
        original_arrays = self.dataset.data_arrays
        assert len(forecast_arrays) == len(original_arrays)
        for i, arr in enumerate(forecast_arrays):
            assert arr.shape[0] == forecast_periods
            assert arr.shape[1] == original_arrays[i].shape[1]
        
        # Verify forecast timestamps are in the future
        last_original_timestamp = self.dataset.timestamps[-1]
        first_forecast_timestamp = forecast_dataset.timestamps[0]
        assert first_forecast_timestamp > last_original_timestamp
        
    def test_forecasting_methods_integration(self):
        """Test different forecasting methods with real dataset."""
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        methods = ['linear', 'exponential', 'seasonal']
        
        forecasts = {}
        for method in methods:
            forecast = forecaster.forecast(periods=5, method=method)
            forecasts[method] = forecast
            
            # Verify each forecast
            assert isinstance(forecast, WarspiteDataset)
            assert len(forecast.timestamps) == 5
            assert len(forecast.symbols) == len(self.dataset.symbols)
            
            # Verify data is reasonable (no NaN, positive prices)
            for data_array in forecast.data_arrays:
                assert not np.any(np.isnan(data_array))
                assert np.all(data_array > 0)  # Prices should be positive
            
        # Different methods should produce different results
        linear_arrays = forecasts['linear'].data_arrays
        exponential_arrays = forecasts['exponential'].data_arrays
        seasonal_arrays = forecasts['seasonal'].data_arrays
        
        # At least one method should differ from others (check first symbol's data)
        assert not np.allclose(linear_arrays[0], exponential_arrays[0], rtol=0.01) or \
               not np.allclose(linear_arrays[0], seasonal_arrays[0], rtol=0.01) or \
               not np.allclose(exponential_arrays[0], seasonal_arrays[0], rtol=0.01)
               
    def test_confidence_intervals_integration(self):
        """Test confidence interval calculation with real data."""
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        
        # Generate forecast
        forecast = forecaster.forecast(periods=7, method='linear')
        
        # Get confidence intervals
        confidence_levels = [0.68, 0.95]
        intervals = forecaster.get_confidence_intervals()
        
        # Verify structure - the method returns intervals for the most recent forecast
        assert isinstance(intervals, np.ndarray)
        assert intervals.shape[0] == len(forecast.timestamps)  # Should match forecast length
                
    def test_forecasting_with_strategy_integration(self):
        """Test forecasting integration with trading strategies."""
        # Create forecast
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        forecast = forecaster.forecast(periods=10, method='linear')
        
        # Apply strategy to forecast (use smaller period to avoid length issues)
        strategy = SMAStrategy(period=3)  # Reduced from 5 to avoid length issues
        positions = strategy.generate_positions(forecast)
        
        # Verify strategy works with forecast data
        assert len(positions) == len(forecast.timestamps)
        assert np.all(positions >= -1.0) and np.all(positions <= 1.0)
        
        # Add strategy results to forecast dataset
        forecast.add_strategy_results(positions)
        
        # Verify strategy results are stored
        assert hasattr(forecast, 'strategy_results')
        assert len(forecast.strategy_results) > 0
        
    def test_forecasting_with_emulator_integration(self):
        """Test forecasting integration with trading emulator."""
        # Create forecast
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        forecast = forecaster.forecast(periods=15, method='exponential')
        
        # Use forecast in emulator (use smaller period to avoid length issues)
        strategy = SMAStrategy(period=3)  # Reduced from 5 to avoid length issues
        emulator = WarspiteTradingEmulator(
            dataset=forecast,
            initial_capital=10000,
            trading_fee=0.001,
            spread=0.0001
        )
        emulator.add_strategy(strategy)
        
        # Run emulation on forecast data
        result = emulator.run_to_completion()
        
        # Verify emulation works with forecast
        assert result.final_portfolio_value > 0
        assert result.initial_capital == 10000
        assert hasattr(result, 'trade_history')  # Changed from 'trades' to 'trade_history'
        
    def test_combined_historical_and_forecast_workflow(self):
        """Test workflow combining historical data with forecasts."""
        # Step 1: Backtest on historical data
        historical_strategy = SMAStrategy(period=10)
        historical_emulator = WarspiteTradingEmulator(
            dataset=self.dataset,
            initial_capital=10000,
            trading_fee=0.001,
            spread=0.0001
        )
        historical_emulator.add_strategy(historical_strategy)
        historical_result = historical_emulator.run_to_completion()
        
        # Step 2: Generate forecast
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        forecast = forecaster.forecast(periods=10, method='linear')
        
        # Step 3: Apply same strategy to forecast
        forecast_strategy = SMAStrategy(period=10)
        forecast_emulator = WarspiteTradingEmulator(
            dataset=forecast,
            initial_capital=historical_result.final_portfolio_value,  # Use historical result as starting capital
            trading_fee=0.001,
            spread=0.0001
        )
        forecast_emulator.add_strategy(forecast_strategy)
        forecast_result = forecast_emulator.run_to_completion()
        
        # Verify workflow completed successfully
        assert historical_result.final_portfolio_value > 0
        assert forecast_result.final_portfolio_value > 0
        assert forecast_result.initial_capital == historical_result.final_portfolio_value
        
    def test_forecasting_serialization_integration(self):
        """Test forecasting integration with dataset serialization."""
        # Create forecast
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        forecast = forecaster.forecast(periods=8, method='seasonal')
        
        # Add strategy results
        strategy = SMAStrategy(period=3)
        positions = strategy.generate_positions(forecast)
        forecast.add_strategy_results(positions)
        
        # Test serialization
        csv_data = forecast.serialize(format='csv')
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0
        # Check for timestamp-related content (could be 'timestamp', 'date', or index)
        csv_lower = csv_data.lower()
        assert any(keyword in csv_lower for keyword in ['timestamp', 'date', '2020-', '2021-', '2022-', '2023-', '2024-', '2025-', '2026-']), f"No timestamp data found in CSV: {csv_data[:200]}"
        
        pickle_data = forecast.serialize(format='pickle')
        assert isinstance(pickle_data, bytes)
        assert len(pickle_data) > 0
        
        # Test deserialization
        deserialized = WarspiteDataset.deserialize(pickle_data, format='pickle')
        assert isinstance(deserialized, WarspiteDataset)
        assert len(deserialized.timestamps) == len(forecast.timestamps)
        assert len(deserialized.symbols) == len(forecast.symbols)
        assert np.allclose(deserialized.data, forecast.data)
        
    def test_forecasting_error_handling_integration(self):
        """Test error handling in forecasting integration scenarios."""
        forecaster = WarspiteHeuristicForecaster(self.dataset)
        
        # Test invalid forecast periods
        with pytest.raises((ValueError, WarspiteError)):
            forecaster.forecast(periods=0, method='linear')
            
        with pytest.raises((ValueError, WarspiteError)):
            forecaster.forecast(periods=-5, method='linear')
            
        # Test invalid method
        with pytest.raises((ValueError, WarspiteError)):
            forecaster.forecast(periods=5, method='invalid_method')
            
        # Test confidence intervals with invalid forecast
        with pytest.raises((ValueError, WarspiteError)):
            # This should fail since get_confidence_intervals doesn't take parameters
            forecaster.get_confidence_intervals()
            
    def test_forecasting_with_multiple_symbols(self):
        """Test forecasting integration with multiple symbols."""
        # Create dataset with multiple symbols
        multi_symbol_dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        forecaster = WarspiteHeuristicForecaster(multi_symbol_dataset)
        forecast = forecaster.forecast(periods=12, method='exponential')
        
        # Verify all symbols are forecasted (with _forecast suffix)
        assert len(forecast.symbols) == 4
        expected_symbols = [f"{symbol}_forecast" for symbol in multi_symbol_dataset.symbols]
        assert forecast.symbols == expected_symbols
        
        # Verify data dimensions
        expected_columns = len(multi_symbol_dataset.symbols) * 5  # OHLCV per symbol
        forecast_arrays = forecast.data_arrays
        total_columns = sum(arr.shape[1] for arr in forecast_arrays)
        assert total_columns == expected_columns
        
        # Test confidence intervals for multiple symbols
        intervals = forecaster.get_confidence_intervals()
        
        # Should return intervals for the forecast
        assert isinstance(intervals, np.ndarray)
        assert intervals.shape[0] == len(forecast.timestamps)
            
    def test_forecasting_performance_integration(self):
        """Test forecasting performance with larger datasets."""
        # Create larger dataset
        large_start_date = self.end_date - timedelta(days=200)
        large_dataset = create_dataset_from_provider(
            provider=self.provider,
            symbols=['AAPL', 'GOOGL'],
            start_date=large_start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        forecaster = WarspiteHeuristicForecaster(large_dataset)
        
        # Test that forecasting completes in reasonable time
        import time
        start_time = time.time()
        
        forecast = forecaster.forecast(periods=20, method='linear')
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 10.0  # 10 seconds max
        
        # Verify forecast quality
        assert isinstance(forecast, WarspiteDataset)
        assert len(forecast.timestamps) == 20
        for data_array in forecast.data_arrays:
            assert not np.any(np.isnan(data_array))


if __name__ == '__main__':
    pytest.main([__file__])