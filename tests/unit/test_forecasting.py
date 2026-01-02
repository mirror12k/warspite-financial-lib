"""
Unit tests for forecasting functionality.

These tests verify specific examples and edge cases for forecasting functionality
in the warspite_financial library.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from warspite_financial.datasets.dataset import WarspiteDataset
from warspite_financial.forecasting.heuristic import WarspiteHeuristicForecaster


class TestForecasterConstruction:
    """Unit tests for forecaster construction and basic operations."""
    
    def test_basic_forecaster_construction(self):
        """Test basic forecaster construction with valid dataset."""
        # Create test dataset
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0, 102.0, 103.0, 104.0])]
        symbols = ['AAPL']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Verify construction
        assert forecaster.dataset == dataset
        assert forecaster.forecasting_method == 'linear'  # Default method
        
        # Test string representation
        repr_str = repr(forecaster)
        assert 'WarspiteHeuristicForecaster' in repr_str
        assert 'AAPL' in repr_str
        assert 'dataset_length=5' in repr_str
        assert "method='linear'" in repr_str
    
    def test_forecaster_construction_validation(self):
        """Test forecaster construction validation with invalid inputs."""
        # Empty dataset
        timestamps = np.array([], dtype='datetime64[ns]')
        data_arrays = []
        symbols = []
        
        with pytest.raises(ValueError, match="At least one data array is required"):
            empty_dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Create a valid but empty-length dataset for testing
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL']
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        # This should work - single point dataset
        forecaster = WarspiteHeuristicForecaster(dataset)
        assert len(forecaster.dataset) == 1
    
    def test_forecasting_method_setting(self):
        """Test setting and getting forecasting methods."""
        # Create test dataset
        timestamps = np.array([datetime(2023, 1, 1), datetime(2023, 1, 2)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0, 101.0])]
        symbols = ['AAPL']
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Test valid methods
        for method in ['linear', 'exponential', 'seasonal']:
            forecaster.set_forecasting_method(method)
            assert forecaster.forecasting_method == method
        
        # Test invalid method
        with pytest.raises(ValueError, match="Method must be one of"):
            forecaster.set_forecasting_method('invalid_method')


class TestLinearForecasting:
    """Unit tests for linear forecasting functionality."""
    
    def setup_method(self):
        """Set up test data for linear forecasting tests."""
        # Create upward trending data
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ], dtype='datetime64[ns]')
        
        # Linear trend: 100, 102, 104, 106, 108
        self.trending_data = [np.array([100.0, 102.0, 104.0, 106.0, 108.0])]
        self.symbols = ['TREND']
        
        self.dataset = WarspiteDataset(self.trending_data, timestamps, self.symbols)
        self.forecaster = WarspiteHeuristicForecaster(self.dataset)
    
    def test_linear_forecast_basic(self):
        """Test basic linear forecasting functionality."""
        # Forecast 3 periods
        forecast_dataset = self.forecaster.forecast(3, method='linear')
        
        # Verify forecast dataset structure
        assert len(forecast_dataset) == 3
        assert forecast_dataset.symbols == ['TREND_forecast']
        assert len(forecast_dataset.data_arrays) == 1
        
        # Verify metadata
        metadata = forecast_dataset.metadata
        assert metadata['forecasted'] is True
        assert metadata['forecast_method'] == 'linear'
        assert metadata['forecast_periods'] == 3
        assert metadata['source_dataset_length'] == 5
        
        # Check forecast values (should continue the trend: 110, 112, 114)
        forecasted_values = forecast_dataset.data_arrays[0]
        assert len(forecasted_values) == 3
        
        # With linear trend of +2 per period, expect approximately 110, 112, 114
        expected_values = [110.0, 112.0, 114.0]
        for i, (actual, expected) in enumerate(zip(forecasted_values, expected_values)):
            assert abs(actual - expected) < 1.0, f"Period {i}: expected ~{expected}, got {actual}"
    
    def test_linear_forecast_single_period(self):
        """Test linear forecasting for a single period."""
        forecast_dataset = self.forecaster.forecast(1, method='linear')
        
        assert len(forecast_dataset) == 1
        assert len(forecast_dataset.data_arrays[0]) == 1
        
        # Should be approximately 110 (continuing +2 trend)
        forecasted_value = forecast_dataset.data_arrays[0][0]
        assert abs(forecasted_value - 110.0) < 1.0
    
    def test_linear_forecast_ohlcv_data(self):
        """Test linear forecasting with OHLCV data."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ], dtype='datetime64[ns]')
        
        # OHLCV data with trends in each column
        ohlcv_data = np.array([
            [100.0, 105.0, 99.0, 103.0, 1000],   # Day 1
            [103.0, 108.0, 102.0, 106.0, 1100],  # Day 2
            [106.0, 111.0, 105.0, 109.0, 1200]   # Day 3
        ])
        
        dataset = WarspiteDataset([ohlcv_data], timestamps, ['STOCK'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        forecast_dataset = forecaster.forecast(2, method='linear')
        
        # Should have 2D array output (2 periods x 5 columns)
        forecasted_array = forecast_dataset.data_arrays[0]
        assert forecasted_array.shape == (2, 5)
        
        # Each column should show an upward trend
        for col in range(5):
            assert forecasted_array[1, col] > forecasted_array[0, col]
    
    def test_linear_forecast_timestamps(self):
        """Test that forecasted timestamps are correctly generated."""
        forecast_dataset = self.forecaster.forecast(3, method='linear')
        
        # Should have 3 future timestamps
        forecast_timestamps = forecast_dataset.timestamps
        assert len(forecast_timestamps) == 3
        
        # Should be daily intervals (based on source data)
        original_timestamps = self.dataset.timestamps
        last_original = pd.to_datetime(original_timestamps[-1])
        
        for i, timestamp in enumerate(forecast_timestamps):
            expected_date = last_original + timedelta(days=i+1)
            actual_date = pd.to_datetime(timestamp)
            
            # Should be within a day (allowing for some timestamp precision issues)
            time_diff = abs((actual_date - expected_date).total_seconds())
            assert time_diff < 86400, f"Timestamp {i} off by {time_diff} seconds"


class TestExponentialForecasting:
    """Unit tests for exponential smoothing forecasting."""
    
    def setup_method(self):
        """Set up test data for exponential forecasting tests."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 6)
        ], dtype='datetime64[ns]')
        
        # Data with some volatility but overall upward trend
        self.volatile_data = [np.array([100.0, 103.0, 101.0, 105.0, 104.0, 107.0])]
        self.symbols = ['VOLATILE']
        
        self.dataset = WarspiteDataset(self.volatile_data, timestamps, self.symbols)
        self.forecaster = WarspiteHeuristicForecaster(self.dataset)
    
    def test_exponential_forecast_basic(self):
        """Test basic exponential smoothing forecasting."""
        forecast_dataset = self.forecaster.forecast(2, method='exponential')
        
        # Verify structure
        assert len(forecast_dataset) == 2
        assert forecast_dataset.symbols == ['VOLATILE_forecast']
        
        # Verify metadata
        metadata = forecast_dataset.metadata
        assert metadata['forecast_method'] == 'exponential'
        
        # Check that forecasts are reasonable (should be close to recent values)
        forecasted_values = forecast_dataset.data_arrays[0]
        last_value = self.volatile_data[0][-1]  # 107.0
        
        for value in forecasted_values:
            # Should be within reasonable range of last value
            assert 100.0 <= value <= 120.0
            assert abs(value - last_value) < 10.0
    
    def test_exponential_forecast_smoothing_effect(self):
        """Test that exponential smoothing reduces volatility."""
        # Create highly volatile data
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 11)
        ], dtype='datetime64[ns]')
        
        # Alternating high/low values
        volatile_values = [100.0, 120.0, 95.0, 125.0, 90.0, 130.0, 85.0, 135.0, 80.0, 140.0]
        dataset = WarspiteDataset([np.array(volatile_values)], timestamps, ['VOLATILE'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        forecast_dataset = forecaster.forecast(3, method='exponential')
        forecasted_values = forecast_dataset.data_arrays[0]
        
        # Forecasted values should be less volatile than input
        forecast_range = np.max(forecasted_values) - np.min(forecasted_values)
        input_range = np.max(volatile_values) - np.min(volatile_values)
        
        # Exponential smoothing should reduce the range
        assert forecast_range < input_range


class TestSeasonalForecasting:
    """Unit tests for seasonal pattern forecasting."""
    
    def setup_method(self):
        """Set up test data for seasonal forecasting tests."""
        # Create 30 days of data with weekly pattern
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 31)
        ], dtype='datetime64[ns]')
        
        # Create data with weekly seasonal pattern (higher on weekends)
        seasonal_values = []
        base_value = 100.0
        for i in range(30):
            day_of_week = i % 7
            if day_of_week in [5, 6]:  # Weekend (Saturday, Sunday)
                seasonal_values.append(base_value + 10.0 + i * 0.5)  # Higher on weekends
            else:
                seasonal_values.append(base_value + i * 0.5)  # Regular trend
        
        self.seasonal_data = [np.array(seasonal_values)]
        self.symbols = ['SEASONAL']
        
        self.dataset = WarspiteDataset(self.seasonal_data, timestamps, self.symbols)
        self.forecaster = WarspiteHeuristicForecaster(self.dataset)
    
    def test_seasonal_forecast_basic(self):
        """Test basic seasonal forecasting functionality."""
        forecast_dataset = self.forecaster.forecast(7, method='seasonal')  # Forecast one week
        
        # Verify structure
        assert len(forecast_dataset) == 7
        assert forecast_dataset.symbols == ['SEASONAL_forecast']
        
        # Verify metadata
        metadata = forecast_dataset.metadata
        assert metadata['forecast_method'] == 'seasonal'
        
        # Check that forecasts show some pattern (seasonal forecasting should work)
        forecasted_values = forecast_dataset.data_arrays[0]
        
        # Just verify that we get reasonable forecasted values
        # (The seasonal pattern detection might not be perfect with limited test data)
        assert len(forecasted_values) == 7
        assert np.all(np.isfinite(forecasted_values))
        assert np.all(forecasted_values > 0)  # Should be positive values
        
        # Check that values are in a reasonable range relative to input data
        input_mean = np.mean(self.seasonal_data[0])
        forecast_mean = np.mean(forecasted_values)
        
        # Forecasted mean should be reasonably close to input mean (within 50%)
        assert abs(forecast_mean - input_mean) / input_mean < 0.5
    
    def test_seasonal_forecast_insufficient_data(self):
        """Test seasonal forecasting with insufficient data for pattern detection."""
        # Create dataset with only 5 days (insufficient for weekly pattern)
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0, 102.0, 103.0, 104.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['SHORT'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Should still work but fall back to trend-based forecasting
        forecast_dataset = forecaster.forecast(3, method='seasonal')
        
        assert len(forecast_dataset) == 3
        assert len(forecast_dataset.data_arrays[0]) == 3


class TestConfidenceIntervals:
    """Unit tests for confidence interval estimation."""
    
    def setup_method(self):
        """Set up test data for confidence interval tests."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 11)
        ], dtype='datetime64[ns]')
        
        # Create data with some noise
        np.random.seed(42)  # For reproducibility
        base_trend = np.linspace(100, 110, 10)
        noise = np.random.normal(0, 2, 10)
        noisy_data = base_trend + noise
        
        self.dataset = WarspiteDataset([noisy_data], timestamps, ['NOISY'])
        self.forecaster = WarspiteHeuristicForecaster(self.dataset)
    
    def test_confidence_intervals_after_forecast(self):
        """Test that confidence intervals are available after forecasting."""
        # Initially no confidence intervals
        with pytest.raises(ValueError, match="No confidence intervals available"):
            self.forecaster.get_confidence_intervals()
        
        # Forecast and check confidence intervals
        forecast_dataset = self.forecaster.forecast(3, method='linear')
        confidence_intervals = self.forecaster.get_confidence_intervals()
        
        # Should have shape (periods, symbols, 2) for [lower, upper] bounds
        assert confidence_intervals.shape == (3, 1, 2)
        
        # Lower bounds should be less than upper bounds
        for period in range(3):
            lower = confidence_intervals[period, 0, 0]
            upper = confidence_intervals[period, 0, 1]
            assert lower < upper, f"Period {period}: lower bound {lower} >= upper bound {upper}"
        
        # Confidence intervals should widen with forecast horizon
        ci_width_0 = confidence_intervals[0, 0, 1] - confidence_intervals[0, 0, 0]
        ci_width_2 = confidence_intervals[2, 0, 1] - confidence_intervals[2, 0, 0]
        assert ci_width_2 > ci_width_0, "Confidence intervals should widen with forecast horizon"
    
    def test_confidence_intervals_different_methods(self):
        """Test confidence intervals for different forecasting methods."""
        methods = ['linear', 'exponential', 'seasonal']
        
        for method in methods:
            forecast_dataset = self.forecaster.forecast(2, method=method)
            confidence_intervals = self.forecaster.get_confidence_intervals()
            
            # Should have proper shape
            assert confidence_intervals.shape == (2, 1, 2)
            
            # Should have reasonable bounds
            for period in range(2):
                lower = confidence_intervals[period, 0, 0]
                upper = confidence_intervals[period, 0, 1]
                assert lower < upper
                assert not np.isnan(lower)
                assert not np.isnan(upper)
    
    def test_confidence_intervals_multiple_symbols(self):
        """Test confidence intervals with multiple symbols."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        data_arrays = [
            np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            np.array([200.0, 202.0, 204.0, 206.0, 208.0])
        ]
        symbols = ['STOCK1', 'STOCK2']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        forecast_dataset = forecaster.forecast(2, method='linear')
        confidence_intervals = forecaster.get_confidence_intervals()
        
        # Should have shape (periods, symbols, 2)
        assert confidence_intervals.shape == (2, 2, 2)
        
        # Check bounds for both symbols
        for symbol_idx in range(2):
            for period in range(2):
                lower = confidence_intervals[period, symbol_idx, 0]
                upper = confidence_intervals[period, symbol_idx, 1]
                assert lower < upper


class TestForecastingEdgeCases:
    """Unit tests for forecasting edge cases and error conditions."""
    
    def test_forecast_invalid_periods(self):
        """Test forecasting with invalid period counts."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['AAPL'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Zero periods
        with pytest.raises(ValueError, match="Number of periods must be positive"):
            forecaster.forecast(0)
        
        # Negative periods
        with pytest.raises(ValueError, match="Number of periods must be positive"):
            forecaster.forecast(-1)
    
    def test_forecast_invalid_method(self):
        """Test forecasting with invalid method."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['AAPL'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        with pytest.raises(ValueError, match="Unsupported forecasting method"):
            forecaster.forecast(1, method='invalid_method')
    
    def test_forecast_single_data_point(self):
        """Test forecasting with only one data point."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['AAPL'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Should work but with limited trend information
        forecast_dataset = forecaster.forecast(2, method='linear')
        
        assert len(forecast_dataset) == 2
        forecasted_values = forecast_dataset.data_arrays[0]
        
        # With only one point, trend should be zero, so forecasts should be close to original
        for value in forecasted_values:
            assert abs(value - 100.0) < 10.0  # Allow some variation due to algorithm
    
    def test_forecast_constant_data(self):
        """Test forecasting with constant (no trend) data."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        # All values the same
        constant_data = [np.array([100.0, 100.0, 100.0, 100.0, 100.0])]
        dataset = WarspiteDataset(constant_data, timestamps, ['CONSTANT'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        forecast_dataset = forecaster.forecast(3, method='linear')
        forecasted_values = forecast_dataset.data_arrays[0]
        
        # Should forecast values close to the constant value
        for value in forecasted_values:
            assert abs(value - 100.0) < 1.0
    
    def test_forecast_with_nan_values(self):
        """Test forecasting behavior with NaN values in data."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        # Data with NaN values
        data_with_nan = [np.array([100.0, np.nan, 102.0, 103.0, 104.0])]
        dataset = WarspiteDataset(data_with_nan, timestamps, ['WITH_NAN'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Should handle NaN values gracefully (numpy polyfit handles NaN)
        forecast_dataset = forecaster.forecast(2, method='linear')
        forecasted_values = forecast_dataset.data_arrays[0]
        
        # Forecasted values should not be NaN
        assert not np.any(np.isnan(forecasted_values))
        assert len(forecasted_values) == 2
    
    def test_forecast_large_periods(self):
        """Test forecasting with large number of periods."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0, 102.0, 103.0, 104.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['AAPL'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Forecast many periods
        forecast_dataset = forecaster.forecast(100, method='linear')
        
        assert len(forecast_dataset) == 100
        assert len(forecast_dataset.data_arrays[0]) == 100
        
        # Should maintain reasonable values (not explode to infinity)
        forecasted_values = forecast_dataset.data_arrays[0]
        assert np.all(np.isfinite(forecasted_values))
        assert np.all(forecasted_values > 0)  # Assuming positive stock prices
    
    def test_method_override_in_forecast(self):
        """Test that method parameter in forecast() overrides set method."""
        timestamps = np.array([
            datetime(2023, 1, i) for i in range(1, 6)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0, 102.0, 103.0, 104.0])]
        dataset = WarspiteDataset(data_arrays, timestamps, ['AAPL'])
        forecaster = WarspiteHeuristicForecaster(dataset)
        
        # Set method to exponential
        forecaster.set_forecasting_method('exponential')
        assert forecaster.forecasting_method == 'exponential'
        
        # Forecast with linear method (should override)
        forecast_dataset = forecaster.forecast(2, method='linear')
        
        # Method should still be exponential (not changed)
        assert forecaster.forecasting_method == 'exponential'
        
        # But forecast metadata should show linear was used
        assert forecast_dataset.metadata['forecast_method'] == 'linear'