"""
Heuristic forecasting implementation for warspite_financial library.

This module contains the WarspiteHeuristicForecaster class for generating
forecasts based on historical data patterns using various heuristic methods.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from ..datasets.dataset import WarspiteDataset


class WarspiteHeuristicForecaster:
    """
    A heuristic forecaster for financial time-series data.
    
    This class provides various forecasting methods including linear extrapolation,
    exponential smoothing, and seasonal pattern recognition to generate future
    predictions based on historical data patterns.
    """
    
    def __init__(self, dataset: WarspiteDataset):
        """
        Initialize the forecaster with a WarspiteDataset.
        
        Args:
            dataset: WarspiteDataset containing historical data for forecasting
            
        Raises:
            ValueError: If dataset is empty or invalid
        """
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        self._dataset = dataset
        self._forecasting_method = 'linear'
        self._confidence_intervals: Optional[np.ndarray] = None
        
    @property
    def dataset(self) -> WarspiteDataset:
        """Get the underlying dataset."""
        return self._dataset
    
    @property
    def forecasting_method(self) -> str:
        """Get the current forecasting method."""
        return self._forecasting_method
    
    def set_forecasting_method(self, method: str) -> None:
        """
        Set the forecasting method to use.
        
        Args:
            method: Forecasting method ('linear', 'exponential', 'seasonal')
            
        Raises:
            ValueError: If method is not supported
        """
        supported_methods = ['linear', 'exponential', 'seasonal']
        if method not in supported_methods:
            raise ValueError(f"Method must be one of: {supported_methods}")
        
        self._forecasting_method = method
    
    def forecast(self, periods: int, method: Optional[str] = None) -> WarspiteDataset:
        """
        Generate forecasts for the specified number of periods.
        
        Args:
            periods: Number of future periods to forecast
            method: Forecasting method to use (overrides current method if provided)
            
        Returns:
            New WarspiteDataset containing forecasted data
            
        Raises:
            ValueError: If periods is invalid or forecasting fails
        """
        if periods <= 0:
            raise ValueError("Number of periods must be positive")
        
        # Use provided method or current method
        forecast_method = method or self._forecasting_method
        
        # Generate forecasts based on the selected method
        if forecast_method == 'linear':
            forecasted_arrays, forecasted_timestamps = self._linear_forecast(periods)
        elif forecast_method == 'exponential':
            forecasted_arrays, forecasted_timestamps = self._exponential_forecast(periods)
        elif forecast_method == 'seasonal':
            forecasted_arrays, forecasted_timestamps = self._seasonal_forecast(periods)
        else:
            raise ValueError(f"Unsupported forecasting method: {forecast_method}")
        
        # Create metadata for the forecasted dataset
        forecast_metadata = {
            'forecasted': True,
            'forecast_method': forecast_method,
            'forecast_periods': periods,
            'forecast_timestamp': datetime.now().isoformat(),
            'source_dataset_length': len(self._dataset),
            'source_dataset_symbols': self._dataset.symbols,
            'original_metadata': self._dataset.metadata
        }
        
        # Create new WarspiteDataset with forecasted data
        forecasted_dataset = WarspiteDataset(
            data_arrays=forecasted_arrays,
            timestamps=forecasted_timestamps,
            symbols=[f"{symbol}_forecast" for symbol in self._dataset.symbols],
            metadata=forecast_metadata
        )
        
        return forecasted_dataset
    
    def get_confidence_intervals(self) -> np.ndarray:
        """
        Get confidence intervals for the most recent forecast.
        
        Returns:
            Array of confidence intervals with shape (n_periods, n_symbols, 2)
            where the last dimension contains [lower_bound, upper_bound]
            
        Raises:
            ValueError: If no forecast has been generated yet
        """
        if self._confidence_intervals is None:
            raise ValueError("No confidence intervals available. Run forecast() first.")
        
        return self._confidence_intervals.copy()
    
    def _linear_forecast(self, periods: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate linear extrapolation forecasts.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (forecasted_arrays, forecasted_timestamps)
        """
        # Get the last few data points for trend calculation
        trend_window = min(30, len(self._dataset) // 2, len(self._dataset))  # Use last 30 points or half the data
        trend_window = max(2, trend_window)  # Need at least 2 points for linear fit
        
        forecasted_arrays = []
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                # Single column data (e.g., close prices)
                recent_data = data_array[-trend_window:]
                
                # Handle edge case of insufficient data
                if len(recent_data) < 2:
                    # If only one point, assume zero trend
                    forecasted_values = np.full(periods, recent_data[0])
                else:
                    # Calculate linear trend
                    x = np.arange(len(recent_data))
                    try:
                        coeffs = np.polyfit(x, recent_data, 1)  # Linear fit
                        
                        # Generate forecasts
                        future_x = np.arange(len(recent_data), len(recent_data) + periods)
                        forecasted_values = np.polyval(coeffs, future_x)
                    except np.linalg.LinAlgError:
                        # If linear fit fails, use last value with zero trend
                        forecasted_values = np.full(periods, recent_data[-1])
                
                forecasted_arrays.append(forecasted_values)
                
            else:
                # Multi-column data (OHLCV)
                n_cols = data_array.shape[1]
                forecasted_cols = []
                
                for col in range(n_cols):
                    recent_data = data_array[-trend_window:, col]
                    
                    # Handle edge case of insufficient data
                    if len(recent_data) < 2:
                        forecasted_values = np.full(periods, recent_data[0])
                    else:
                        # Calculate linear trend for this column
                        x = np.arange(len(recent_data))
                        try:
                            coeffs = np.polyfit(x, recent_data, 1)
                            
                            # Generate forecasts
                            future_x = np.arange(len(recent_data), len(recent_data) + periods)
                            forecasted_values = np.polyval(coeffs, future_x)
                        except np.linalg.LinAlgError:
                            # If linear fit fails, use last value with zero trend
                            forecasted_values = np.full(periods, recent_data[-1])
                    
                    forecasted_cols.append(forecasted_values)
                
                # Stack columns to create multi-column forecast
                forecasted_arrays.append(np.column_stack(forecasted_cols))
        
        # Generate future timestamps
        forecasted_timestamps = self._generate_future_timestamps(periods)
        
        # Calculate confidence intervals based on historical volatility
        self._calculate_confidence_intervals_linear(periods, trend_window)
        
        return forecasted_arrays, forecasted_timestamps
    
    def _exponential_forecast(self, periods: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate exponential smoothing forecasts.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (forecasted_arrays, forecasted_timestamps)
        """
        # Exponential smoothing parameters
        alpha = 0.3  # Smoothing parameter for level
        beta = 0.1   # Smoothing parameter for trend
        
        forecasted_arrays = []
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                # Single column data
                forecasted_values = self._exponential_smooth_series(data_array, periods, alpha, beta)
                forecasted_arrays.append(forecasted_values)
                
            else:
                # Multi-column data (OHLCV)
                n_cols = data_array.shape[1]
                forecasted_cols = []
                
                for col in range(n_cols):
                    series = data_array[:, col]
                    forecasted_values = self._exponential_smooth_series(series, periods, alpha, beta)
                    forecasted_cols.append(forecasted_values)
                
                # Stack columns to create multi-column forecast
                forecasted_arrays.append(np.column_stack(forecasted_cols))
        
        # Generate future timestamps
        forecasted_timestamps = self._generate_future_timestamps(periods)
        
        # Calculate confidence intervals based on exponential smoothing errors
        self._calculate_confidence_intervals_exponential(periods, alpha, beta)
        
        return forecasted_arrays, forecasted_timestamps
    
    def _seasonal_forecast(self, periods: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate seasonal pattern-based forecasts.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (forecasted_arrays, forecasted_timestamps)
        """
        # Detect seasonal patterns (assume daily data with weekly/monthly patterns)
        seasonal_periods = [7, 30]  # Weekly and monthly patterns
        
        forecasted_arrays = []
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                # Single column data
                forecasted_values = self._seasonal_forecast_series(data_array, periods, seasonal_periods)
                forecasted_arrays.append(forecasted_values)
                
            else:
                # Multi-column data (OHLCV)
                n_cols = data_array.shape[1]
                forecasted_cols = []
                
                for col in range(n_cols):
                    series = data_array[:, col]
                    forecasted_values = self._seasonal_forecast_series(series, periods, seasonal_periods)
                    forecasted_cols.append(forecasted_values)
                
                # Stack columns to create multi-column forecast
                forecasted_arrays.append(np.column_stack(forecasted_cols))
        
        # Generate future timestamps
        forecasted_timestamps = self._generate_future_timestamps(periods)
        
        # Calculate confidence intervals based on seasonal pattern variance
        self._calculate_confidence_intervals_seasonal(periods, seasonal_periods)
        
        return forecasted_arrays, forecasted_timestamps
    
    def _exponential_smooth_series(self, series: np.ndarray, periods: int, 
                                 alpha: float, beta: float) -> np.ndarray:
        """
        Apply exponential smoothing to a single time series.
        
        Args:
            series: Input time series
            periods: Number of periods to forecast
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
            
        Returns:
            Forecasted values
        """
        # Initialize level and trend
        level = series[0]
        trend = series[1] - series[0] if len(series) > 1 else 0
        
        # Apply exponential smoothing to historical data
        for i in range(1, len(series)):
            prev_level = level
            level = alpha * series[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Generate forecasts
        forecasts = []
        for h in range(1, periods + 1):
            forecast = level + h * trend
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def _seasonal_forecast_series(self, series: np.ndarray, periods: int, 
                                seasonal_periods: List[int]) -> np.ndarray:
        """
        Apply seasonal pattern forecasting to a single time series.
        
        Args:
            series: Input time series
            periods: Number of periods to forecast
            seasonal_periods: List of seasonal period lengths to consider
            
        Returns:
            Forecasted values
        """
        # Detrend the series
        x = np.arange(len(series))
        trend_coeffs = np.polyfit(x, series, 1)
        trend = np.polyval(trend_coeffs, x)
        detrended = series - trend
        
        # Extract seasonal patterns
        seasonal_components = []
        for season_length in seasonal_periods:
            if len(series) >= season_length * 2:  # Need at least 2 full seasons
                # Calculate average seasonal pattern
                n_seasons = len(series) // season_length
                seasonal_data = detrended[:n_seasons * season_length].reshape(n_seasons, season_length)
                seasonal_pattern = np.mean(seasonal_data, axis=0)
                seasonal_components.append(seasonal_pattern)
        
        # Generate forecasts
        forecasts = []
        for h in range(periods):
            # Extend trend
            future_trend = np.polyval(trend_coeffs, len(series) + h)
            
            # Add seasonal components
            seasonal_value = 0
            for j, pattern in enumerate(seasonal_components):
                season_length = seasonal_periods[j]
                seasonal_index = (len(series) + h) % season_length
                seasonal_value += pattern[seasonal_index] / len(seasonal_components)
            
            forecast = future_trend + seasonal_value
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def _generate_future_timestamps(self, periods: int) -> np.ndarray:
        """
        Generate future timestamps based on the dataset's time frequency.
        
        Args:
            periods: Number of future periods
            
        Returns:
            Array of future timestamps
        """
        timestamps = self._dataset.timestamps
        
        # Estimate time frequency from existing timestamps
        if len(timestamps) > 1:
            # Calculate most common time difference
            time_diffs = np.diff(timestamps)
            # Convert to timedelta in days for easier handling
            time_diffs_days = time_diffs / np.timedelta64(1, 'D')
            
            # Use median time difference to handle irregularities
            median_diff_days = np.median(time_diffs_days)
            time_delta = timedelta(days=median_diff_days)
        else:
            # Default to daily frequency
            time_delta = timedelta(days=1)
        
        # Generate future timestamps
        last_timestamp = pd.to_datetime(timestamps[-1])
        future_timestamps = []
        
        for i in range(1, periods + 1):
            future_timestamp = last_timestamp + i * time_delta
            future_timestamps.append(future_timestamp)
        
        return np.array(future_timestamps, dtype='datetime64[ns]')
    
    def _calculate_confidence_intervals_linear(self, periods: int, trend_window: int) -> None:
        """
        Calculate confidence intervals for linear forecasts.
        
        Args:
            periods: Number of forecasted periods
            trend_window: Window size used for trend calculation
        """
        n_symbols = len(self._dataset.symbols)
        confidence_intervals = np.zeros((periods, n_symbols, 2))
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                # Single column data
                recent_data = data_array[-trend_window:]
                
                if len(recent_data) < 2:
                    # Insufficient data for trend calculation, use simple variance
                    std_error = np.std(data_array) if len(data_array) > 1 else abs(data_array[0]) * 0.1
                    
                    for period in range(periods):
                        uncertainty = std_error * np.sqrt(1 + period)
                        forecast_value = recent_data[0]  # Use last available value
                        
                        z_score = 1.96  # 95% confidence interval
                        confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                        confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
                else:
                    # Calculate residuals from linear fit
                    x = np.arange(len(recent_data))
                    try:
                        coeffs = np.polyfit(x, recent_data, 1)
                        fitted = np.polyval(coeffs, x)
                        residuals = recent_data - fitted
                        
                        # Estimate standard error
                        std_error = np.std(residuals) if len(residuals) > 1 else np.std(recent_data) * 0.1
                        
                        # Calculate confidence intervals (95% confidence)
                        z_score = 1.96  # 95% confidence interval
                        for period in range(periods):
                            # Uncertainty increases with forecast horizon
                            uncertainty = std_error * np.sqrt(1 + (period + 1) / len(recent_data))
                            
                            # Get the forecasted value (need to recalculate)
                            future_x = len(recent_data) + period
                            forecast_value = np.polyval(coeffs, future_x)
                            
                            confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty  # Lower bound
                            confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty  # Upper bound
                    except np.linalg.LinAlgError:
                        # Fallback to simple variance-based intervals
                        std_error = np.std(recent_data)
                        forecast_value = recent_data[-1]
                        
                        z_score = 1.96
                        for period in range(periods):
                            uncertainty = std_error * np.sqrt(1 + period)
                            confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                            confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
            
            else:
                # Multi-column data - use close prices (column 3 or last column)
                close_col = min(3, data_array.shape[1] - 1)
                recent_data = data_array[-trend_window:, close_col]
                
                if len(recent_data) < 2:
                    # Insufficient data for trend calculation
                    std_error = np.std(data_array[:, close_col]) if len(data_array) > 1 else abs(data_array[0, close_col]) * 0.1
                    
                    for period in range(periods):
                        uncertainty = std_error * np.sqrt(1 + period)
                        forecast_value = recent_data[0]
                        
                        z_score = 1.96
                        confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                        confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
                else:
                    # Same calculation as single column
                    x = np.arange(len(recent_data))
                    try:
                        coeffs = np.polyfit(x, recent_data, 1)
                        fitted = np.polyval(coeffs, x)
                        residuals = recent_data - fitted
                        
                        std_error = np.std(residuals) if len(residuals) > 1 else np.std(recent_data) * 0.1
                        z_score = 1.96
                        
                        for period in range(periods):
                            uncertainty = std_error * np.sqrt(1 + (period + 1) / len(recent_data))
                            future_x = len(recent_data) + period
                            forecast_value = np.polyval(coeffs, future_x)
                            
                            confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                            confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
                    except np.linalg.LinAlgError:
                        # Fallback to simple variance-based intervals
                        std_error = np.std(recent_data)
                        forecast_value = recent_data[-1]
                        
                        z_score = 1.96
                        for period in range(periods):
                            uncertainty = std_error * np.sqrt(1 + period)
                            confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                            confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
        
        self._confidence_intervals = confidence_intervals
    
    def _calculate_confidence_intervals_exponential(self, periods: int, alpha: float, beta: float) -> None:
        """
        Calculate confidence intervals for exponential smoothing forecasts.
        
        Args:
            periods: Number of forecasted periods
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
        """
        n_symbols = len(self._dataset.symbols)
        confidence_intervals = np.zeros((periods, n_symbols, 2))
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                series = data_array
            else:
                # Use close prices
                close_col = min(3, data_array.shape[1] - 1)
                series = data_array[:, close_col]
            
            # Calculate exponential smoothing errors
            level = series[0]
            trend = series[1] - series[0] if len(series) > 1 else 0
            errors = []
            
            for j in range(1, len(series)):
                # Forecast one step ahead
                forecast = level + trend
                error = series[j] - forecast
                errors.append(error)
                
                # Update level and trend
                prev_level = level
                level = alpha * series[j] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            
            # Estimate forecast error variance
            error_variance = np.var(errors) if errors else np.var(series) * 0.1
            
            # Calculate confidence intervals
            z_score = 1.96  # 95% confidence interval
            
            for period in range(periods):
                # Forecast uncertainty increases with horizon
                forecast_variance = error_variance * (1 + period * 0.1)
                uncertainty = np.sqrt(forecast_variance)
                
                # Get the forecasted value
                forecast_value = level + (period + 1) * trend
                
                confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
        
        self._confidence_intervals = confidence_intervals
    
    def _calculate_confidence_intervals_seasonal(self, periods: int, seasonal_periods: List[int]) -> None:
        """
        Calculate confidence intervals for seasonal forecasts.
        
        Args:
            periods: Number of forecasted periods
            seasonal_periods: List of seasonal period lengths
        """
        n_symbols = len(self._dataset.symbols)
        confidence_intervals = np.zeros((periods, n_symbols, 2))
        
        for i, data_array in enumerate(self._dataset.data_arrays):
            if data_array.ndim == 1:
                series = data_array
            else:
                # Use close prices
                close_col = min(3, data_array.shape[1] - 1)
                series = data_array[:, close_col]
            
            # Estimate forecast uncertainty based on historical variance
            # For seasonal forecasts, use variance of seasonal residuals
            x = np.arange(len(series))
            trend_coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(trend_coeffs, x)
            detrended = series - trend
            
            # Calculate variance of detrended series
            seasonal_variance = np.var(detrended)
            
            # Calculate confidence intervals
            z_score = 1.96  # 95% confidence interval
            
            for period in range(periods):
                # Uncertainty increases with forecast horizon
                uncertainty = np.sqrt(seasonal_variance * (1 + period * 0.05))
                
                # Get the forecasted value (need to recalculate)
                future_trend = np.polyval(trend_coeffs, len(series) + period)
                
                # Add average seasonal component (simplified)
                seasonal_value = 0
                for season_length in seasonal_periods:
                    if len(series) >= season_length * 2:
                        seasonal_index = (len(series) + period) % season_length
                        n_seasons = len(series) // season_length
                        seasonal_data = detrended[:n_seasons * season_length].reshape(n_seasons, season_length)
                        seasonal_pattern = np.mean(seasonal_data, axis=0)
                        seasonal_value += seasonal_pattern[seasonal_index] / len(seasonal_periods)
                
                forecast_value = future_trend + seasonal_value
                
                confidence_intervals[period, i, 0] = forecast_value - z_score * uncertainty
                confidence_intervals[period, i, 1] = forecast_value + z_score * uncertainty
        
        self._confidence_intervals = confidence_intervals
    
    def __repr__(self) -> str:
        """Return string representation of the forecaster."""
        return (f"WarspiteHeuristicForecaster(dataset_length={len(self._dataset)}, "
                f"symbols={self._dataset.symbols}, method='{self._forecasting_method}')")