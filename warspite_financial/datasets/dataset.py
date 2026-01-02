"""
Dataset management for warspite_financial library.

This module contains the WarspiteDataset class for managing financial time-series data.
"""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from .serializer import WarspiteDatasetSerializer


class WarspiteDataset:
    """
    A dataset class for managing financial time-series data.
    
    This class stores financial data as numpy arrays with associated timestamps
    and symbols, providing methods for data manipulation, slicing, and serialization.
    """
    
    def __init__(self, data_arrays: List[np.ndarray], 
                 timestamps: np.ndarray, 
                 symbols: List[str],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset with data arrays, timestamps, and symbols.
        
        Args:
            data_arrays: List of numpy arrays containing financial data (OHLCV format expected)
            timestamps: Array of timestamps (datetime64 or datetime objects)
            symbols: List of symbol names corresponding to data_arrays
            metadata: Optional metadata dictionary
            
        Raises:
            ValueError: If dimensions don't match or data is invalid
        """
        # Validate inputs
        if len(data_arrays) != len(symbols):
            raise ValueError("Number of data arrays must match number of symbols")
        
        if len(data_arrays) == 0:
            raise ValueError("At least one data array is required")
        
        # Ensure all data arrays have the same length as timestamps
        timestamp_length = len(timestamps)
        for i, array in enumerate(data_arrays):
            if len(array) != timestamp_length:
                raise ValueError(f"Data array {i} length ({len(array)}) doesn't match timestamps length ({timestamp_length})")
        
        # Store data
        self._data_arrays = [np.array(arr) for arr in data_arrays]
        
        # Convert timestamps to numpy datetime64[ns] if needed
        if hasattr(timestamps, 'dtype') and timestamps.dtype == 'datetime64[ns]':
            self._timestamps = timestamps
        else:
            self._timestamps = np.array(timestamps, dtype='datetime64[ns]')
            
        self._symbols = list(symbols)
        self._metadata = metadata or {}
        
        # Initialize strategy results storage
        self._strategy_results: Optional[np.ndarray] = None
        
    @property
    def data_arrays(self) -> List[np.ndarray]:
        """Get the data arrays."""
        return self._data_arrays.copy()
    
    @property
    def data(self) -> np.ndarray:
        """Get the data as a single numpy array (backward compatibility)."""
        if len(self._data_arrays) == 1:
            return self._data_arrays[0].copy()
        else:
            # For multiple symbols, concatenate all data arrays horizontally
            # This maintains backward compatibility with tests expecting shape (timestamps, symbols * fields)
            return np.concatenate(self._data_arrays, axis=1)
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get the timestamps."""
        return self._timestamps.copy()
    
    @property
    def symbols(self) -> List[str]:
        """Get the symbols."""
        return self._symbols.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata."""
        return self._metadata.copy()
    
    @property
    def strategy_results(self) -> Optional[np.ndarray]:
        """Get the strategy results if available."""
        return self._strategy_results.copy() if self._strategy_results is not None else None
    
    def get_slice(self, start_date: datetime, end_date: datetime) -> 'WarspiteDataset':
        """
        Get a slice of the dataset for a specific date range.
        
        Args:
            start_date: Start date for the slice
            end_date: End date for the slice
            
        Returns:
            New WarspiteDataset containing only data within the date range
            
        Raises:
            ValueError: If date range is invalid
        """
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        # Convert to numpy datetime64 for comparison
        start_dt64 = np.datetime64(start_date)
        end_dt64 = np.datetime64(end_date)
        
        # Find indices within the date range
        mask = (self._timestamps >= start_dt64) & (self._timestamps <= end_dt64)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError("No data found in the specified date range")
        
        # Slice all data arrays
        sliced_arrays = [arr[indices] for arr in self._data_arrays]
        sliced_timestamps = self._timestamps[indices]
        
        # Create new dataset with sliced data
        sliced_dataset = WarspiteDataset(
            data_arrays=sliced_arrays,
            timestamps=sliced_timestamps,
            symbols=self._symbols,
            metadata=self._metadata
        )
        
        # Slice strategy results if they exist
        if self._strategy_results is not None:
            sliced_dataset._strategy_results = self._strategy_results[indices]
        
        return sliced_dataset
    
    def add_strategy_results(self, results: np.ndarray) -> None:
        """
        Add strategy results to the dataset.
        
        Args:
            results: Array of strategy results. Can be:
                    - 1D array with same length as timestamps (legacy single-symbol format)
                    - 2D array with shape (n_timestamps, n_symbols) for multi-symbol positions
            
        Raises:
            ValueError: If results dimensions don't match dataset structure
        """
        if results.ndim == 1:
            # Legacy format: single column of results
            if len(results) != len(self._timestamps):
                raise ValueError(f"Results length ({len(results)}) doesn't match dataset length ({len(self._timestamps)})")
            self._strategy_results = np.array(results)
        elif results.ndim == 2:
            # New format: multi-symbol positions
            if results.shape[0] != len(self._timestamps):
                raise ValueError(f"Results first dimension ({results.shape[0]}) doesn't match dataset length ({len(self._timestamps)})")
            if results.shape[1] != len(self._symbols):
                raise ValueError(f"Results second dimension ({results.shape[1]}) doesn't match number of symbols ({len(self._symbols)})")
            self._strategy_results = np.array(results)
        else:
            raise ValueError(f"Results must be 1D or 2D array, got {results.ndim}D")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.
        
        Returns:
            DataFrame with timestamps as index and 2-level MultiIndex columns (symbol, field)
        """
        # Create a dictionary to hold all data with MultiIndex columns
        data_dict = {}
        
        # Add data for each symbol
        for i, symbol in enumerate(self._symbols):
            data_array = self._data_arrays[i]
            
            # If data array is 2D (OHLCV), create multi-level columns
            if data_array.ndim == 2:
                columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for j, col in enumerate(columns):
                    if j < data_array.shape[1]:
                        data_dict[(symbol, col)] = data_array[:, j]
            else:
                # If 1D, assume it's close prices
                data_dict[(symbol, 'Close')] = data_array
        
        # Add strategy results if available
        if self._strategy_results is not None:
            if self._strategy_results.ndim == 1:
                # Legacy format: single column
                data_dict[('Strategy', 'Results')] = self._strategy_results
            else:
                # Multi-symbol format: add position for each symbol
                for i, symbol in enumerate(self._symbols):
                    data_dict[(symbol, 'Position')] = self._strategy_results[:, i]
        
        # Create DataFrame with MultiIndex columns
        df = pd.DataFrame(data_dict, index=pd.to_datetime(self._timestamps))
        
        # Set proper MultiIndex column names
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Symbol', 'Field'])
        
        return df
    
    def to_numpy_multiindex(self) -> np.ndarray:
        """
        Convert the dataset to a numpy array with 2-level MultiIndex structure.
        
        Returns:
            numpy array where first level is symbol index, second level is field index
            Shape: (n_timestamps, n_symbols, n_fields)
        """
        # Determine maximum number of fields across all symbols
        max_fields = 1  # At least Close price
        for data_array in self._data_arrays:
            if data_array.ndim == 2:
                max_fields = max(max_fields, data_array.shape[1])
        
        # Create 3D array: (timestamps, symbols, fields)
        n_timestamps = len(self._timestamps)
        n_symbols = len(self._symbols)
        
        # Initialize with NaN
        result = np.full((n_timestamps, n_symbols, max_fields), np.nan)
        
        # Fill the array
        for i, data_array in enumerate(self._data_arrays):
            if data_array.ndim == 1:
                # Single column data (Close price)
                result[:, i, 0] = data_array
            else:
                # Multi-column data (OHLCV)
                n_cols = min(data_array.shape[1], max_fields)
                result[:, i, :n_cols] = data_array[:, :n_cols]
        
        return result
    
    def serialize(self, format: str = 'csv') -> Union[str, bytes]:
        """
        Serialize the dataset to a string or bytes.
        
        Args:
            format: Serialization format ('csv', 'json', or 'pickle')
            
        Returns:
            Serialized data as string (csv, json) or bytes (pickle)
            
        Raises:
            ValueError: If format is not supported
        """
        return WarspiteDatasetSerializer.serialize(self, format)
    
    @classmethod
    def deserialize(cls, data: Union[str, bytes], format: str = 'csv') -> 'WarspiteDataset':
        """
        Deserialize data to create a WarspiteDataset.
        
        Args:
            data: Serialized data
            format: Format of the serialized data ('csv', 'json', or 'pickle')
            
        Returns:
            New WarspiteDataset instance
            
        Raises:
            ValueError: If format is not supported or data is invalid
        """
        return WarspiteDatasetSerializer.deserialize(data, format)
    
    
    def __len__(self) -> int:
        """Return the length of the dataset (number of time points)."""
        return len(self._timestamps)
    
    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (f"WarspiteDataset(symbols={self._symbols}, "
                f"length={len(self)}, "
                f"date_range={self._timestamps[0]} to {self._timestamps[-1]})")
    
    def __eq__(self, other) -> bool:
        """Check equality with another WarspiteDataset."""
        if not isinstance(other, WarspiteDataset):
            return False
        
        # Check basic properties
        if (self._symbols != other._symbols or 
            not np.array_equal(self._timestamps, other._timestamps) or
            self._metadata != other._metadata):
            return False
        
        # Check data arrays
        if len(self._data_arrays) != len(other._data_arrays):
            return False
        
        for arr1, arr2 in zip(self._data_arrays, other._data_arrays):
            if not np.array_equal(arr1, arr2):
                return False
        
        # Check strategy results
        if self._strategy_results is None and other._strategy_results is None:
            return True
        elif self._strategy_results is None or other._strategy_results is None:
            return False
        else:
            return np.array_equal(self._strategy_results, other._strategy_results)
    
    @classmethod
    def from_provider(cls, provider: 'BaseProvider', 
                     symbols: List[str], 
                     start_date: datetime, 
                     end_date: datetime,
                     interval: str = '1d') -> 'WarspiteDataset':
        """
        Create a WarspiteDataset from a single provider with multiple symbols.
        
        Args:
            provider: Data provider to fetch data from
            symbols: List of symbols to fetch from the provider
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval (default '1d')
            
        Returns:
            New WarspiteDataset containing data from the provider
            
        Raises:
            ValueError: If data retrieval fails or no valid data is found
        """
        if not symbols:
            raise ValueError("At least one symbol must be provided")
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        data_arrays = []
        timestamps = None
        successful_symbols = []
        
        for symbol in symbols:
            try:
                # Get data from provider
                df = provider.get_data(symbol, start_date, end_date, interval)
                
                if df.empty:
                    continue  # Skip empty data
                
                # Use timestamps from first successful symbol
                if timestamps is None:
                    timestamps = df.index.values
                
                # Convert DataFrame to numpy array
                if len(df.columns) == 1:
                    # Single column (e.g., just Close price)
                    data_array = df.iloc[:, 0].values
                else:
                    # Multiple columns (OHLCV)
                    data_array = df.values
                
                data_arrays.append(data_array)
                successful_symbols.append(symbol)
                
            except Exception as e:
                # Log warning but continue with other symbols
                print(f"Warning: Failed to get data for {symbol} from provider: {e}")
                continue
        
        if len(data_arrays) == 0:
            raise ValueError("No data could be retrieved for any symbol")
        
        # Create metadata with provider information
        metadata = {
            'created_from_provider': True,
            'provider_type': type(provider).__name__,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'interval': interval,
            'creation_timestamp': datetime.now().isoformat()
        }
        
        return cls(data_arrays, timestamps, successful_symbols, metadata)
    
    def resample(self, new_interval: str) -> 'WarspiteDataset':
        """
        Resample the dataset to a different time interval.
        
        Args:
            new_interval: New time interval ('1d', '1h', '1w', etc.)
            
        Returns:
            New WarspiteDataset with resampled data
            
        Raises:
            ValueError: If resampling fails or interval is invalid
        """
        # Convert to DataFrame for easier resampling
        df = self.to_dataframe()
        
        # Remove strategy results column for resampling (will be lost)
        if 'Strategy_Results' in df.columns:
            df = df.drop('Strategy_Results', axis=1)
        
        # Map interval to pandas frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D',
            '1w': '1W',
            '1mo': '1ME'
        }
        
        if new_interval not in freq_map:
            raise ValueError(f"Unsupported interval: {new_interval}")
        
        pandas_freq = freq_map[new_interval]
        
        # Resample the data
        # For OHLCV data, use appropriate aggregation methods
        agg_dict = {}
        for col in df.columns:
            if col.endswith('_Open') or col == 'Open':
                agg_dict[col] = 'first'
            elif col.endswith('_High') or col == 'High':
                agg_dict[col] = 'max'
            elif col.endswith('_Low') or col == 'Low':
                agg_dict[col] = 'min'
            elif col.endswith('_Close') or col == 'Close':
                agg_dict[col] = 'last'
            elif col.endswith('_Volume') or col == 'Volume':
                agg_dict[col] = 'sum'
            else:
                # Default to last value for other columns
                agg_dict[col] = 'last'
        
        resampled_df = df.resample(pandas_freq).agg(agg_dict).dropna()
        
        if resampled_df.empty:
            raise ValueError("Resampling resulted in empty dataset")
        
        # Convert back to WarspiteDataset format
        new_symbols = []
        new_data_arrays = []
        
        # Group columns by symbol
        symbol_columns = {}
        for col in resampled_df.columns:
            if '_' in col:
                symbol, field = col.rsplit('_', 1)
                if symbol not in symbol_columns:
                    symbol_columns[symbol] = {}
                symbol_columns[symbol][field] = resampled_df[col].values
            else:
                symbol_columns[col] = {'Close': resampled_df[col].values}
        
        # Convert to arrays
        for symbol, fields in symbol_columns.items():
            new_symbols.append(symbol)
            if len(fields) > 1:
                # Multi-column data (OHLCV)
                ohlcv_order = ['Open', 'High', 'Low', 'Close', 'Volume']
                array_data = []
                for field in ohlcv_order:
                    if field in fields:
                        array_data.append(fields[field])
                new_data_arrays.append(np.column_stack(array_data))
            else:
                # Single column data
                new_data_arrays.append(list(fields.values())[0])
        
        # Create new metadata
        new_metadata = self._metadata.copy()
        new_metadata['resampled_from'] = getattr(self._metadata, 'interval', 'unknown')
        new_metadata['resampled_to'] = new_interval
        new_metadata['resample_timestamp'] = datetime.now().isoformat()
        
        return WarspiteDataset(
            data_arrays=new_data_arrays,
            timestamps=resampled_df.index.values,
            symbols=new_symbols,
            metadata=new_metadata
        )
    
    def merge_with(self, other: 'WarspiteDataset', how: str = 'inner') -> 'WarspiteDataset':
        """
        Merge this dataset with another dataset on timestamps.
        
        Args:
            other: Another WarspiteDataset to merge with
            how: How to handle the merge ('inner', 'outer', 'left', 'right')
            
        Returns:
            New WarspiteDataset containing merged data
            
        Raises:
            ValueError: If merge fails or datasets are incompatible
        """
        if not isinstance(other, WarspiteDataset):
            raise ValueError("Can only merge with another WarspiteDataset")
        
        # Convert both datasets to DataFrames
        df1 = self.to_dataframe()
        df2 = other.to_dataframe()
        
        # Remove strategy results for merging (they would conflict)
        if 'Strategy_Results' in df1.columns:
            df1 = df1.drop('Strategy_Results', axis=1)
        if 'Strategy_Results' in df2.columns:
            df2 = df2.drop('Strategy_Results', axis=1)
        
        # Merge on index (timestamps)
        merged_df = df1.join(df2, how=how, rsuffix='_other')
        
        if merged_df.empty:
            raise ValueError("Merge resulted in empty dataset")
        
        # Convert back to WarspiteDataset format
        new_symbols = []
        new_data_arrays = []
        
        # Group columns by symbol
        symbol_columns = {}
        for col in merged_df.columns:
            if col.endswith('_other'):
                # Handle suffixed columns from the other dataset
                base_col = col[:-6]  # Remove '_other' suffix
                if '_' in base_col:
                    symbol, field = base_col.rsplit('_', 1)
                    symbol = f"{symbol}_other"
                else:
                    symbol = f"{base_col}_other"
                    field = 'Close'
            else:
                if '_' in col:
                    symbol, field = col.rsplit('_', 1)
                else:
                    symbol = col
                    field = 'Close'
            
            if symbol not in symbol_columns:
                symbol_columns[symbol] = {}
            symbol_columns[symbol][field] = merged_df[col].values
        
        # Convert to arrays
        for symbol, fields in symbol_columns.items():
            new_symbols.append(symbol)
            if len(fields) > 1:
                # Multi-column data (OHLCV)
                ohlcv_order = ['Open', 'High', 'Low', 'Close', 'Volume']
                array_data = []
                for field in ohlcv_order:
                    if field in fields:
                        array_data.append(fields[field])
                new_data_arrays.append(np.column_stack(array_data))
            else:
                # Single column data
                new_data_arrays.append(list(fields.values())[0])
        
        # Create merged metadata
        merged_metadata = {
            'merged_from': [self._metadata, other._metadata],
            'merge_type': how,
            'merge_timestamp': datetime.now().isoformat()
        }
        
        return WarspiteDataset(
            data_arrays=new_data_arrays,
            timestamps=merged_df.index.values,
            symbols=new_symbols,
            metadata=merged_metadata
        )
    
    def calculate_returns(self, method: str = 'simple') -> 'WarspiteDataset':
        """
        Calculate returns for all symbols in the dataset.
        
        Args:
            method: Return calculation method ('simple' or 'log')
            
        Returns:
            New WarspiteDataset containing return data
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in ['simple', 'log']:
            raise ValueError("Method must be 'simple' or 'log'")
        
        new_data_arrays = []
        
        for i, array in enumerate(self._data_arrays):
            if array.ndim == 1:
                # Single price series
                prices = array
            else:
                # Multi-column data, use Close prices (column 3)
                if array.shape[1] > 3:
                    prices = array[:, 3]  # Close prices
                else:
                    prices = array[:, -1]  # Last column
            
            # Calculate returns
            if method == 'simple':
                returns = np.diff(prices) / prices[:-1]
            else:  # log returns
                returns = np.diff(np.log(prices))
            
            # Pad with NaN for first observation
            returns = np.concatenate([[np.nan], returns])
            new_data_arrays.append(returns)
        
        # Create new metadata
        new_metadata = self._metadata.copy()
        new_metadata['returns_method'] = method
        new_metadata['returns_calculation_timestamp'] = datetime.now().isoformat()
        new_metadata['original_data_type'] = 'prices'
        
        return WarspiteDataset(
            data_arrays=new_data_arrays,
            timestamps=self._timestamps,
            symbols=[f"{symbol}_returns" for symbol in self._symbols],
            metadata=new_metadata
        )
    
    def rolling_window(self, window_size: int, operation: str = 'mean') -> 'WarspiteDataset':
        """
        Apply a rolling window operation to the dataset.
        
        Args:
            window_size: Size of the rolling window
            operation: Operation to apply ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            New WarspiteDataset with rolling window results
            
        Raises:
            ValueError: If window_size is invalid or operation is not supported
        """
        if window_size < 1 or window_size > len(self):
            raise ValueError(f"Window size must be between 1 and {len(self)}")
        
        if operation not in ['mean', 'std', 'min', 'max', 'sum']:
            raise ValueError("Operation must be one of: 'mean', 'std', 'min', 'max', 'sum'")
        
        # Convert to DataFrame for easier rolling operations
        df = self.to_dataframe()
        
        # Remove strategy results column
        if 'Strategy_Results' in df.columns:
            df = df.drop('Strategy_Results', axis=1)
        
        # Apply rolling operation
        if operation == 'mean':
            rolled_df = df.rolling(window=window_size).mean()
        elif operation == 'std':
            rolled_df = df.rolling(window=window_size).std()
        elif operation == 'min':
            rolled_df = df.rolling(window=window_size).min()
        elif operation == 'max':
            rolled_df = df.rolling(window=window_size).max()
        elif operation == 'sum':
            rolled_df = df.rolling(window=window_size).sum()
        
        # Drop NaN values from the beginning
        rolled_df = rolled_df.dropna()
        
        if rolled_df.empty:
            raise ValueError("Rolling operation resulted in empty dataset")
        
        # Convert back to WarspiteDataset format (similar to resample method)
        new_symbols = []
        new_data_arrays = []
        
        # Group columns by symbol
        symbol_columns = {}
        for col in rolled_df.columns:
            if '_' in col:
                symbol, field = col.rsplit('_', 1)
                if symbol not in symbol_columns:
                    symbol_columns[symbol] = {}
                symbol_columns[symbol][field] = rolled_df[col].values
            else:
                symbol_columns[col] = {'Close': rolled_df[col].values}
        
        # Convert to arrays
        for symbol, fields in symbol_columns.items():
            new_symbols.append(f"{symbol}_{operation}_{window_size}")
            if len(fields) > 1:
                # Multi-column data (OHLCV)
                ohlcv_order = ['Open', 'High', 'Low', 'Close', 'Volume']
                array_data = []
                for field in ohlcv_order:
                    if field in fields:
                        array_data.append(fields[field])
                new_data_arrays.append(np.column_stack(array_data))
            else:
                # Single column data
                new_data_arrays.append(list(fields.values())[0])
        
        # Create new metadata
        new_metadata = self._metadata.copy()
        new_metadata['rolling_operation'] = operation
        new_metadata['rolling_window_size'] = window_size
        new_metadata['rolling_timestamp'] = datetime.now().isoformat()
        
        return WarspiteDataset(
            data_arrays=new_data_arrays,
            timestamps=rolled_df.index.values,
            symbols=new_symbols,
            metadata=new_metadata
        )