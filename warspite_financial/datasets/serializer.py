"""
Serialization utilities for WarspiteDataset.

This module contains the WarspiteDatasetSerializer class for handling
serialization and deserialization of WarspiteDataset objects.
"""

from typing import Union, Dict, Any, List
import json
import pickle
import io
import numpy as np
import pandas as pd


class WarspiteDatasetSerializer:
    """
    Handles serialization and deserialization of WarspiteDataset objects.
    
    Supports multiple formats: CSV, JSON, and Pickle.
    """
    
    @staticmethod
    def serialize(dataset: 'WarspiteDataset', format: str = 'csv') -> Union[str, bytes]:
        """
        Serialize a WarspiteDataset to a string or bytes.
        
        Args:
            dataset: The WarspiteDataset to serialize
            format: Serialization format ('csv', 'json', or 'pickle')
            
        Returns:
            Serialized data as string (csv, json) or bytes (pickle)
            
        Raises:
            ValueError: If format is not supported
        """
        format_lower = format.lower()
        if format_lower == 'csv':
            return WarspiteDatasetSerializer._serialize_csv(dataset)
        elif format_lower == 'json':
            return WarspiteDatasetSerializer._serialize_json(dataset)
        elif format_lower == 'pickle':
            return WarspiteDatasetSerializer._serialize_pickle(dataset)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'pickle'")
    
    @staticmethod
    def deserialize(data: Union[str, bytes], format: str = 'csv') -> 'WarspiteDataset':
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
        format_lower = format.lower()
        if format_lower == 'csv':
            return WarspiteDatasetSerializer._deserialize_csv(data)
        elif format_lower == 'json':
            return WarspiteDatasetSerializer._deserialize_json(data)
        elif format_lower == 'pickle':
            return WarspiteDatasetSerializer._deserialize_pickle(data)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'pickle'")
    
    @staticmethod
    def _serialize_csv(dataset: 'WarspiteDataset') -> str:
        """Serialize to CSV format."""
        df = dataset.to_dataframe()
        return df.to_csv()
    
    @staticmethod
    def _serialize_json(dataset: 'WarspiteDataset') -> str:
        """Serialize to JSON format."""
        # Convert to a JSON-serializable format
        data = {
            'timestamps': [ts.isoformat() for ts in pd.to_datetime(dataset.timestamps)],
            'symbols': dataset.symbols,
            'data_arrays': [arr.tolist() for arr in dataset.data_arrays],
            'metadata': dataset.metadata
        }
        
        if dataset.strategy_results is not None:
            data['strategy_results'] = dataset.strategy_results.tolist()
        
        return json.dumps(data, indent=2)
    
    @staticmethod
    def _serialize_pickle(dataset: 'WarspiteDataset') -> bytes:
        """Serialize to pickle format."""
        data = {
            'data_arrays': dataset.data_arrays,
            'timestamps': dataset.timestamps,
            'symbols': dataset.symbols,
            'metadata': dataset.metadata,
            'strategy_results': dataset.strategy_results
        }
        
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        return buffer.getvalue()
    
    @staticmethod
    def _deserialize_csv(data: str) -> 'WarspiteDataset':
        """Deserialize from CSV format."""
        from .dataset import WarspiteDataset  # Import here to avoid circular imports
        
        # Read CSV with MultiIndex columns
        df = pd.read_csv(io.StringIO(data), index_col=0, header=[0, 1], parse_dates=True)
        
        # Extract symbols and data arrays from DataFrame
        symbols = []
        data_arrays = []
        strategy_results = None
        
        # Group by symbol (first level of MultiIndex)
        for symbol in df.columns.get_level_values(0).unique():
            if symbol == 'Strategy':
                # Handle strategy results
                if ('Strategy', 'Results') in df.columns:
                    strategy_results = df[('Strategy', 'Results')].values
                continue
            
            symbols.append(symbol)
            symbol_data = df[symbol]
            
            if len(symbol_data.columns) == 1:
                # Single column (Close price)
                data_arrays.append(symbol_data.iloc[:, 0].values)
            else:
                # Multiple columns (OHLCV)
                # Reorder columns to OHLCV format
                ohlcv_order = ['Open', 'High', 'Low', 'Close', 'Volume']
                ordered_data = []
                for field in ohlcv_order:
                    if field in symbol_data.columns:
                        ordered_data.append(symbol_data[field].values)
                
                if ordered_data:
                    data_arrays.append(np.column_stack(ordered_data))
                else:
                    # Fallback: use all available columns
                    data_arrays.append(symbol_data.values)
        
        # Create dataset
        dataset = WarspiteDataset(
            data_arrays=data_arrays,
            timestamps=df.index.values,
            symbols=symbols
        )
        
        # Add strategy results if present
        if strategy_results is not None:
            dataset.add_strategy_results(strategy_results)
        
        return dataset
    
    @staticmethod
    def _deserialize_json(data: str) -> 'WarspiteDataset':
        """Deserialize from JSON format."""
        from .dataset import WarspiteDataset  # Import here to avoid circular imports
        
        parsed = json.loads(data)
        
        # Convert timestamps back to datetime64
        timestamps = np.array([np.datetime64(ts) for ts in parsed['timestamps']])
        
        # Convert data arrays back to numpy arrays
        data_arrays = [np.array(arr) for arr in parsed['data_arrays']]
        
        # Create dataset
        dataset = WarspiteDataset(
            data_arrays=data_arrays,
            timestamps=timestamps,
            symbols=parsed['symbols'],
            metadata=parsed.get('metadata', {})
        )
        
        # Add strategy results if present
        if 'strategy_results' in parsed:
            dataset.add_strategy_results(np.array(parsed['strategy_results']))
        
        return dataset
    
    @staticmethod
    def _deserialize_pickle(data: bytes) -> 'WarspiteDataset':
        """Deserialize from pickle format."""
        from .dataset import WarspiteDataset  # Import here to avoid circular imports
        
        buffer = io.BytesIO(data)
        parsed = pickle.load(buffer)
        
        # Create dataset
        dataset = WarspiteDataset(
            data_arrays=parsed['data_arrays'],
            timestamps=parsed['timestamps'],
            symbols=parsed['symbols'],
            metadata=parsed.get('metadata', {})
        )
        
        # Add strategy results if present
        if parsed.get('strategy_results') is not None:
            dataset._strategy_results = parsed['strategy_results']
        
        return dataset