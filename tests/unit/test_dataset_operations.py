"""
Unit tests for dataset operations.

These tests verify specific examples and edge cases for dataset functionality
in the warspite_financial library.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import io

from warspite_financial.datasets.dataset import WarspiteDataset


class TestDatasetConstruction:
    """Unit tests for dataset construction and basic operations."""
    
    def test_basic_dataset_construction(self):
        """Test basic dataset construction with valid inputs."""
        # Create test data
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ], dtype='datetime64[ns]')
        
        data_arrays = [
            np.array([100.0, 101.0, 102.0]),  # Single price series
            np.array([[100.0, 105.0, 99.0, 103.0, 1000],  # OHLCV data
                     [103.0, 106.0, 102.0, 105.0, 1200],
                     [105.0, 108.0, 104.0, 107.0, 1100]])
        ]
        
        symbols = ['AAPL', 'GOOGL']
        metadata = {'source': 'test', 'created': '2023-01-01'}
        
        # Create dataset
        dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Verify construction
        assert len(dataset) == 3
        assert dataset.symbols == ['AAPL', 'GOOGL']
        assert len(dataset.data_arrays) == 2
        assert dataset.metadata == metadata
        assert dataset.strategy_results is None
        
        # Verify data preservation
        assert np.array_equal(dataset.data_arrays[0], data_arrays[0])
        assert np.array_equal(dataset.data_arrays[1], data_arrays[1])
        assert np.array_equal(dataset.timestamps, timestamps)
    
    def test_dataset_construction_validation(self):
        """Test dataset construction validation with invalid inputs."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        
        # Test empty data arrays with matching empty symbols
        with pytest.raises(ValueError, match="At least one data array is required"):
            WarspiteDataset([], timestamps, [])
        
        # Test mismatched array and symbol counts
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL', 'GOOGL']
        with pytest.raises(ValueError, match="Number of data arrays must match number of symbols"):
            WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Test mismatched array and timestamp lengths
        data_arrays = [np.array([100.0, 101.0])]  # Length 2
        symbols = ['AAPL']
        with pytest.raises(ValueError, match="Data array .* length .* doesn't match timestamps length"):
            WarspiteDataset(data_arrays, timestamps, symbols)  # timestamps length 1
    
    def test_dataset_properties_return_copies(self):
        """Test that dataset properties return copies, not references."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL']
        metadata = {'test': 'value'}
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Modify returned copies
        symbols_copy = dataset.symbols
        symbols_copy.append('GOOGL')
        assert 'GOOGL' not in dataset.symbols
        
        timestamps_copy = dataset.timestamps
        timestamps_copy[0] = np.datetime64('1900-01-01')
        assert not np.array_equal(timestamps_copy, dataset.timestamps)
        
        metadata_copy = dataset.metadata
        metadata_copy['new_key'] = 'new_value'
        assert 'new_key' not in dataset.metadata
    
    def test_dataset_equality(self):
        """Test dataset equality comparison."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL']
        metadata = {'test': 'value'}
        
        dataset1 = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        dataset2 = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Should be equal
        assert dataset1 == dataset2
        
        # Different symbols
        dataset3 = WarspiteDataset(data_arrays, timestamps, ['GOOGL'], metadata)
        assert dataset1 != dataset3
        
        # Different data
        different_data = [np.array([200.0])]
        dataset4 = WarspiteDataset(different_data, timestamps, symbols, metadata)
        assert dataset1 != dataset4
        
        # Not a dataset
        assert dataset1 != "not_a_dataset"
        assert dataset1 != None


class TestDatasetSlicing:
    """Unit tests for dataset date slicing operations."""
    
    def setup_method(self):
        """Set up test data for slicing tests."""
        # Create 5 days of data
        self.timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ], dtype='datetime64[ns]')
        
        self.data_arrays = [
            np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            np.array([200.0, 201.0, 202.0, 203.0, 204.0])
        ]
        
        self.symbols = ['AAPL', 'GOOGL']
        self.dataset = WarspiteDataset(self.data_arrays, self.timestamps, self.symbols)
    
    def test_basic_date_slicing(self):
        """Test basic date slicing functionality."""
        # Slice middle 3 days
        start_date = datetime(2023, 1, 2)
        end_date = datetime(2023, 1, 4)
        
        sliced = self.dataset.get_slice(start_date, end_date)
        
        assert len(sliced) == 3
        assert sliced.symbols == self.symbols
        
        # Check data
        expected_data1 = np.array([101.0, 102.0, 103.0])
        expected_data2 = np.array([201.0, 202.0, 203.0])
        
        assert np.array_equal(sliced.data_arrays[0], expected_data1)
        assert np.array_equal(sliced.data_arrays[1], expected_data2)
    
    def test_single_day_slice(self):
        """Test slicing a single day."""
        target_date = datetime(2023, 1, 3)
        sliced = self.dataset.get_slice(target_date, target_date)
        
        assert len(sliced) == 1
        assert sliced.data_arrays[0][0] == 102.0
        assert sliced.data_arrays[1][0] == 202.0
    
    def test_full_range_slice(self):
        """Test slicing the entire dataset range."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)
        
        sliced = self.dataset.get_slice(start_date, end_date)
        
        assert len(sliced) == len(self.dataset)
        assert np.array_equal(sliced.data_arrays[0], self.data_arrays[0])
        assert np.array_equal(sliced.data_arrays[1], self.data_arrays[1])
    
    def test_slice_with_strategy_results(self):
        """Test slicing when strategy results are present."""
        # Add strategy results
        strategy_results = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.dataset.add_strategy_results(strategy_results)
        
        # Slice middle 3 days
        start_date = datetime(2023, 1, 2)
        end_date = datetime(2023, 1, 4)
        
        sliced = self.dataset.get_slice(start_date, end_date)
        
        assert sliced.strategy_results is not None
        expected_strategy = np.array([0.2, 0.3, 0.4])
        assert np.array_equal(sliced.strategy_results, expected_strategy)
    
    def test_invalid_date_ranges(self):
        """Test error handling for invalid date ranges."""
        # Start date after end date
        with pytest.raises(ValueError, match="Start date must be before end date"):
            self.dataset.get_slice(datetime(2023, 1, 5), datetime(2023, 1, 1))
        
        # Same start and end date should work (single point slice)
        valid_slice = self.dataset.get_slice(datetime(2023, 1, 3), datetime(2023, 1, 3))
        assert len(valid_slice) == 1
        
        # Date range outside dataset
        with pytest.raises(ValueError, match="No data found in the specified date range"):
            self.dataset.get_slice(datetime(2022, 1, 1), datetime(2022, 1, 2))


class TestStrategyResults:
    """Unit tests for strategy results functionality."""
    
    def setup_method(self):
        """Set up test data."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0, 102.0])]
        symbols = ['AAPL']
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols)
    
    def test_add_strategy_results(self):
        """Test adding strategy results to dataset."""
        results = np.array([0.1, 0.2, 0.3])
        self.dataset.add_strategy_results(results)
        
        assert self.dataset.strategy_results is not None
        assert np.array_equal(self.dataset.strategy_results, results)
    
    def test_strategy_results_validation(self):
        """Test validation of strategy results length."""
        # Wrong length
        wrong_results = np.array([0.1, 0.2])  # Length 2, dataset length 3
        
        with pytest.raises(ValueError, match="Results length .* doesn't match dataset length"):
            self.dataset.add_strategy_results(wrong_results)
    
    def test_strategy_results_in_dataframe(self):
        """Test that strategy results appear in DataFrame conversion."""
        results = np.array([0.1, 0.2, 0.3])
        self.dataset.add_strategy_results(results)
        
        df = self.dataset.to_dataframe()
        assert ('Strategy', 'Results') in df.columns
        assert np.array_equal(df[('Strategy', 'Results')].values, results)


class TestDataFrameConversion:
    """Unit tests for DataFrame conversion functionality."""
    
    def test_single_column_data_conversion(self):
        """Test conversion of single-column data to DataFrame."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        ], dtype='datetime64[ns]')
        
        data_arrays = [
            np.array([100.0, 101.0]),
            np.array([200.0, 201.0])
        ]
        symbols = ['AAPL', 'GOOGL']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        df = dataset.to_dataframe()
        
        # Should have MultiIndex columns for each symbol
        assert ('AAPL', 'Close') in df.columns
        assert ('GOOGL', 'Close') in df.columns
        assert len(df) == 2
        
        # Check data
        assert df[('AAPL', 'Close')].iloc[0] == 100.0
        assert df[('GOOGL', 'Close')].iloc[1] == 201.0
    
    def test_ohlcv_data_conversion(self):
        """Test conversion of OHLCV data to DataFrame."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        
        # OHLCV data
        ohlcv_data = np.array([[100.0, 105.0, 99.0, 103.0, 1000]])
        data_arrays = [ohlcv_data]
        symbols = ['AAPL']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        df = dataset.to_dataframe()
        
        # Should have multi-level columns
        expected_columns = [('AAPL', 'Open'), ('AAPL', 'High'), ('AAPL', 'Low'), ('AAPL', 'Close'), ('AAPL', 'Volume')]
        for col in expected_columns:
            assert col in df.columns
        
        # Check data
        assert df[('AAPL', 'Open')].iloc[0] == 100.0
        assert df[('AAPL', 'High')].iloc[0] == 105.0
        assert df[('AAPL', 'Volume')].iloc[0] == 1000
    
    def test_mixed_data_conversion(self):
        """Test conversion with mixed single and OHLCV data."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        
        data_arrays = [
            np.array([100.0]),  # Single value
            np.array([[200.0, 205.0, 199.0, 203.0, 2000]])  # OHLCV
        ]
        symbols = ['SIMPLE', 'COMPLEX']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        df = dataset.to_dataframe()
        
        # Should have both types of columns
        assert ('SIMPLE', 'Close') in df.columns
        assert ('COMPLEX', 'Open') in df.columns
        assert ('COMPLEX', 'Close') in df.columns


class TestSerialization:
    """Unit tests for dataset serialization functionality."""
    
    def setup_method(self):
        """Set up test data for serialization tests."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        ], dtype='datetime64[ns]')
        
        data_arrays = [
            np.array([100.0, 101.0]),
            np.array([[200.0, 205.0, 199.0, 203.0, 2000],
                     [203.0, 206.0, 202.0, 205.0, 2100]])
        ]
        symbols = ['SIMPLE', 'COMPLEX']
        metadata = {'source': 'test', 'version': 1.0}
        
        self.dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Add strategy results
        strategy_results = np.array([0.1, 0.2])
        self.dataset.add_strategy_results(strategy_results)
    
    def test_csv_serialization(self):
        """Test CSV serialization and deserialization."""
        # Serialize
        csv_data = self.dataset.serialize('csv')
        assert isinstance(csv_data, str)
        assert len(csv_data) > 0
        
        # Should contain expected data
        assert 'SIMPLE' in csv_data
        assert 'COMPLEX' in csv_data
        assert 'Strategy' in csv_data
        
        # Deserialize
        deserialized = WarspiteDataset.deserialize(csv_data, 'csv')
        assert isinstance(deserialized, WarspiteDataset)
        
        # Check basic properties
        assert len(deserialized) == len(self.dataset)
        
        # Convert to DataFrames for comparison
        orig_df = self.dataset.to_dataframe()
        deser_df = deserialized.to_dataframe()
        
        # Data should be numerically equivalent
        for col in orig_df.columns:
            if col in deser_df.columns:
                assert np.allclose(orig_df[col].values, deser_df[col].values)
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Serialize
        json_data = self.dataset.serialize('json')
        assert isinstance(json_data, str)
        
        # Should be valid JSON
        parsed = json.loads(json_data)
        assert 'timestamps' in parsed
        assert 'symbols' in parsed
        assert 'data_arrays' in parsed
        assert 'metadata' in parsed
        assert 'strategy_results' in parsed
        
        # Deserialize
        deserialized = WarspiteDataset.deserialize(json_data, 'json')
        
        # Should be exactly equal
        assert deserialized == self.dataset
    
    def test_pickle_serialization(self):
        """Test pickle serialization and deserialization."""
        # Serialize
        pickle_data = self.dataset.serialize('pickle')
        assert isinstance(pickle_data, bytes)
        assert len(pickle_data) > 0
        
        # Deserialize
        deserialized = WarspiteDataset.deserialize(pickle_data, 'pickle')
        
        # Should be exactly equal
        assert deserialized == self.dataset
        
        # All properties should be identical
        assert deserialized.symbols == self.dataset.symbols
        assert np.array_equal(deserialized.timestamps, self.dataset.timestamps)
        assert deserialized.metadata == self.dataset.metadata
        
        for orig, deser in zip(self.dataset.data_arrays, deserialized.data_arrays):
            assert np.array_equal(orig, deser)
        
        assert np.array_equal(deserialized.strategy_results, self.dataset.strategy_results)
    
    def test_serialization_format_validation(self):
        """Test serialization format validation."""
        # Invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            self.dataset.serialize('xml')
        
        with pytest.raises(ValueError, match="Unsupported format"):
            WarspiteDataset.deserialize("dummy", 'xml')
        
        # Case insensitive
        csv_data = self.dataset.serialize('CSV')
        assert isinstance(csv_data, str)
    
    def test_serialization_without_strategy_results(self):
        """Test serialization when no strategy results are present."""
        # Create dataset without strategy results
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Test all formats
        for format_name in ['csv', 'json', 'pickle']:
            serialized = dataset.serialize(format_name)
            deserialized = WarspiteDataset.deserialize(serialized, format_name)
            
            assert deserialized.strategy_results is None
            assert len(deserialized) == 1
            assert deserialized.symbols == ['AAPL']


class TestDatasetEdgeCases:
    """Unit tests for dataset edge cases and error conditions."""
    
    def test_empty_metadata_handling(self):
        """Test dataset creation with no metadata."""
        timestamps = np.array([datetime(2023, 1, 1)], dtype='datetime64[ns]')
        data_arrays = [np.array([100.0])]
        symbols = ['AAPL']
        
        # No metadata provided
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        assert dataset.metadata == {}
        
        # None metadata
        dataset2 = WarspiteDataset(data_arrays, timestamps, symbols, None)
        assert dataset2.metadata == {}
    
    def test_large_dataset_handling(self):
        """Test dataset with larger amounts of data."""
        # Create 1000 data points
        num_points = 1000
        base_date = datetime(2020, 1, 1)
        timestamps = np.array([
            base_date + timedelta(days=i) for i in range(num_points)
        ], dtype='datetime64[ns]')
        
        # Create random data
        np.random.seed(42)  # For reproducibility
        data_arrays = [
            np.random.rand(num_points) * 100,
            np.random.rand(num_points, 5) * 100  # OHLCV
        ]
        symbols = ['STOCK1', 'STOCK2']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Basic operations should work
        assert len(dataset) == num_points
        assert len(dataset.symbols) == 2
        
        # Slicing should work
        mid_date = base_date + timedelta(days=500)
        end_date = base_date + timedelta(days=600)
        sliced = dataset.get_slice(mid_date, end_date)
        assert len(sliced) == 101  # Inclusive range
        
        # Serialization should work (test with smaller slice for speed)
        small_slice = dataset.get_slice(base_date, base_date + timedelta(days=10))
        json_data = small_slice.serialize('json')
        deserialized = WarspiteDataset.deserialize(json_data, 'json')
        assert deserialized == small_slice
    
    def test_datetime_type_conversion(self):
        """Test that different datetime types are properly converted."""
        # Test with regular datetime objects
        timestamps_dt = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        
        # Test with pandas timestamps
        timestamps_pd = pd.to_datetime(timestamps_dt)
        
        # Test with numpy datetime64
        timestamps_np = np.array(timestamps_dt, dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 101.0])]
        symbols = ['AAPL']
        
        # All should work and result in datetime64[ns]
        for timestamps in [timestamps_dt, timestamps_pd, timestamps_np]:
            dataset = WarspiteDataset(data_arrays, timestamps, symbols)
            assert dataset.timestamps.dtype == np.dtype('datetime64[ns]')
            assert len(dataset) == 2
    
    def test_dataset_representation(self):
        """Test dataset string representation."""
        timestamps = np.array([
            datetime(2023, 1, 1),
            datetime(2023, 1, 5)
        ], dtype='datetime64[ns]')
        
        data_arrays = [np.array([100.0, 105.0])]
        symbols = ['AAPL']
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        repr_str = repr(dataset)
        
        assert isinstance(repr_str, str)
        assert 'WarspiteDataset' in repr_str
        assert 'AAPL' in repr_str
        assert 'length=2' in repr_str
        assert '2023-01-01' in repr_str
        assert '2023-01-05' in repr_str
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted serialized data."""
        # Invalid JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            WarspiteDataset.deserialize('{"invalid": json}', 'json')
        
        # Invalid pickle
        with pytest.raises((pickle.UnpicklingError, ValueError)):
            WarspiteDataset.deserialize(b'invalid pickle data', 'pickle')
        
        # Invalid CSV (empty)
        with pytest.raises((pd.errors.EmptyDataError, ValueError)):
            WarspiteDataset.deserialize('', 'csv')