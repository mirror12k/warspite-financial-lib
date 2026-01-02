"""
Unit tests for WarspiteDatasetSerializer.

This module tests the serialization and deserialization functionality
extracted to the WarspiteDatasetSerializer class.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import io

from warspite_financial.datasets import WarspiteDataset, WarspiteDatasetSerializer


class TestWarspiteDatasetSerializer(unittest.TestCase):
    """Test cases for WarspiteDatasetSerializer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Sample OHLCV data for two symbols
        np.random.seed(42)  # For reproducible tests
        self.data_arrays = [
            np.random.rand(len(self.timestamps), 5) * 100,  # AAPL OHLCV
            np.random.rand(len(self.timestamps), 5) * 50    # GOOGL OHLCV
        ]
        self.symbols = ['AAPL', 'GOOGL']
        self.metadata = {
            'source': 'test',
            'created_at': '2023-01-01T00:00:00'
        }
        
        # Create test dataset
        self.dataset = WarspiteDataset(
            data_arrays=self.data_arrays,
            timestamps=self.timestamps.values,
            symbols=self.symbols,
            metadata=self.metadata
        )
        
        # Add strategy results
        self.strategy_results = np.random.rand(len(self.timestamps))
        self.dataset.add_strategy_results(self.strategy_results)
    
    def test_serialize_csv_format(self):
        """Test CSV serialization."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'csv')
        
        # Should return a string
        self.assertIsInstance(serialized, str)
        
        # Should contain expected headers
        self.assertIn('AAPL', serialized)
        self.assertIn('GOOGL', serialized)
        self.assertIn('Close', serialized)
        self.assertIn('Open', serialized)
        self.assertIn('Results', serialized)
        
        # Should contain timestamp data
        self.assertIn('2023-01-01', serialized)
    
    def test_serialize_json_format(self):
        """Test JSON serialization."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'json')
        
        # Should return a string
        self.assertIsInstance(serialized, str)
        
        # Should be valid JSON
        data = json.loads(serialized)
        
        # Check structure
        self.assertIn('timestamps', data)
        self.assertIn('symbols', data)
        self.assertIn('data_arrays', data)
        self.assertIn('metadata', data)
        self.assertIn('strategy_results', data)
        
        # Check content
        self.assertEqual(data['symbols'], self.symbols)
        self.assertEqual(len(data['timestamps']), len(self.timestamps))
        self.assertEqual(len(data['data_arrays']), len(self.symbols))
    
    def test_serialize_pickle_format(self):
        """Test Pickle serialization."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'pickle')
        
        # Should return bytes
        self.assertIsInstance(serialized, bytes)
        
        # Should be valid pickle data
        data = pickle.loads(serialized)
        
        # Check structure
        self.assertIn('timestamps', data)
        self.assertIn('symbols', data)
        self.assertIn('data_arrays', data)
        self.assertIn('metadata', data)
        self.assertIn('strategy_results', data)
    
    def test_serialize_invalid_format(self):
        """Test serialization with invalid format."""
        with self.assertRaises(ValueError) as context:
            WarspiteDatasetSerializer.serialize(self.dataset, 'invalid')
        
        self.assertIn("Unsupported format", str(context.exception))
    
    def test_deserialize_csv_format(self):
        """Test CSV deserialization."""
        # Serialize first
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'csv')
        
        # Deserialize
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'csv')
        
        # Check basic properties
        self.assertEqual(deserialized.symbols, self.symbols)
        self.assertEqual(len(deserialized), len(self.dataset))
        
        # Check data arrays (allowing for small floating point differences)
        for i, (orig, deser) in enumerate(zip(self.dataset.data_arrays, deserialized.data_arrays)):
            np.testing.assert_array_almost_equal(orig, deser, decimal=10)
        
        # Check strategy results
        np.testing.assert_array_almost_equal(
            self.dataset.strategy_results, 
            deserialized.strategy_results, 
            decimal=10
        )
    
    def test_deserialize_json_format(self):
        """Test JSON deserialization."""
        # Serialize first
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'json')
        
        # Deserialize
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'json')
        
        # Check basic properties
        self.assertEqual(deserialized.symbols, self.symbols)
        self.assertEqual(len(deserialized), len(self.dataset))
        self.assertEqual(deserialized.metadata, self.metadata)
        
        # Check data arrays
        for i, (orig, deser) in enumerate(zip(self.dataset.data_arrays, deserialized.data_arrays)):
            np.testing.assert_array_almost_equal(orig, deser)
        
        # Check strategy results
        np.testing.assert_array_almost_equal(
            self.dataset.strategy_results, 
            deserialized.strategy_results
        )
    
    def test_deserialize_pickle_format(self):
        """Test Pickle deserialization."""
        # Serialize first
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'pickle')
        
        # Deserialize
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'pickle')
        
        # Check equality (pickle should preserve exact data)
        self.assertEqual(deserialized, self.dataset)
    
    def test_deserialize_invalid_format(self):
        """Test deserialization with invalid format."""
        with self.assertRaises(ValueError) as context:
            WarspiteDatasetSerializer.deserialize("dummy data", 'invalid')
        
        self.assertIn("Unsupported format", str(context.exception))
    
    def test_round_trip_serialization_csv(self):
        """Test complete round-trip serialization for CSV."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'csv')
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'csv')
        
        # Should be functionally equivalent
        self.assertEqual(deserialized.symbols, self.dataset.symbols)
        self.assertEqual(len(deserialized), len(self.dataset))
        
        # Data should be very close (CSV has some precision loss)
        for orig, deser in zip(self.dataset.data_arrays, deserialized.data_arrays):
            np.testing.assert_array_almost_equal(orig, deser, decimal=6)
    
    def test_round_trip_serialization_json(self):
        """Test complete round-trip serialization for JSON."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'json')
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'json')
        
        # Should be functionally equivalent
        self.assertEqual(deserialized.symbols, self.dataset.symbols)
        self.assertEqual(len(deserialized), len(self.dataset))
        self.assertEqual(deserialized.metadata, self.dataset.metadata)
        
        # Data should be very close
        for orig, deser in zip(self.dataset.data_arrays, deserialized.data_arrays):
            np.testing.assert_array_almost_equal(orig, deser)
    
    def test_round_trip_serialization_pickle(self):
        """Test complete round-trip serialization for Pickle."""
        serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'pickle')
        deserialized = WarspiteDatasetSerializer.deserialize(serialized, 'pickle')
        
        # Should be exactly equal
        self.assertEqual(deserialized, self.dataset)
    
    def test_serialization_without_strategy_results(self):
        """Test serialization of dataset without strategy results."""
        # Create dataset without strategy results
        dataset_no_strategy = WarspiteDataset(
            data_arrays=self.data_arrays,
            timestamps=self.timestamps.values,
            symbols=self.symbols,
            metadata=self.metadata
        )
        
        # Test all formats
        for format_type in ['csv', 'json', 'pickle']:
            with self.subTest(format=format_type):
                serialized = WarspiteDatasetSerializer.serialize(dataset_no_strategy, format_type)
                deserialized = WarspiteDatasetSerializer.deserialize(serialized, format_type)
                
                # Should not have strategy results
                self.assertIsNone(deserialized.strategy_results)
                
                # Other data should be preserved
                self.assertEqual(deserialized.symbols, dataset_no_strategy.symbols)
                self.assertEqual(len(deserialized), len(dataset_no_strategy))
    
    def test_serialization_with_single_column_data(self):
        """Test serialization with single-column data arrays."""
        # Create dataset with single-column data (e.g., just close prices)
        single_col_data = [
            np.random.rand(len(self.timestamps)),  # AAPL close prices
            np.random.rand(len(self.timestamps))   # GOOGL close prices
        ]
        
        dataset_single_col = WarspiteDataset(
            data_arrays=single_col_data,
            timestamps=self.timestamps.values,
            symbols=self.symbols,
            metadata=self.metadata
        )
        
        # Test all formats
        for format_type in ['csv', 'json', 'pickle']:
            with self.subTest(format=format_type):
                serialized = WarspiteDatasetSerializer.serialize(dataset_single_col, format_type)
                deserialized = WarspiteDatasetSerializer.deserialize(serialized, format_type)
                
                # Check data preservation
                self.assertEqual(deserialized.symbols, dataset_single_col.symbols)
                self.assertEqual(len(deserialized), len(dataset_single_col))
                
                for orig, deser in zip(dataset_single_col.data_arrays, deserialized.data_arrays):
                    np.testing.assert_array_almost_equal(orig, deser)
    
    def test_serialization_case_insensitive_format(self):
        """Test that format parameter is case-insensitive."""
        # Test uppercase formats
        csv_upper = WarspiteDatasetSerializer.serialize(self.dataset, 'CSV')
        json_upper = WarspiteDatasetSerializer.serialize(self.dataset, 'JSON')
        pickle_upper = WarspiteDatasetSerializer.serialize(self.dataset, 'PICKLE')
        
        # Test mixed case formats
        csv_mixed = WarspiteDatasetSerializer.serialize(self.dataset, 'Csv')
        json_mixed = WarspiteDatasetSerializer.serialize(self.dataset, 'Json')
        pickle_mixed = WarspiteDatasetSerializer.serialize(self.dataset, 'Pickle')
        
        # Should work without errors
        self.assertIsInstance(csv_upper, str)
        self.assertIsInstance(json_upper, str)
        self.assertIsInstance(pickle_upper, bytes)
        self.assertIsInstance(csv_mixed, str)
        self.assertIsInstance(json_mixed, str)
        self.assertIsInstance(pickle_mixed, bytes)
    
    def test_serializer_integration_with_dataset_methods(self):
        """Test that dataset serialize/deserialize methods use the serializer."""
        # Test that dataset methods delegate to serializer
        dataset_serialized = self.dataset.serialize('json')
        serializer_serialized = WarspiteDatasetSerializer.serialize(self.dataset, 'json')
        
        # Should produce identical results
        self.assertEqual(dataset_serialized, serializer_serialized)
        
        # Test deserialization
        dataset_deserialized = WarspiteDataset.deserialize(dataset_serialized, 'json')
        serializer_deserialized = WarspiteDatasetSerializer.deserialize(serializer_serialized, 'json')
        
        # Should produce equivalent datasets
        self.assertEqual(dataset_deserialized.symbols, serializer_deserialized.symbols)
        self.assertEqual(len(dataset_deserialized), len(serializer_deserialized))


if __name__ == '__main__':
    unittest.main()