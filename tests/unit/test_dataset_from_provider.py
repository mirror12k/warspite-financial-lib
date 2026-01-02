"""
Unit tests for WarspiteDataset.from_provider method.

This module tests the new from_provider class method that replaces
the old from_providers method.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from warspite_financial.datasets import WarspiteDataset
from warspite_financial.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider for testing."""
    
    def __init__(self, mock_data=None):
        super().__init__()
        self.mock_data = mock_data or {}
    
    def get_data(self, symbol, start_date, end_date, interval='1d'):
        """Return mock data for the symbol."""
        if symbol in self.mock_data:
            return self.mock_data[symbol]
        else:
            # Return empty DataFrame for unknown symbols
            return pd.DataFrame()
    
    def get_available_symbols(self):
        """Return list of available symbols."""
        return list(self.mock_data.keys())
    
    def validate_symbol(self, symbol):
        """Validate if symbol is available."""
        return symbol in self.mock_data


class TestDatasetFromProvider(unittest.TestCase):
    """Test cases for WarspiteDataset.from_provider method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Create mock OHLCV data
        np.random.seed(42)  # For reproducible tests
        
        # Mock data for different symbols
        self.mock_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.rand(len(self.timestamps)) * 100 + 150,
                'High': np.random.rand(len(self.timestamps)) * 100 + 160,
                'Low': np.random.rand(len(self.timestamps)) * 100 + 140,
                'Close': np.random.rand(len(self.timestamps)) * 100 + 155,
                'Volume': np.random.randint(1000000, 10000000, len(self.timestamps))
            }, index=self.timestamps),
            
            'GOOGL': pd.DataFrame({
                'Open': np.random.rand(len(self.timestamps)) * 50 + 2500,
                'High': np.random.rand(len(self.timestamps)) * 50 + 2510,
                'Low': np.random.rand(len(self.timestamps)) * 50 + 2490,
                'Close': np.random.rand(len(self.timestamps)) * 50 + 2505,
                'Volume': np.random.randint(500000, 5000000, len(self.timestamps))
            }, index=self.timestamps),
            
            'TSLA': pd.DataFrame({
                'Close': np.random.rand(len(self.timestamps)) * 50 + 200
            }, index=self.timestamps)  # Single column data
        }
        
        self.provider = MockProvider(self.mock_data)
    
    def test_from_provider_single_symbol(self):
        """Test creating dataset from provider with single symbol."""
        symbols = ['AAPL']
        
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        # Check basic properties
        self.assertEqual(dataset.symbols, symbols)
        self.assertEqual(len(dataset), len(self.timestamps))
        self.assertEqual(len(dataset.data_arrays), 1)
        
        # Check metadata
        metadata = dataset.metadata
        self.assertTrue(metadata['created_from_provider'])
        self.assertEqual(metadata['provider_type'], 'MockProvider')
        self.assertEqual(metadata['interval'], '1d')
        
        # Check data shape (should be OHLCV - 5 columns)
        self.assertEqual(dataset.data_arrays[0].shape, (len(self.timestamps), 5))
    
    def test_from_provider_multiple_symbols(self):
        """Test creating dataset from provider with multiple symbols."""
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1d'
        )
        
        # Check basic properties
        self.assertEqual(dataset.symbols, symbols)
        self.assertEqual(len(dataset), len(self.timestamps))
        self.assertEqual(len(dataset.data_arrays), 3)
        
        # Check data shapes
        self.assertEqual(dataset.data_arrays[0].shape, (len(self.timestamps), 5))  # AAPL OHLCV
        self.assertEqual(dataset.data_arrays[1].shape, (len(self.timestamps), 5))  # GOOGL OHLCV
        self.assertEqual(dataset.data_arrays[2].shape, (len(self.timestamps),))    # TSLA single column
    
    def test_from_provider_empty_symbols_list(self):
        """Test error handling with empty symbols list."""
        with self.assertRaises(ValueError) as context:
            WarspiteDataset.from_provider(
                provider=self.provider,
                symbols=[],
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        self.assertIn("At least one symbol must be provided", str(context.exception))
    
    def test_from_provider_invalid_date_range(self):
        """Test error handling with invalid date range."""
        with self.assertRaises(ValueError) as context:
            WarspiteDataset.from_provider(
                provider=self.provider,
                symbols=['AAPL'],
                start_date=self.end_date,  # Start after end
                end_date=self.start_date
            )
        
        self.assertIn("Start date must be before end date", str(context.exception))
    
    def test_from_provider_unknown_symbol(self):
        """Test handling of unknown symbols."""
        symbols = ['AAPL', 'UNKNOWN_SYMBOL', 'GOOGL']
        
        # Should succeed but skip unknown symbol
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Should only have data for known symbols
        self.assertEqual(dataset.symbols, ['AAPL', 'GOOGL'])
        self.assertEqual(len(dataset.data_arrays), 2)
    
    def test_from_provider_all_unknown_symbols(self):
        """Test error handling when all symbols are unknown."""
        with self.assertRaises(ValueError) as context:
            WarspiteDataset.from_provider(
                provider=self.provider,
                symbols=['UNKNOWN1', 'UNKNOWN2'],
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        self.assertIn("No data could be retrieved for any symbol", str(context.exception))
    
    def test_from_provider_with_provider_exception(self):
        """Test handling of provider exceptions."""
        # Create a provider that raises exceptions
        failing_provider = Mock(spec=BaseProvider)
        failing_provider.get_data.side_effect = Exception("Provider error")
        
        with self.assertRaises(ValueError) as context:
            WarspiteDataset.from_provider(
                provider=failing_provider,
                symbols=['AAPL'],
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        self.assertIn("No data could be retrieved for any symbol", str(context.exception))
    
    def test_from_provider_partial_failure(self):
        """Test handling when some symbols fail to load."""
        # Create a provider that fails for specific symbols
        partial_provider = Mock(spec=BaseProvider)
        
        def mock_get_data(symbol, start_date, end_date, interval='1d'):
            if symbol == 'AAPL':
                return self.mock_data['AAPL']
            elif symbol == 'FAIL':
                raise Exception("Failed to get data")
            else:
                return pd.DataFrame()  # Empty for unknown
        
        partial_provider.get_data.side_effect = mock_get_data
        
        # Should succeed with only successful symbols
        dataset = WarspiteDataset.from_provider(
            provider=partial_provider,
            symbols=['AAPL', 'FAIL', 'UNKNOWN'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Should only have AAPL data
        self.assertEqual(dataset.symbols, ['AAPL'])
        self.assertEqual(len(dataset.data_arrays), 1)
    
    def test_from_provider_metadata_content(self):
        """Test that metadata contains expected information."""
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=['AAPL'],
            start_date=self.start_date,
            end_date=self.end_date,
            interval='1h'
        )
        
        metadata = dataset.metadata
        
        # Check all expected metadata fields
        self.assertTrue(metadata['created_from_provider'])
        self.assertEqual(metadata['provider_type'], 'MockProvider')
        self.assertEqual(metadata['start_date'], self.start_date.isoformat())
        self.assertEqual(metadata['end_date'], self.end_date.isoformat())
        self.assertEqual(metadata['interval'], '1h')
        self.assertIn('creation_timestamp', metadata)
        
        # Verify timestamp format
        creation_time = datetime.fromisoformat(metadata['creation_timestamp'])
        self.assertIsInstance(creation_time, datetime)
    
    def test_from_provider_different_intervals(self):
        """Test from_provider with different intervals."""
        intervals = ['1d', '1h', '5m']
        
        for interval in intervals:
            with self.subTest(interval=interval):
                dataset = WarspiteDataset.from_provider(
                    provider=self.provider,
                    symbols=['AAPL'],
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=interval
                )
                
                # Should succeed and store interval in metadata
                self.assertEqual(dataset.metadata['interval'], interval)
    
    def test_from_provider_timestamps_consistency(self):
        """Test that timestamps are consistent across symbols."""
        dataset = WarspiteDataset.from_provider(
            provider=self.provider,
            symbols=['AAPL', 'GOOGL'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # All data arrays should have same length as timestamps
        for data_array in dataset.data_arrays:
            self.assertEqual(len(data_array), len(dataset.timestamps))
        
        # Timestamps should match our expected range
        expected_timestamps = self.timestamps.values
        np.testing.assert_array_equal(dataset.timestamps, expected_timestamps)
    
    @patch('builtins.print')  # Mock print to capture warnings
    def test_from_provider_warning_messages(self, mock_print):
        """Test that warning messages are printed for failed symbols."""
        # Create provider that fails for one symbol
        partial_provider = Mock(spec=BaseProvider)
        
        def mock_get_data(symbol, start_date, end_date, interval='1d'):
            if symbol == 'AAPL':
                return self.mock_data['AAPL']
            else:
                raise Exception("Test error")
        
        partial_provider.get_data.side_effect = mock_get_data
        
        dataset = WarspiteDataset.from_provider(
            provider=partial_provider,
            symbols=['AAPL', 'FAIL'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Should have printed a warning
        mock_print.assert_called()
        warning_call = mock_print.call_args[0][0]
        self.assertIn("Warning: Failed to get data for FAIL", warning_call)


if __name__ == '__main__':
    unittest.main()