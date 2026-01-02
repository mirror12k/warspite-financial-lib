"""
Property-based tests for dataset management.

These tests verify universal properties that should hold across all valid inputs
for the dataset system in the warspite_financial library.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from warspite_financial.datasets.dataset import WarspiteDataset


# Hypothesis strategies for generating test data
@st.composite
def valid_symbols(draw):
    """Generate valid symbol lists for testing."""
    num_symbols = draw(st.integers(1, 5))  # 1 to 5 symbols
    symbols = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=num_symbols,
        max_size=num_symbols,
        unique=True
    ))
    return symbols


@st.composite
def valid_timestamps(draw):
    """Generate valid timestamp arrays for testing."""
    num_timestamps = draw(st.integers(1, 100))  # 1 to 100 timestamps
    
    # Generate a base date and create sequential timestamps
    base_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 1, 1)
    ))
    
    # Create sequential timestamps (daily intervals)
    timestamps = []
    for i in range(num_timestamps):
        timestamps.append(base_date + timedelta(days=i))
    
    return np.array(timestamps, dtype='datetime64[ns]')


@st.composite
def valid_data_arrays(draw, num_symbols, num_timestamps):
    """Generate valid data arrays for testing."""
    data_arrays = []
    
    for _ in range(num_symbols):
        # Decide if this should be 1D (single value) or 2D (OHLCV)
        is_ohlcv = draw(st.booleans())
        
        if is_ohlcv:
            # Generate OHLCV data (5 columns)
            array = draw(arrays(
                dtype=np.float64,
                shape=(num_timestamps, 5),
                elements=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
            ))
            
            # Ensure OHLCV constraints: High >= max(Open, Close), Low <= min(Open, Close)
            for i in range(num_timestamps):
                open_price = array[i, 0]
                close_price = array[i, 3]
                
                # Set High to be at least max(Open, Close)
                array[i, 1] = max(array[i, 1], max(open_price, close_price))
                
                # Set Low to be at most min(Open, Close)
                array[i, 2] = min(array[i, 2], min(open_price, close_price))
                
                # Ensure Volume is positive integer-like
                array[i, 4] = abs(array[i, 4])
        else:
            # Generate 1D data (single values, e.g., close prices)
            array = draw(arrays(
                dtype=np.float64,
                shape=(num_timestamps,),
                elements=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
            ))
        
        data_arrays.append(array)
    
    return data_arrays


@st.composite
def valid_dataset_inputs(draw):
    """Generate valid inputs for WarspiteDataset construction."""
    symbols = draw(valid_symbols())
    timestamps = draw(valid_timestamps())
    data_arrays = draw(valid_data_arrays(len(symbols), len(timestamps)))
    
    # Optional metadata
    metadata = draw(st.one_of(
        st.none(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False))
        )
    ))
    
    return data_arrays, timestamps, symbols, metadata


class TestDatasetConstructionRobustness:
    """
    Property-based tests for Dataset Construction Robustness.
    
    **Feature: warspite-financial-library, Property 2: Dataset Construction Robustness**
    **Validates: Requirements 3.1**
    """
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=100, deadline=None)
    def test_dataset_construction_robustness(self, dataset_inputs):
        """
        Property 2: Dataset Construction Robustness
        
        For any list of valid numpy arrays with consistent dimensions, 
        WarspiteDataset construction should succeed and preserve all input data.
        
        **Feature: warspite-financial-library, Property 2: Dataset Construction Robustness**
        **Validates: Requirements 3.1**
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        
        # Construct the dataset
        if metadata is not None:
            dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        else:
            dataset = WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Property assertions: Construction robustness
        
        # 1. Dataset should be successfully created
        assert isinstance(dataset, WarspiteDataset), "Dataset construction should succeed"
        
        # 2. All input data should be preserved
        assert len(dataset.symbols) == len(symbols), "Number of symbols should be preserved"
        assert dataset.symbols == symbols, "Symbol names should be preserved"
        
        # 3. Timestamps should be preserved and properly formatted
        assert len(dataset.timestamps) == len(timestamps), "Number of timestamps should be preserved"
        assert np.array_equal(dataset.timestamps, timestamps), "Timestamps should be preserved"
        assert dataset.timestamps.dtype == np.dtype('datetime64[ns]'), "Timestamps should be datetime64[ns]"
        
        # 4. Data arrays should be preserved
        assert len(dataset.data_arrays) == len(data_arrays), "Number of data arrays should be preserved"
        for i, (original, stored) in enumerate(zip(data_arrays, dataset.data_arrays)):
            assert np.array_equal(original, stored), f"Data array {i} should be preserved"
            assert original.shape == stored.shape, f"Data array {i} shape should be preserved"
            assert original.dtype == stored.dtype, f"Data array {i} dtype should be preserved"
        
        # 5. Metadata should be preserved
        if metadata is not None:
            assert dataset.metadata == metadata, "Metadata should be preserved"
        else:
            assert dataset.metadata == {}, "Default metadata should be empty dict"
        
        # 6. Dataset length should match timestamp length
        assert len(dataset) == len(timestamps), "Dataset length should match timestamp length"
        
        # 7. Strategy results should be initially None
        assert dataset.strategy_results is None, "Strategy results should initially be None"
        
        # 8. Dataset should be properly representable
        repr_str = repr(dataset)
        assert isinstance(repr_str, str), "Dataset should have string representation"
        assert len(repr_str) > 0, "Dataset representation should not be empty"
        
        # 9. Properties should return copies, not references
        symbols_copy = dataset.symbols
        symbols_copy.append("NEW_SYMBOL")
        assert "NEW_SYMBOL" not in dataset.symbols, "Symbols property should return a copy"
        
        timestamps_copy = dataset.timestamps
        timestamps_copy[0] = np.datetime64('1900-01-01')
        assert not np.array_equal(timestamps_copy, dataset.timestamps), "Timestamps property should return a copy"
    
    @given(
        data_arrays=st.lists(
            arrays(dtype=np.float64, shape=(10,), elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False)),
            min_size=1,
            max_size=3
        ),
        timestamps=arrays(dtype='datetime64[ns]', shape=(10,)),
        symbols=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3, unique=True)
    )
    @settings(max_examples=50, deadline=None)
    def test_dataset_construction_with_mismatched_dimensions(self, data_arrays, timestamps, symbols):
        """
        Test that dataset construction properly validates dimension consistency.
        
        When dimensions don't match, construction should fail with clear error messages.
        """
        # Test case 1: Mismatched number of arrays and symbols
        if len(data_arrays) != len(symbols):
            with pytest.raises(ValueError, match="Number of data arrays must match number of symbols"):
                WarspiteDataset(data_arrays, timestamps, symbols)
        
        # Test case 2: Array length doesn't match timestamp length
        elif len(data_arrays) == len(symbols):
            # Modify one array to have wrong length
            if len(data_arrays) > 0:
                wrong_length_arrays = data_arrays.copy()
                wrong_length_arrays[0] = np.array([1.0, 2.0])  # Different length
                
                with pytest.raises(ValueError, match="Data array .* length .* doesn't match timestamps length"):
                    WarspiteDataset(wrong_length_arrays, timestamps, symbols)
    
    @given(
        timestamps=arrays(dtype='datetime64[ns]', shape=(5,)),
        symbols=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=1, unique=True)
    )
    @settings(max_examples=30)
    def test_dataset_construction_edge_cases(self, timestamps, symbols):
        """
        Test dataset construction with edge cases.
        """
        # Test with empty data arrays and empty symbols
        with pytest.raises(ValueError, match="At least one data array is required"):
            WarspiteDataset([], timestamps, [])
        
        # Test with single valid array
        single_array = [np.random.rand(len(timestamps))]
        dataset = WarspiteDataset(single_array, timestamps, symbols)
        assert len(dataset) == len(timestamps)
        assert len(dataset.symbols) == 1
    
    @given(valid_dataset_inputs())
    @settings(max_examples=50, deadline=None)
    def test_dataset_equality(self, dataset_inputs):
        """
        Test that dataset equality works correctly.
        
        Two datasets with identical data should be equal.
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        
        # Create two identical datasets
        dataset1 = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        dataset2 = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # They should be equal
        assert dataset1 == dataset2, "Identical datasets should be equal"
        
        # Test inequality with different symbols
        if len(symbols) > 1:
            different_symbols = symbols.copy()
            different_symbols[0] = different_symbols[0] + "_DIFFERENT"
            dataset3 = WarspiteDataset(data_arrays, timestamps, different_symbols, metadata)
            assert dataset1 != dataset3, "Datasets with different symbols should not be equal"
        
        # Test inequality with non-dataset object
        assert dataset1 != "not_a_dataset", "Dataset should not equal non-dataset objects"
        assert dataset1 != None, "Dataset should not equal None"


@st.composite
def valid_dataset_with_date_range(draw):
    """Generate a valid dataset with a valid date range for slicing."""
    # Create a dataset with at least 10 timestamps for meaningful slicing
    num_timestamps = draw(st.integers(10, 50))
    symbols = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=1,
        max_size=3,
        unique=True
    ))
    
    # Generate sequential timestamps (daily intervals)
    base_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 1, 1)
    ))
    
    timestamps = []
    for i in range(num_timestamps):
        timestamps.append(base_date + timedelta(days=i))
    
    timestamps = np.array(timestamps, dtype='datetime64[ns]')
    
    # Generate data arrays
    data_arrays = []
    for _ in range(len(symbols)):
        array = draw(arrays(
            dtype=np.float64,
            shape=(num_timestamps,),
            elements=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ))
        data_arrays.append(array)
    
    # Create dataset
    dataset = WarspiteDataset(data_arrays, timestamps, symbols)
    
    # Generate a valid date range within the dataset's time bounds
    min_date = pd.to_datetime(timestamps[0]).to_pydatetime()
    max_date = pd.to_datetime(timestamps[-1]).to_pydatetime()
    
    # Choose slice boundaries within the dataset range
    start_idx = draw(st.integers(0, num_timestamps - 2))
    end_idx = draw(st.integers(start_idx + 1, num_timestamps - 1))
    
    slice_start = pd.to_datetime(timestamps[start_idx]).to_pydatetime()
    slice_end = pd.to_datetime(timestamps[end_idx]).to_pydatetime()
    
    return dataset, slice_start, slice_end, start_idx, end_idx


class TestDatasetDateSlicingAccuracy:
    """
    Property-based tests for Dataset Date Slicing Accuracy.
    
    **Feature: warspite-financial-library, Property 3: Dataset Date Slicing Accuracy**
    **Validates: Requirements 3.3**
    """
    
    @given(dataset_with_range=valid_dataset_with_date_range())
    @settings(max_examples=100, deadline=None)
    def test_dataset_date_slicing_accuracy(self, dataset_with_range):
        """
        Property 3: Dataset Date Slicing Accuracy
        
        For any valid date range within a dataset's time bounds, the returned slice 
        should contain exactly the data points within that range.
        
        **Feature: warspite-financial-library, Property 3: Dataset Date Slicing Accuracy**
        **Validates: Requirements 3.3**
        """
        dataset, slice_start, slice_end, expected_start_idx, expected_end_idx = dataset_with_range
        
        # Perform the slice operation
        sliced_dataset = dataset.get_slice(slice_start, slice_end)
        
        # Property assertions: Date slicing accuracy
        
        # 1. Sliced dataset should be a WarspiteDataset
        assert isinstance(sliced_dataset, WarspiteDataset), "Slice should return WarspiteDataset"
        
        # 2. Sliced dataset should have the same symbols
        assert sliced_dataset.symbols == dataset.symbols, "Sliced dataset should preserve symbols"
        
        # 3. All timestamps in slice should be within the requested range
        slice_timestamps = sliced_dataset.timestamps
        slice_start_dt64 = np.datetime64(slice_start)
        slice_end_dt64 = np.datetime64(slice_end)
        
        assert len(slice_timestamps) > 0, "Slice should contain at least one timestamp"
        assert (slice_timestamps >= slice_start_dt64).all(), "All timestamps should be >= start_date"
        assert (slice_timestamps <= slice_end_dt64).all(), "All timestamps should be <= end_date"
        
        # 4. No timestamps outside the range should be included
        original_timestamps = dataset.timestamps
        expected_mask = (original_timestamps >= slice_start_dt64) & (original_timestamps <= slice_end_dt64)
        expected_timestamps = original_timestamps[expected_mask]
        
        assert np.array_equal(slice_timestamps, expected_timestamps), \
            "Slice should contain exactly the timestamps within the range"
        
        # 5. Data arrays should be sliced consistently
        assert len(sliced_dataset.data_arrays) == len(dataset.data_arrays), \
            "Number of data arrays should be preserved"
        
        for i, (original_array, sliced_array) in enumerate(zip(dataset.data_arrays, sliced_dataset.data_arrays)):
            expected_sliced_array = original_array[expected_mask]
            assert np.array_equal(sliced_array, expected_sliced_array), \
                f"Data array {i} should be sliced consistently with timestamps"
            assert sliced_array.shape[0] == len(slice_timestamps), \
                f"Data array {i} length should match timestamp length"
        
        # 6. Metadata should be preserved
        assert sliced_dataset.metadata == dataset.metadata, "Metadata should be preserved in slice"
        
        # 7. Strategy results should be sliced if they exist
        if dataset.strategy_results is not None:
            assert sliced_dataset.strategy_results is not None, \
                "Strategy results should be preserved if they exist"
            expected_strategy_slice = dataset.strategy_results[expected_mask]
            assert np.array_equal(sliced_dataset.strategy_results, expected_strategy_slice), \
                "Strategy results should be sliced consistently"
        else:
            assert sliced_dataset.strategy_results is None, \
                "Strategy results should remain None if not present"
        
        # 8. Slice length should match expected length
        expected_length = np.sum(expected_mask)
        assert len(sliced_dataset) == expected_length, \
            f"Slice length should be {expected_length}"
        
        # 9. Slice should be properly representable
        repr_str = repr(sliced_dataset)
        assert isinstance(repr_str, str), "Sliced dataset should have string representation"
        assert len(repr_str) > 0, "Sliced dataset representation should not be empty"
    
    @given(valid_dataset_inputs())
    @settings(max_examples=50, deadline=None)
    def test_invalid_date_range_handling(self, dataset_inputs):
        """
        Test that invalid date ranges are handled properly.
        
        Invalid ranges should raise ValueError with clear messages.
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Test case 1: start_date > end_date (not equal)
        base_date = datetime(2023, 1, 15)
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            dataset.get_slice(base_date + timedelta(days=1), base_date)  # Start after end
        
        # Test case 2: Date range completely outside dataset range
        min_dataset_date = pd.to_datetime(timestamps[0]).to_pydatetime()
        max_dataset_date = pd.to_datetime(timestamps[-1]).to_pydatetime()
        
        # Range before dataset
        early_start = min_dataset_date - timedelta(days=10)
        early_end = min_dataset_date - timedelta(days=1)
        
        with pytest.raises(ValueError, match="No data found in the specified date range"):
            dataset.get_slice(early_start, early_end)
        
        # Range after dataset
        late_start = max_dataset_date + timedelta(days=1)
        late_end = max_dataset_date + timedelta(days=10)
        
        with pytest.raises(ValueError, match="No data found in the specified date range"):
            dataset.get_slice(late_start, late_end)
    
    @given(dataset_with_range=valid_dataset_with_date_range())
    @settings(max_examples=30, deadline=None)
    def test_slice_with_strategy_results(self, dataset_with_range):
        """
        Test that slicing works correctly when strategy results are present.
        """
        dataset, slice_start, slice_end, _, _ = dataset_with_range
        
        # Add strategy results to the dataset
        strategy_results = np.random.rand(len(dataset))
        dataset.add_strategy_results(strategy_results)
        
        # Perform slice
        sliced_dataset = dataset.get_slice(slice_start, slice_end)
        
        # Strategy results should be sliced consistently
        assert sliced_dataset.strategy_results is not None, \
            "Strategy results should be preserved in slice"
        
        # Verify the sliced strategy results match the expected slice
        slice_start_dt64 = np.datetime64(slice_start)
        slice_end_dt64 = np.datetime64(slice_end)
        expected_mask = (dataset.timestamps >= slice_start_dt64) & (dataset.timestamps <= slice_end_dt64)
        expected_strategy_slice = strategy_results[expected_mask]
        
        assert np.array_equal(sliced_dataset.strategy_results, expected_strategy_slice), \
            "Strategy results should be sliced consistently with timestamps"
    
    @given(dataset_with_range=valid_dataset_with_date_range())
    @settings(max_examples=30, deadline=None)
    def test_slice_boundary_conditions(self, dataset_with_range):
        """
        Test slicing with boundary conditions (exact start/end dates).
        """
        dataset, _, _, _, _ = dataset_with_range
        
        # Test slicing with exact dataset boundaries
        min_date = pd.to_datetime(dataset.timestamps[0]).to_pydatetime()
        max_date = pd.to_datetime(dataset.timestamps[-1]).to_pydatetime()
        
        # Slice the entire dataset
        full_slice = dataset.get_slice(min_date, max_date)
        
        # Should be identical to original dataset
        assert len(full_slice) == len(dataset), "Full slice should have same length as original"
        assert np.array_equal(full_slice.timestamps, dataset.timestamps), \
            "Full slice should have identical timestamps"
        
        for i, (orig_array, slice_array) in enumerate(zip(dataset.data_arrays, full_slice.data_arrays)):
            assert np.array_equal(orig_array, slice_array), \
                f"Full slice data array {i} should be identical to original"
        
        # Test single-point slice (if dataset has multiple points)
        if len(dataset) > 1:
            single_date = pd.to_datetime(dataset.timestamps[1]).to_pydatetime()
            single_slice = dataset.get_slice(single_date, single_date)
            
            assert len(single_slice) == 1, "Single-point slice should have length 1"
            assert single_slice.timestamps[0] == dataset.timestamps[1], \
                "Single-point slice should contain the correct timestamp"


class TestDatasetSerializationRoundTrip:
    """
    Property-based tests for Dataset Serialization Round-Trip Integrity.
    
    **Feature: warspite-financial-library, Property 8: Serialization Round-Trip Integrity**
    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=100, deadline=None)
    def test_serialization_round_trip_csv(self, dataset_inputs):
        """
        Property 8: Serialization Round-Trip Integrity (CSV format)
        
        For any WarspiteDataset, serializing to CSV format then deserializing 
        should produce a dataset with identical data, metadata, and functionality.
        
        **Feature: warspite-financial-library, Property 8: Serialization Round-Trip Integrity**
        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        original_dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Add strategy results to test their preservation
        if len(original_dataset) > 0:
            strategy_results = np.random.rand(len(original_dataset))
            original_dataset.add_strategy_results(strategy_results)
        
        # Serialize to CSV
        csv_data = original_dataset.serialize('csv')
        assert isinstance(csv_data, str), "CSV serialization should return string"
        assert len(csv_data) > 0, "CSV data should not be empty"
        
        # Deserialize from CSV
        deserialized_dataset = WarspiteDataset.deserialize(csv_data, 'csv')
        
        # Property assertions: Round-trip integrity
        
        # 1. Deserialized dataset should be a WarspiteDataset
        assert isinstance(deserialized_dataset, WarspiteDataset), \
            "Deserialization should return WarspiteDataset"
        
        # 2. Basic properties should be preserved
        assert len(deserialized_dataset) == len(original_dataset), \
            "Dataset length should be preserved"
        
        # 3. Timestamps should be preserved (allowing for minor precision differences)
        assert len(deserialized_dataset.timestamps) == len(original_dataset.timestamps), \
            "Number of timestamps should be preserved"
        
        # Convert to pandas for easier comparison (CSV round-trip may affect precision)
        orig_df = original_dataset.to_dataframe()
        deser_df = deserialized_dataset.to_dataframe()
        
        # 4. Data should be numerically equivalent (allowing for floating point precision)
        for col in orig_df.columns:
            if col in deser_df.columns:
                if col == 'Strategy_Results':
                    # Strategy results should be preserved
                    assert np.allclose(orig_df[col].values, deser_df[col].values, rtol=1e-10), \
                        "Strategy results should be preserved in CSV round-trip"
                else:
                    # Financial data should be preserved
                    assert np.allclose(orig_df[col].values, deser_df[col].values, rtol=1e-10), \
                        f"Column {col} should be preserved in CSV round-trip"
        
        # 5. Symbols should be preserved (extracted from column names)
        original_symbols_set = set(original_dataset.symbols)
        deserialized_symbols_set = set(deserialized_dataset.symbols)
        assert original_symbols_set == deserialized_symbols_set, \
            "Symbols should be preserved in CSV round-trip"
        
        # 6. Functional equivalence - both datasets should behave the same
        assert len(deserialized_dataset.data_arrays) == len(original_dataset.data_arrays), \
            "Number of data arrays should be preserved"
        
        # 7. Dataset should be usable for further operations
        repr_str = repr(deserialized_dataset)
        assert isinstance(repr_str, str) and len(repr_str) > 0, \
            "Deserialized dataset should be properly representable"
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=100, deadline=None)
    def test_serialization_round_trip_json(self, dataset_inputs):
        """
        Property 8: Serialization Round-Trip Integrity (JSON format)
        
        For any WarspiteDataset, serializing to JSON format then deserializing 
        should produce a dataset with identical data, metadata, and functionality.
        
        **Feature: warspite-financial-library, Property 8: Serialization Round-Trip Integrity**
        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        original_dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Add strategy results to test their preservation
        if len(original_dataset) > 0:
            strategy_results = np.random.rand(len(original_dataset))
            original_dataset.add_strategy_results(strategy_results)
        
        # Serialize to JSON
        json_data = original_dataset.serialize('json')
        assert isinstance(json_data, str), "JSON serialization should return string"
        assert len(json_data) > 0, "JSON data should not be empty"
        
        # Deserialize from JSON
        deserialized_dataset = WarspiteDataset.deserialize(json_data, 'json')
        
        # Property assertions: Round-trip integrity
        
        # 1. Deserialized dataset should be a WarspiteDataset
        assert isinstance(deserialized_dataset, WarspiteDataset), \
            "Deserialization should return WarspiteDataset"
        
        # 2. Basic properties should be preserved exactly
        assert len(deserialized_dataset) == len(original_dataset), \
            "Dataset length should be preserved"
        assert deserialized_dataset.symbols == original_dataset.symbols, \
            "Symbols should be preserved exactly in JSON round-trip"
        
        # 3. Timestamps should be preserved exactly
        assert len(deserialized_dataset.timestamps) == len(original_dataset.timestamps), \
            "Number of timestamps should be preserved"
        assert np.array_equal(deserialized_dataset.timestamps, original_dataset.timestamps), \
            "Timestamps should be preserved exactly in JSON round-trip"
        
        # 4. Data arrays should be preserved exactly
        assert len(deserialized_dataset.data_arrays) == len(original_dataset.data_arrays), \
            "Number of data arrays should be preserved"
        
        for i, (orig_array, deser_array) in enumerate(zip(original_dataset.data_arrays, deserialized_dataset.data_arrays)):
            assert orig_array.shape == deser_array.shape, \
                f"Data array {i} shape should be preserved"
            assert np.allclose(orig_array, deser_array, rtol=1e-15), \
                f"Data array {i} should be preserved exactly in JSON round-trip"
        
        # 5. Metadata should be preserved exactly
        assert deserialized_dataset.metadata == original_dataset.metadata, \
            "Metadata should be preserved exactly in JSON round-trip"
        
        # 6. Strategy results should be preserved exactly
        if original_dataset.strategy_results is not None:
            assert deserialized_dataset.strategy_results is not None, \
                "Strategy results should be preserved if they exist"
            assert np.allclose(deserialized_dataset.strategy_results, original_dataset.strategy_results, rtol=1e-15), \
                "Strategy results should be preserved exactly in JSON round-trip"
        else:
            assert deserialized_dataset.strategy_results is None, \
                "Strategy results should remain None if not present"
        
        # 7. Datasets should be equal
        assert deserialized_dataset == original_dataset, \
            "Deserialized dataset should be equal to original"
        
        # 8. Dataset should be usable for further operations
        repr_str = repr(deserialized_dataset)
        assert isinstance(repr_str, str) and len(repr_str) > 0, \
            "Deserialized dataset should be properly representable"
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=100, deadline=None)
    def test_serialization_round_trip_pickle(self, dataset_inputs):
        """
        Property 8: Serialization Round-Trip Integrity (Pickle format)
        
        For any WarspiteDataset, serializing to pickle format then deserializing 
        should produce a dataset with identical data, metadata, and functionality.
        
        **Feature: warspite-financial-library, Property 8: Serialization Round-Trip Integrity**
        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        original_dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Add strategy results to test their preservation
        if len(original_dataset) > 0:
            strategy_results = np.random.rand(len(original_dataset))
            original_dataset.add_strategy_results(strategy_results)
        
        # Serialize to pickle
        pickle_data = original_dataset.serialize('pickle')
        assert isinstance(pickle_data, bytes), "Pickle serialization should return bytes"
        assert len(pickle_data) > 0, "Pickle data should not be empty"
        
        # Deserialize from pickle
        deserialized_dataset = WarspiteDataset.deserialize(pickle_data, 'pickle')
        
        # Property assertions: Round-trip integrity
        
        # 1. Deserialized dataset should be a WarspiteDataset
        assert isinstance(deserialized_dataset, WarspiteDataset), \
            "Deserialization should return WarspiteDataset"
        
        # 2. Datasets should be exactly equal (pickle preserves everything)
        assert deserialized_dataset == original_dataset, \
            "Pickle round-trip should preserve exact equality"
        
        # 3. All properties should be identical
        assert deserialized_dataset.symbols == original_dataset.symbols, \
            "Symbols should be preserved exactly"
        assert np.array_equal(deserialized_dataset.timestamps, original_dataset.timestamps), \
            "Timestamps should be preserved exactly"
        assert deserialized_dataset.metadata == original_dataset.metadata, \
            "Metadata should be preserved exactly"
        
        # 4. Data arrays should be identical
        assert len(deserialized_dataset.data_arrays) == len(original_dataset.data_arrays), \
            "Number of data arrays should be preserved"
        
        for i, (orig_array, deser_array) in enumerate(zip(original_dataset.data_arrays, deserialized_dataset.data_arrays)):
            assert np.array_equal(orig_array, deser_array), \
                f"Data array {i} should be preserved exactly in pickle round-trip"
            assert orig_array.dtype == deser_array.dtype, \
                f"Data array {i} dtype should be preserved"
        
        # 5. Strategy results should be identical
        if original_dataset.strategy_results is not None:
            assert deserialized_dataset.strategy_results is not None, \
                "Strategy results should be preserved"
            assert np.array_equal(deserialized_dataset.strategy_results, original_dataset.strategy_results), \
                "Strategy results should be preserved exactly in pickle round-trip"
        else:
            assert deserialized_dataset.strategy_results is None, \
                "Strategy results should remain None"
        
        # 6. Dataset should be usable for further operations
        repr_str = repr(deserialized_dataset)
        assert isinstance(repr_str, str) and len(repr_str) > 0, \
            "Deserialized dataset should be properly representable"
        
        # 7. Functional operations should work identically
        df_orig = original_dataset.to_dataframe()
        df_deser = deserialized_dataset.to_dataframe()
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df_orig, df_deser, check_dtype=True, check_index_type=True)
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=50, deadline=None)
    def test_serialization_format_validation(self, dataset_inputs):
        """
        Test that serialization properly validates format parameters.
        """
        data_arrays, timestamps, symbols, metadata = dataset_inputs
        dataset = WarspiteDataset(data_arrays, timestamps, symbols, metadata)
        
        # Test invalid format for serialization
        with pytest.raises(ValueError, match="Unsupported format.*Use 'csv', 'json', or 'pickle'"):
            dataset.serialize('invalid_format')
        
        # Test invalid format for deserialization
        with pytest.raises(ValueError, match="Unsupported format.*Use 'csv', 'json', or 'pickle'"):
            WarspiteDataset.deserialize("dummy_data", 'invalid_format')
        
        # Test case sensitivity
        csv_data = dataset.serialize('CSV')  # Should work (case insensitive)
        assert isinstance(csv_data, str)
        
        json_data = dataset.serialize('JSON')  # Should work (case insensitive)
        assert isinstance(json_data, str)
        
        pickle_data = dataset.serialize('PICKLE')  # Should work (case insensitive)
        assert isinstance(pickle_data, bytes)
    
    @given(dataset_inputs=valid_dataset_inputs())
    @settings(max_examples=30, deadline=None)
    def test_serialization_with_complex_metadata(self, dataset_inputs):
        """
        Test serialization with complex metadata structures.
        """
        data_arrays, timestamps, symbols, _ = dataset_inputs
        
        # Create complex metadata
        complex_metadata = {
            'nested_dict': {'key1': 'value1', 'key2': 42},
            'list_data': [1, 2, 3, 'string'],
            'numeric_data': 3.14159,
            'boolean_data': True,
            'none_data': None
        }
        
        dataset = WarspiteDataset(data_arrays, timestamps, symbols, complex_metadata)
        
        # Test JSON round-trip with complex metadata
        json_data = dataset.serialize('json')
        deserialized_dataset = WarspiteDataset.deserialize(json_data, 'json')
        
        assert deserialized_dataset.metadata == complex_metadata, \
            "Complex metadata should be preserved in JSON round-trip"
        
        # Test pickle round-trip with complex metadata
        pickle_data = dataset.serialize('pickle')
        deserialized_dataset = WarspiteDataset.deserialize(pickle_data, 'pickle')
        
        assert deserialized_dataset.metadata == complex_metadata, \
            "Complex metadata should be preserved in pickle round-trip"