"""
Regression tests for __init__.py export issues.

This test module verifies that:
1. All strategy classes are properly exported from the main package
2. Functions are properly organized in separate modules rather than __init__.py
3. Import patterns are consistent and complete
"""

import pytest
import importlib
import inspect
from warspite_financial import strategies


class TestMainPackageExports:
    """Test that all expected classes are exported from the main warspite_financial package."""
    
    def test_all_strategies_exported_from_main_package(self):
        """Test that all strategy classes can be imported directly from warspite_financial."""
        import warspite_financial
        
        # Get all strategy classes from the strategies module
        strategy_classes = []
        for name in dir(strategies):
            obj = getattr(strategies, name)
            if (inspect.isclass(obj) and 
                hasattr(obj, '__module__') and 
                obj.__module__.startswith('warspite_financial.strategies') and
                name != 'BaseStrategy'):  # BaseStrategy is the base class
                strategy_classes.append(name)
        
        # Verify each strategy class is available from main package
        missing_exports = []
        for strategy_name in strategy_classes:
            if not hasattr(warspite_financial, strategy_name):
                missing_exports.append(strategy_name)
        
        assert not missing_exports, f"Strategy classes not exported from main package: {missing_exports}"
    
    def test_specific_missing_strategies(self):
        """Test specific strategies that were previously missing from main exports."""
        import warspite_financial
        
        # These strategies should now be available from main package
        previously_missing_strategies = ['BuyAndHoldStrategy', 'ShortStrategy']
        
        for strategy_name in previously_missing_strategies:
            # Should be available in strategies submodule
            assert hasattr(strategies, strategy_name), f"{strategy_name} not found in strategies module"
            
            # Should now be available from main package (this should pass after fix)
            assert hasattr(warspite_financial, strategy_name), f"{strategy_name} not exported from main package"


class TestInitFileOrganization:
    """Test that __init__.py files are properly organized."""
    
    def test_convenience_functions_moved_to_helpers(self):
        """Test that convenience functions have been moved to appropriate helper modules."""
        import warspite_financial
        
        # These functions should now be imported from helper modules, not defined in __init__.py
        convenience_functions = [
            ('create_dataset_from_provider', 'warspite_financial.datasets.helpers'),
            ('run_strategy_backtest', 'warspite_financial.emulator.helpers'), 
            ('create_visualization', 'warspite_financial.visualization.helpers')
        ]
        
        for func_name, expected_module in convenience_functions:
            assert hasattr(warspite_financial, func_name), f"Function {func_name} not found"
            
            # Check if function is imported from the correct helper module
            func = getattr(warspite_financial, func_name)
            assert func.__module__ == expected_module, f"Function {func_name} should be from {expected_module}, got {func.__module__}"
    
    def test_init_file_length(self):
        """Test that main __init__.py is now concise after refactoring."""
        import warspite_financial
        import inspect
        
        # Get the source file path
        init_file = inspect.getfile(warspite_financial)
        
        with open(init_file, 'r') as f:
            lines = f.readlines()
        
        # After refactoring, __init__.py should be much shorter (under 50 lines)
        assert len(lines) < 50, f"Main __init__.py should be concise after refactoring, got {len(lines)} lines"


class TestDynamicExportSystem:
    """Test the new dynamic export system."""
    
    def test_dynamic_exports_work(self):
        """Test that the dynamic export system correctly exports all submodule items."""
        import warspite_financial
        
        # Test that __all__ is populated dynamically
        assert len(warspite_financial.__all__) > 30, "Dynamic __all__ should contain many exports"
        
        # Test that all items in __all__ are actually available
        missing_items = []
        for item_name in warspite_financial.__all__:
            if not hasattr(warspite_financial, item_name):
                missing_items.append(item_name)
        
        assert not missing_items, f"Items in __all__ but not available: {missing_items}"
    
    def test_new_strategy_automatically_exported(self):
        """Test that new strategies would be automatically exported without manual updates."""
        # This test verifies that the dynamic system would pick up new strategies
        from warspite_financial import strategies
        
        # Get all strategy classes from strategies module
        all_strategies = [name for name in dir(strategies) 
                         if name.endswith('Strategy') and not name.startswith('Base')]
        
        import warspite_financial
        
        # All strategies should be in main package __all__
        missing_strategies = []
        for strategy_name in all_strategies:
            if strategy_name not in warspite_financial.__all__:
                missing_strategies.append(strategy_name)
        
        assert not missing_strategies, f"Strategies not in main __all__: {missing_strategies}"
        
        # All strategies should be available from main package
        unavailable_strategies = []
        for strategy_name in all_strategies:
            if not hasattr(warspite_financial, strategy_name):
                unavailable_strategies.append(strategy_name)
        
        assert not unavailable_strategies, f"Strategies not available from main package: {unavailable_strategies}"


class TestImportPatterns:
    """Test import pattern consistency."""
    
    def test_blanket_imports_usage(self):
        """Test which modules use blanket imports (should be minimized)."""
        # Check examples module
        from warspite_financial import examples
        
        # This will pass initially but documents the blanket import usage
        assert hasattr(examples, 'basic_backtest_example'), "Examples module should export functions"
        
        # Check utils module  
        from warspite_financial import utils
        assert hasattr(utils, 'WarspiteError'), "Utils module should export exceptions"


class TestStrategyExportRegression:
    """Specific regression test for the SMAStrategy export issue mentioned."""
    
    def test_sma_strategy_export(self):
        """Test that SMAStrategy is properly exported (this should pass)."""
        import warspite_financial
        
        # SMAStrategy should be available from main package
        assert hasattr(warspite_financial, 'SMAStrategy'), "SMAStrategy not exported from main package"
        
        # Should be the same class as in strategies module
        from warspite_financial.strategies import SMAStrategy as StrategiesSMA
        MainSMA = warspite_financial.SMAStrategy
        
        assert MainSMA is StrategiesSMA, "SMAStrategy from main package should be same as from strategies module"
    
    def test_new_strategy_would_be_missing(self):
        """Test that demonstrates how new strategies might not be automatically exported."""
        # This test simulates what would happen if a new strategy was added
        # but not added to the main __init__.py exports
        
        from warspite_financial import strategies
        
        # Get all strategy classes from strategies module
        all_strategies = [name for name in dir(strategies) 
                         if name.endswith('Strategy') and not name.startswith('Base')]
        
        import warspite_financial
        main_strategies = [name for name in dir(warspite_financial)
                          if name.endswith('Strategy') and not name.startswith('Base')]
        
        # Find strategies that are in the strategies module but not in main package
        missing_from_main = set(all_strategies) - set(main_strategies)
        
        # This will show which strategies are missing from main exports
        print(f"Strategies in strategies module: {sorted(all_strategies)}")
        print(f"Strategies in main package: {sorted(main_strategies)}")
        print(f"Missing from main package: {sorted(missing_from_main)}")
        
        # This assertion will fail, demonstrating the issue
        assert not missing_from_main, f"Strategies missing from main package exports: {missing_from_main}"