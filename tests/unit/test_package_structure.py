"""
Unit tests for package structure and imports.
"""

import pytest


class TestPackageStructure:
    """Test that the package structure is correctly set up."""
    
    def test_main_package_import(self):
        """Test that the main package can be imported."""
        import warspite_financial
        assert hasattr(warspite_financial, '__version__')
        assert warspite_financial.__version__ == "0.1.0"
    
    def test_base_provider_import(self):
        """Test that BaseProvider can be imported."""
        from warspite_financial.providers import BaseProvider
        assert BaseProvider is not None
    
    def test_trading_provider_import(self):
        """Test that TradingProvider can be imported."""
        from warspite_financial.providers import TradingProvider
        assert TradingProvider is not None
    
    def test_base_strategy_import(self):
        """Test that BaseStrategy can be imported."""
        from warspite_financial.strategies import BaseStrategy
        assert BaseStrategy is not None
    
    def test_dataset_import(self):
        """Test that WarspiteDataset can be imported."""
        from warspite_financial.datasets import WarspiteDataset
        assert WarspiteDataset is not None
    
    def test_emulator_import(self):
        """Test that WarspiteTradingEmulator can be imported."""
        from warspite_financial.emulator import WarspiteTradingEmulator
        assert WarspiteTradingEmulator is not None
    
    def test_main_package_exports(self):
        """Test that main package exports expected classes."""
        import warspite_financial
        
        expected_exports = [
            "BaseProvider",
            "TradingProvider", 
            "BaseStrategy",
            "WarspiteDataset",
            "WarspiteTradingEmulator"
        ]
        
        for export in expected_exports:
            assert hasattr(warspite_financial, export), f"Missing export: {export}"