"""
Base renderer class for warspite_financial visualization.

This module contains the abstract base class for all dataset renderers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
import os


class WarspiteDatasetRenderer(ABC):
    """
    Abstract base class for dataset renderers.
    
    This class defines the interface that all dataset renderers must implement
    for visualizing WarspiteDataset objects in various formats.
    """
    
    def __init__(self, dataset):
        """
        Initialize renderer with a dataset.
        
        Args:
            dataset: WarspiteDataset instance to render
            
        Raises:
            ValueError: If dataset is invalid
        """
        # Import here to avoid circular imports
        from ...datasets.dataset import WarspiteDataset
        
        if not isinstance(dataset, WarspiteDataset):
            raise ValueError("Dataset must be a WarspiteDataset instance")
        
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        self._dataset = dataset
        self._style_options = {}
    
    @property
    def dataset(self):
        """Get the dataset being rendered."""
        return self._dataset
    
    @property
    def style_options(self) -> Dict[str, Any]:
        """Get current style options."""
        return self._style_options.copy()
    
    def set_style_options(self, **kwargs) -> None:
        """
        Set style options for rendering.
        
        Args:
            **kwargs: Style options specific to the renderer implementation
        """
        self._style_options.update(kwargs)
    
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """
        Render the dataset.
        
        Args:
            **kwargs: Renderer-specific options
            
        Returns:
            Rendered output (format depends on renderer implementation)
        """
        pass
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the rendered output to a file.
        
        Args:
            filepath: Path where to save the rendered output
            **kwargs: Additional save options
            
        Raises:
            ValueError: If filepath is invalid
            IOError: If save operation fails
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Render and save
        try:
            rendered_output = self.render(**kwargs)
            self._save_output(rendered_output, filepath, **kwargs)
        except Exception as e:
            raise IOError(f"Failed to save rendered output: {e}")
    
    @abstractmethod
    def _save_output(self, rendered_output: Any, filepath: str, **kwargs) -> None:
        """
        Save the rendered output to a file.
        
        This method must be implemented by subclasses to handle the specific
        format of their rendered output.
        
        Args:
            rendered_output: The output from render() method
            filepath: Path where to save the output
            **kwargs: Additional save options
        """
        pass
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported output formats for this renderer.
        
        Returns:
            List of supported format strings
        """
        return ['default']
    
    def validate_dataset_for_rendering(self) -> bool:
        """
        Validate that the dataset is suitable for rendering.
        
        Returns:
            True if dataset can be rendered, False otherwise
        """
        try:
            # Basic validation
            if len(self._dataset) == 0:
                return False
            
            if not self._dataset.symbols:
                return False
            
            # Check that we have valid data arrays
            for data_array in self._dataset.data_arrays:
                if data_array is None or len(data_array) == 0:
                    return False
            
            # Check timestamps
            if len(self._dataset.timestamps) != len(self._dataset.data_arrays[0]):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset for rendering purposes.
        
        Returns:
            Dictionary containing dataset summary information
        """
        return {
            'symbols': self._dataset.symbols,
            'length': len(self._dataset),
            'start_date': self._dataset.timestamps[0] if len(self._dataset) > 0 else None,
            'end_date': self._dataset.timestamps[-1] if len(self._dataset) > 0 else None,
            'has_strategy_results': self._dataset.strategy_results is not None,
            'metadata': self._dataset.metadata
        }