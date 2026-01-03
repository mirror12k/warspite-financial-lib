"""
Helper functions for visualization operations.

This module provides convenience functions for common visualization patterns.
"""

from .renderers import (
    MatplotlibRenderer, 
    ASCIIRenderer, 
    PDFRenderer, 
    CSVRenderer
)
from ..utils.exceptions import VisualizationError


def create_visualization(dataset, renderer_type='matplotlib', **kwargs):
    """
    Convenience function to create visualizations.
    
    Args:
        dataset: WarspiteDataset to visualize
        renderer_type: Type of renderer ('matplotlib', 'ascii', 'pdf', 'csv')
        **kwargs: Additional arguments for the render method (not constructor)
        
    Returns:
        Rendered visualization output
        
    Example:
        >>> from warspite_financial.visualization.helpers import create_visualization
        >>> result = create_visualization(dataset, renderer_type='ascii')
        >>> print(result)
    """
    renderer_map = {
        'matplotlib': MatplotlibRenderer,
        'ascii': ASCIIRenderer,
        'pdf': PDFRenderer,
        'csv': CSVRenderer
    }
    
    if renderer_type not in renderer_map:
        raise VisualizationError(f"Unknown renderer type: {renderer_type}. Available: {list(renderer_map.keys())}")
    
    try:
        renderer_class = renderer_map[renderer_type]
        renderer = renderer_class(dataset)  # Only pass dataset to constructor
        return renderer.render(**kwargs)     # Pass kwargs to render method
    except Exception as e:
        raise VisualizationError(f"Failed to create visualization: {str(e)}") from e


__all__ = ['create_visualization']