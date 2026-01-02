"""
PDF renderer for warspite_financial visualization.

This module provides PDF export functionality for matplotlib-based visualizations.
"""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .matplotlib_renderer import MatplotlibRenderer


class PDFRenderer(MatplotlibRenderer):
    """
    PDF export renderer that extends MatplotlibRenderer.
    
    This renderer creates PDF documents containing financial visualizations
    with support for multi-page documents and custom formatting.
    """
    
    def __init__(self, dataset):
        """
        Initialize PDF renderer.
        
        Args:
            dataset: WarspiteDataset instance to render
        """
        super().__init__(dataset)
        
        # PDF-specific style options
        self._style_options.update({
            'pdf_title': None,
            'pdf_author': 'warspite_financial',
            'pdf_subject': 'Financial Data Analysis',
            'pdf_keywords': 'finance,trading,analysis',
            'multi_page': False,
            'page_size': 'letter',  # 'letter', 'a4', 'legal'
            'orientation': 'portrait'  # 'portrait', 'landscape'
        })
    
    def get_supported_formats(self) -> list:
        """Get supported output formats."""
        return ['pdf']
    
    def render(self, **kwargs) -> Figure:
        """
        Render the dataset as a matplotlib figure optimized for PDF output.
        
        Args:
            **kwargs: Rendering options including PDF-specific options
                
        Returns:
            matplotlib Figure object optimized for PDF
        """
        # Merge options
        options = {**self._style_options, **kwargs}
        
        # Set PDF-optimized figure parameters
        if options.get('page_size') == 'a4':
            if options.get('orientation') == 'landscape':
                figsize = (11.69, 8.27)  # A4 landscape
            else:
                figsize = (8.27, 11.69)  # A4 portrait
        elif options.get('page_size') == 'legal':
            if options.get('orientation') == 'landscape':
                figsize = (14, 8.5)  # Legal landscape
            else:
                figsize = (8.5, 14)  # Legal portrait
        else:  # letter
            if options.get('orientation') == 'landscape':
                figsize = (11, 8.5)  # Letter landscape
            else:
                figsize = (8.5, 11)  # Letter portrait
        
        options['figsize'] = figsize
        options['dpi'] = 300  # High DPI for PDF
        
        # Use parent class rendering with PDF optimizations
        return super().render(**options)
    
    def _save_output(self, rendered_output: Figure, filepath: str, **kwargs) -> None:
        """Save matplotlib figure to PDF file."""
        options = {**self._style_options, **kwargs}
        
        # PDF metadata
        metadata = {
            'Title': options.get('pdf_title', 'Financial Analysis Report'),
            'Author': options.get('pdf_author', 'warspite_financial'),
            'Subject': options.get('pdf_subject', 'Financial Data Analysis'),
            'Keywords': options.get('pdf_keywords', 'finance,trading,analysis'),
            'Creator': 'warspite_financial library'
        }
        
        # Save options
        save_options = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'metadata': metadata
        }
        
        rendered_output.savefig(filepath, format='pdf', **save_options)
    
    def create_multi_page_report(self, emulator=None, **kwargs) -> str:
        """
        Create a comprehensive multi-page PDF report.
        
        Args:
            emulator: Optional WarspiteTradingEmulator for portfolio analysis
            **kwargs: Additional rendering options
            
        Returns:
            Path to the created PDF file
        """
        options = {**self._style_options, **kwargs}
        filepath = kwargs.get('filepath', 'financial_report.pdf')
        
        with PdfPages(filepath) as pdf:
            # Page 1: Price Analysis
            fig1 = self.render(chart_type='price', show_volume=True, **options)
            fig1.suptitle('Price and Volume Analysis', fontsize=16, fontweight='bold')
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            # Page 2: Strategy Analysis (if strategy results available)
            if self._dataset.strategy_results is not None:
                fig2 = self.render(chart_type='strategy', **options)
                fig2.suptitle('Strategy Analysis', fontsize=16, fontweight='bold')
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            
            # Page 3: Portfolio Performance (if emulator provided)
            if emulator is not None and len(emulator.portfolio_history) >= 2:
                fig3 = self.render(chart_type='portfolio', emulator=emulator, **options)
                fig3.suptitle('Portfolio Performance', fontsize=16, fontweight='bold')
                pdf.savefig(fig3, bbox_inches='tight')
                plt.close(fig3)
                
                # Page 4: Performance Summary
                fig4 = self.create_performance_summary(emulator, **options)
                pdf.savefig(fig4, bbox_inches='tight')
                plt.close(fig4)
            
            # Set PDF metadata
            pdf_info = pdf.infodict()
            pdf_info['Title'] = options.get('pdf_title', 'Financial Analysis Report')
            pdf_info['Author'] = options.get('pdf_author', 'warspite_financial')
            pdf_info['Subject'] = options.get('pdf_subject', 'Financial Data Analysis')
            pdf_info['Keywords'] = options.get('pdf_keywords', 'finance,trading,analysis')
            pdf_info['Creator'] = 'warspite_financial library'
        
        return filepath
    
    def create_summary_page(self, emulator=None, **kwargs) -> Figure:
        """
        Create a summary page with key metrics and charts.
        
        Args:
            emulator: Optional WarspiteTradingEmulator for metrics
            **kwargs: Additional rendering options
            
        Returns:
            matplotlib Figure with summary information
        """
        options = {**self._style_options, **kwargs}
        
        # Create figure with custom layout for summary
        fig = plt.figure(figsize=options['figsize'], dpi=options['dpi'])
        
        # Add title
        fig.suptitle('Financial Analysis Summary', fontsize=20, fontweight='bold', y=0.95)
        
        # Create text summary
        summary_text = []
        summary_text.append(f"Dataset: {len(self._dataset.symbols)} symbols, {len(self._dataset)} data points")
        
        if len(self._dataset) > 0:
            start_date = self._dataset.timestamps[0]
            end_date = self._dataset.timestamps[-1]
            summary_text.append(f"Period: {start_date} to {end_date}")
        
        if self._dataset.strategy_results is not None:
            summary_text.append("Strategy results: Available")
        
        if emulator is not None:
            metrics = emulator.get_performance_metrics()
            summary_text.append(f"Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            summary_text.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            summary_text.append(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        
        # Add text to figure
        text_ax = fig.add_subplot(1, 1, 1)
        text_ax.axis('off')
        
        for i, text in enumerate(summary_text):
            text_ax.text(0.1, 0.8 - i*0.1, text, fontsize=14, 
                        transform=text_ax.transAxes, fontweight='bold')
        
        # Add timestamp
        from datetime import datetime
        fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='right', va='bottom', fontsize=10, alpha=0.7)
        
        return fig