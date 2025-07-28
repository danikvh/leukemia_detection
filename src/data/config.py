from typing import Dict, Any
from common.base_config import BaseConfig

"""
Data Processing Configuration

Specialized configuration for data processing tasks.
"""

class DataProcessingConfig(BaseConfig):
    """Configuration for data processing tasks."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        # Start with common config
        return {
            # Core processing
            'patch_size': 512,
            'stride': 384,
            'border_margin_ratio': 0.1,
            'overlap_threshold': 0.85,
            'min_polygon_area': 10.0,
            'white_threshold': 240,
            'variance_threshold': 10.0,
            
            # Hardware
            'device': 'auto',
            'num_workers': None,
            'batch_size': 8,
            
            # Paths
            'data_dir': './data',
            'output_dir': './output',
            'temp_dir': './temp',
            'model_path': None,
            
            # Processing options
            'use_multiprocessing': True,
            'memory_limit_gb': None,
            'enable_validation': True,
            'strict_validation': False,
            
            # Model inference
            'bbox_threshold': 0.19,
            'generate_masks': False,
            'use_rois': False,
            'save_masks': True,
            
            # Visualization
            'show_results': False,
            'save_overlays': True,
            'overlay_alpha': 0.3,
            
            # Logging
            'log_level': 'INFO',
            'log_file': None,
            
            # File patterns
            'svs_pattern': '*.svs',
            'geojson_pattern': '*.geojson',
            
            # QuPath
            'qupath_project_path': None,
            
            # Label mappings
            'label_mappings': {
                'positivo': 'positive',
                'negativo': 'negative'
            }
        }