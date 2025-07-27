"""
Factory for creating loss functions based on configuration.
"""
from typing import Dict, Any
import torch.nn as nn

from .base_losses import BaseLoss, WeightedCombinedLoss
from .stage1_losses import DETRCombinedLoss, ClassificationLoss, BBoxRegressionLoss
from .stage2_losses import DiceLoss, FocalLoss, BoundaryLoss, CombinedSegmentationLoss


class LossFactory:
    """Factory class for creating loss functions."""
    
    # Registry of available losses
    LOSS_REGISTRY = {
        # Stage 1 losses
        'detr_combined': DETRCombinedLoss,
        'classification': ClassificationLoss,
        'bbox_regression': BBoxRegressionLoss,
        
        # Stage 2 losses  
        'dice': DiceLoss,
        'focal': FocalLoss,
        'boundary': BoundaryLoss,
        'combined_segmentation': CombinedSegmentationLoss,
    }
    
    @classmethod
    def create_loss(cls, loss_config: Dict[str, Any]) -> BaseLoss:
        """
        Create a loss function from configuration.
        
        Args:
            loss_config: Dictionary containing loss configuration
                - type: Loss type name
                - params: Parameters for the loss function
        
        Returns:
            Configured loss function
        """
        loss_type = loss_config.get('type')
        loss_params = loss_config.get('params', {})
        
        if loss_type not in cls.LOSS_REGISTRY:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(cls.LOSS_REGISTRY.keys())}")
        
        loss_class = cls.LOSS_REGISTRY[loss_type]
        return loss_class(**loss_params)
    
    @classmethod
    def create_stage1_loss(cls, config) -> BaseLoss:
        """Create Stage 1 (DETR) loss."""
        return DETRCombinedLoss(
            num_classes=config.num_classes,
            ce_weight=config.ce_loss_weight,
            bbox_weight=config.bbox_loss_weight,
            giou_weight=config.giou_loss_weight,
            focal_alpha=config.focal_alpha
        )
    
    @classmethod
    def create_stage2_loss(cls, config) -> BaseLoss:
        """Create Stage 2 (segmentation) loss."""
        return CombinedSegmentationLoss(config)
    
    @classmethod
    def create_combined_loss(cls, losses_config: Dict[str, Dict[str, Any]]) -> WeightedCombinedLoss:
        """
        Create a weighted combination of multiple losses.
        
        Args:
            losses_config: Dictionary of loss configurations
                Format: {
                    'loss_name': {
                        'type': 'loss_type',
                        'weight': weight_value,
                        'params': {...}
                    }
                }
        
        Returns:
            WeightedCombinedLoss instance
        """
        losses_dict = {}
        
        for name, config in losses_config.items():
            loss_fn = cls.create_loss({
                'type': config['type'],
                'params': config.get('params', {})
            })
            weight = config.get('weight', 1.0)
            losses_dict[name] = (loss_fn, weight)
        
        return WeightedCombinedLoss(losses_dict)
    
    @classmethod
    def register_loss(cls, name: str, loss_class: type):
        """Register a new loss function."""
        if not issubclass(loss_class, (BaseLoss, nn.Module)):
            raise ValueError("Loss class must inherit from BaseLoss or nn.Module")
        
        cls.LOSS_REGISTRY[name] = loss_class
    
    @classmethod
    def list_available_losses(cls):
        """List all available loss types."""
        return list(cls.LOSS_REGISTRY.keys())


def create_loss_from_config(config) -> Dict[str, BaseLoss]:
    """
    Create loss functions for both stages from configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        Dictionary with 'stage1' and 'stage2' loss functions
    """
    factory = LossFactory()
    
    return {
        'stage1': factory.create_stage1_loss(config),
        'stage2': factory.create_stage2_loss(config)
    }