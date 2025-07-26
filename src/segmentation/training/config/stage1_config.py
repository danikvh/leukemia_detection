"""
Stage 1 (Detection) training configuration.
"""

from typing import Dict, Any, List
from segmentation.training.config.base_config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    """Loss configuration for Stage 1 (detection)."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Extend parent default config with Stage 1 specific values."""
        config = super()._get_default_config()  # Get defaults from BaseTrainingConfig
        config.update({
            # Loss
            "ce_loss_weight": 1.0,
            "bbox_loss_weight": 5.0,
            "giou_loss_weight": 2.0,
            "focal_alpha": 0.25,
            "num_classes": 2,

            # Model
            "backbone_type": "cellfinder",  # Options: cellfinder, image_encoder, all_backbones, head_only
            "freeze_backbone": False,
            "transfer_weights": False,
            "backbone_lr_ratio": 0.1,  # Ratio of backbone LR to head LR

            # Visualization
            "visualize_predictions": True,
            "visualization_frequency": 1,
            "max_visualizations_per_epoch": 4,
        })
        return config
    
    def _validate_config(self) -> None:
        """Extend parent validation with Stage 1 specific checks."""
        super()._validate_config()  # Run base validations

        if self.get("ce_loss_weight") < 0:
            raise ValueError("CE loss weight must be non-negative")
        
        if self.get("bbox_loss_weight") < 0:
            raise ValueError("Bbox loss weight must be non-negative")
        
        if self.get("giou_loss_weight") < 0:
            raise ValueError("GIoU loss weight must be non-negative")
        
        if not 0 <= self.get("focal_alpha") <= 1:
            raise ValueError("Focal alpha must be between 0 and 1")
        
        if self.get("num_classes") <= 0:
            raise ValueError("Number of classes must be positive")
        
        valid_backbones = ["cellfinder", "image_encoder", "all_backbones", "head_only"]
        if self.get("backbone_type") not in valid_backbones:
            raise ValueError(f"Backbone type must be one of {valid_backbones}")
        
        if not 0 < self.get("backbone_lr_ratio") <= 1:
            raise ValueError("Backbone LR ratio must be between 0 and 1")

    def get_weight_dict(self) -> Dict[str, float]:
        """Get loss weight dictionary for criterion."""
        return {
            'loss_ce': self.get("loss.ce_loss_weight"),
            'loss_bbox': self.get("loss.bbox_loss_weight"),
            'loss_giou': self.get("loss.giou_loss_weight")
        }
    
    def get_optimizer_param_groups(self, model_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer with different learning rates."""
        param_groups = []
        
        # Head parameters (transformer)
        if 'head_params' in model_components:
            param_groups.append({
                "params": model_components['head_params'],
                "lr": self.get("optimizer.lr"),
                "name": "head"
            })
        
        # Backbone parameters (if training backbone)
        if not self.get("model.freeze_backbone") and 'backbone_params' in model_components:
            backbone_lr = self.get("optimizer.lr") * self.get("model.backbone_lr_ratio")
            param_groups.append({
                "params": model_components['backbone_params'],
                "lr": backbone_lr,
                "name": "backbone"
            })
        
        return param_groups
    
    def should_visualize(self, epoch: int) -> bool:
        """Check if should visualize predictions for current epoch."""
        return (
            self.get("visualize_predictions") and 
            epoch % self.get("visualization_frequency") == 0
        )