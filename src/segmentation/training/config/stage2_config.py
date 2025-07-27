"""
Stage 2 (Segmentation) training configuration.
"""

from typing import Dict, Any, Optional
from segmentation.training.config.base_config import BaseTrainingConfig


class Stage2Config(BaseTrainingConfig):
    """Loss configuration for Stage 2 (segmentation)."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Extend parent default config with Stage 2 specific values."""
        config = super()._get_default_config()  # Get defaults from BaseTrainingConfig
        config.update({
            # Loss
            "focal_loss_weight": 1.0,
            "dice_loss_weight": 1.0,
            "boundary_loss_weight": 1.0,
            "use_boundary_loss": False,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "dice_smooth": 1.0,

            # Model
            "freeze_image_encoder": True,
            "freeze_prompt_encoder": True,
            "freeze_mask_decoder": True,
            "train_neck_only": True,

            # Scheduler
            "use_scheduler": False,

            # OHEM
            "online_hard_negative_mining": True,
            "online_hard_negative_mining_weighted": True,
            "ohem_fraction": 0.1,
            "ohem_hard_weight_min": 1.2,
            "ohem_hard_weight_max": 2.5,
            "ohem_hard_weight": 2.0,

            # Visualization
            "visualize_masks": True,
            "visualization_frequency": 2,
            "max_visualizations_per_epoch": 4,
        })
        return config

    def _validate_config(self) -> None:
        """Extend parent validation with Stage 2 specific checks."""
        super()._validate_config()  # Run base validations

        if self.get("focal_loss_weight") < 0:
            raise ValueError("Focal loss weight must be non-negative")
        
        if self.get("dice_loss_weight") < 0:
            raise ValueError("Dice loss weight must be non-negative")
        
        if self.get("boundary_loss_weight") < 0:
            raise ValueError("Boundary loss weight must be non-negative")
        
        if not 0 <= self.get("focal_alpha") <= 1:
            raise ValueError("Focal alpha must be between 0 and 1")
        
        if self.get("focal_gamma") < 0:
            raise ValueError("Focal gamma must be non-negative")
        
        if self.get("dice_smooth") <= 0:
            raise ValueError("Dice smooth must be positive")
        
        valid_strategies = ["simple", "weighted"]
        if self.get("strategy") not in valid_strategies:
            raise ValueError(f"OHEM strategy must be one of {valid_strategies}")
        
        if not 0 < self.get("hard_fraction") <= 1:
            raise ValueError("Hard fraction must be between 0 and 1")
        
        if self.get("hard_weight_min") < 1:
            raise ValueError("Hard weight min must be >= 1")
        
        if self.get("hard_weight_max") < self.get("hard_weight_min"):
            raise ValueError("Hard weight max must be >= hard weight min")

    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for Stage 2."""
        return {
            'focal': self.get("focal_loss_weight"),
            'dice': self.get("dice_loss_weight"),
            'boundary': self.get("boundary_loss_weight")
        }
    
    def should_visualize(self, epoch: int) -> bool:
        """Check if should visualize masks for current epoch."""
        return (
            self.get("visualize_masks") and 
            epoch % self.get("visualization_frequency") == 0
        )
    
    def is_deep_model(self) -> bool:
        """Determine if this is a deep model based on epochs."""
        return self.get("epochs", 0) > 600