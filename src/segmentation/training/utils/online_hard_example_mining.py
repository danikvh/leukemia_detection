"""
Online Hard Example Mining (OHEM) implementation for training.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging


class OnlineHardExampleMining:
    """
    Online Hard Example Mining for focusing on difficult samples during training.
    """
    
    def __init__(
        self,
        fraction: float = 0.25,
        hard_weight: float = 2.0,
        min_kept: int = 1,
        weighted: bool = True,
        use_loss_threshold: bool = False,
        loss_threshold_percentile: float = 75.0,
        min_hard_weight: float = 1.2,
        max_hard_weight: float = 2.5
    ):
        """
        Initialize OHEM with original logic parameters.
        
        Args:
            fraction: Fraction of samples to consider as hard examples
            hard_weight: Weight multiplier for hard examples (used in basic mode)
            min_kept: Minimum number of examples to keep
            weighted: Whether to use weighted sampling vs selection
            use_loss_threshold: Whether to use loss threshold for hard example selection
            loss_threshold_percentile: Percentile threshold for loss-based selection
            min_hard_weight: Minimum weight for adaptive weighting
            max_hard_weight: Maximum weight for adaptive weighting
        """
        self.fraction = fraction
        self.hard_weight = hard_weight
        self.min_kept = min_kept
        self.weighted = weighted
        self.use_loss_threshold = use_loss_threshold
        self.loss_threshold_percentile = loss_threshold_percentile
        self.min_hard_weight = min_hard_weight
        self.max_hard_weight = max_hard_weight
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.hard_examples_count = 0
        self.total_examples_count = 0
        self.loss_history = []
    
    def apply_mining(
        self, 
        losses_per_sample: List[Dict[str, torch.Tensor]], 
        batch_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply OHEM using the original weighted reconstruction logic.
        
        - Sort examples by combined_weighted (total_loss)
        - Apply different weighting strategies based on mode
        - Reconstruct individual loss components with weights
        
        Args:
            losses_per_sample: List of loss dictionaries for each sample
            batch_losses: Aggregated batch losses (not used, reconstructed from scratch)
            
        Returns:
            Reconstructed batch losses with OHEM applied
        """
        if not losses_per_sample:
            return batch_losses
        
        # Apply the original OHEM logic
        if self.weighted:
            return self._apply_adaptive_weighted_mining(losses_per_sample)
        else:
            return self._apply_basic_hard_mining(losses_per_sample)
    
    def _apply_basic_hard_mining(
        self,
        losses_per_sample: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply basic hard mining
        
        - Sort by total loss
        - Select top fraction as hard examples
        - Apply hard_weight to hard examples, 1.0 to others
        """
        num_prompts_processed = len(losses_per_sample)
        
        # Sort by total loss (combined_weighted in original)
        sorted_losses = sorted(
            enumerate(losses_per_sample), 
            key=lambda x: x[1]['total_loss'].item(), 
            reverse=True
        )
        
        # Calculate number of hard examples
        num_hard_examples = int(num_prompts_processed * self.fraction)
        if num_hard_examples == 0 and num_prompts_processed > 0:
            num_hard_examples = 1
        
        # Initialize accumulated losses
        device = losses_per_sample[0]['total_loss'].device
        accumulated_losses = {
            'focal_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'dice_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'boundary_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'total_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }
        
        # Process each example with appropriate weight
        for i, (original_idx, data) in enumerate(sorted_losses):
            if i < num_hard_examples:  # Hard example
                weight = self.hard_weight
                self.hard_examples_count += 1
            else:  # Easy example
                weight = 1.0
            
            # Accumulate weighted losses (original reconstruction logic)
            accumulated_losses['focal_loss'] = accumulated_losses['focal_loss'] + data['focal_loss'] * weight
            accumulated_losses['dice_loss'] = accumulated_losses['dice_loss'] + data['dice_loss'] * weight
            accumulated_losses['boundary_loss'] = accumulated_losses['boundary_loss'] + data['boundary_loss'] * weight
            accumulated_losses['total_loss'] = accumulated_losses['total_loss'] + data['total_loss'] * weight
        
        # Average over number of samples
        for key in accumulated_losses:
            accumulated_losses[key] = accumulated_losses[key] / num_prompts_processed
        
        self.total_examples_count += num_prompts_processed
        return accumulated_losses
    
    def _apply_adaptive_weighted_mining(
        self,
        losses_per_sample: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply adaptive weighted mining
        
        - Sort by total loss
        - Calculate adaptive weights based on loss magnitude within hard set
        - Apply normalized weighting between min_hard_weight and max_hard_weight
        """
        num_prompts_processed = len(losses_per_sample)
        
        # Sort by total loss (combined_weighted in original)
        sorted_losses = sorted(
            enumerate(losses_per_sample), 
            key=lambda x: x[1]['total_loss'].item(), 
            reverse=True
        )
        
        # Calculate number of examples to consider for adaptive weighting
        num_prompts_to_consider = int(num_prompts_processed * self.fraction)
        if num_prompts_to_consider == 0 and num_prompts_processed > 0:
            num_prompts_to_consider = 1
        
        # Get loss values of the selected fraction for scaling
        hard_example_losses = [
            sorted_losses[i][1]['total_loss'].item() 
            for i in range(min(num_prompts_to_consider, len(sorted_losses)))
        ]
        
        min_loss_in_hard_set = min(hard_example_losses) if hard_example_losses else 0
        max_loss_in_hard_set = max(hard_example_losses) if hard_example_losses else 0
        
        # Initialize accumulated losses
        device = losses_per_sample[0]['total_loss'].device
        accumulated_losses = {
            'focal_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'dice_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'boundary_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'total_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }
        
        # Process each example with adaptive weight
        for i, (original_idx, data) in enumerate(sorted_losses):
            weight_multiplier = 1.0  # Default weight for "easy" examples
            
            if i < num_prompts_to_consider:  # This is a "hard" example
                if max_loss_in_hard_set > min_loss_in_hard_set:  # Avoid division by zero
                    # Normalize badness within the hard set (0 for least hard, 1 for most hard)
                    normalized_badness = (
                        data['total_loss'].item() - min_loss_in_hard_set
                    ) / (max_loss_in_hard_set - min_loss_in_hard_set)
                elif max_loss_in_hard_set > 0:  # All hard examples have same loss
                    normalized_badness = 0.5
                else:  # All losses are zero or negative
                    normalized_badness = 0.0
                
                weight_multiplier = (
                    self.min_hard_weight + 
                    (self.max_hard_weight - self.min_hard_weight) * normalized_badness
                )
                self.hard_examples_count += 1
            
            # Accumulate weighted losses (exact original logic)
            accumulated_losses['focal_loss'] = accumulated_losses['focal_loss'] + data['focal_loss'] * weight_multiplier
            accumulated_losses['dice_loss'] = accumulated_losses['dice_loss'] + data['dice_loss'] * weight_multiplier
            accumulated_losses['boundary_loss'] = accumulated_losses['boundary_loss'] + data['boundary_loss'] * weight_multiplier
            accumulated_losses['total_loss'] = accumulated_losses['total_loss'] + data['total_loss'] * weight_multiplier
        
        # Average over number of samples
        for key in accumulated_losses:
            accumulated_losses[key] = accumulated_losses[key] / num_prompts_processed
        
        self.total_examples_count += num_prompts_processed
        return accumulated_losses
    
    def get_mining_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mining process."""
        if self.total_examples_count == 0:
            return {}
        
        return {
            'hard_examples_ratio': self.hard_examples_count / self.total_examples_count,
            'total_examples_processed': self.total_examples_count,
            'hard_examples_selected': self.hard_examples_count,
            'recent_loss_stats': self.loss_history[-10:] if self.loss_history else []
        }
    
    def reset_statistics(self) -> None:
        """Reset mining statistics."""
        self.hard_examples_count = 0
        self.total_examples_count = 0
        self.loss_history.clear()
    
    def update_parameters(
        self, 
        fraction: Optional[float] = None,
        hard_weight: Optional[float] = None,
        min_hard_weight: Optional[float] = None,
        max_hard_weight: Optional[float] = None
    ) -> None:
        """
        Update OHEM parameters during training.
        
        Args:
            fraction: New fraction of hard examples
            hard_weight: New weight for hard examples (basic mode)
            min_hard_weight: New minimum weight for adaptive mode
            max_hard_weight: New maximum weight for adaptive mode
        """
        if fraction is not None:
            self.fraction = max(0.0, min(1.0, fraction))
            self.logger.info(f"Updated OHEM fraction to {self.fraction}")
        
        if hard_weight is not None:
            self.hard_weight = max(1.0, hard_weight)
            self.logger.info(f"Updated OHEM hard weight to {self.hard_weight}")
        
        if min_hard_weight is not None:
            self.min_hard_weight = max(1.0, min_hard_weight)
            self.logger.info(f"Updated OHEM min hard weight to {self.min_hard_weight}")
        
        if max_hard_weight is not None:
            self.max_hard_weight = max(self.min_hard_weight, max_hard_weight)
            self.logger.info(f"Updated OHEM max hard weight to {self.max_hard_weight}")

