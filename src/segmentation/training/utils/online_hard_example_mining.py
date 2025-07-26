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
        loss_threshold_percentile: float = 75.0
    ):
        """
        Initialize OHEM.
        
        Args:
            fraction: Fraction of samples to consider as hard examples
            hard_weight: Weight multiplier for hard examples
            min_kept: Minimum number of examples to keep
            weighted: Whether to use weighted sampling
            use_loss_threshold: Whether to use loss threshold for hard example selection
            loss_threshold_percentile: Percentile threshold for loss-based selection
        """
        self.fraction = fraction
        self.hard_weight = hard_weight
        self.min_kept = min_kept
        self.weighted = weighted
        self.use_loss_threshold = use_loss_threshold
        self.loss_threshold_percentile = loss_threshold_percentile
        
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
        Apply OHEM to modify batch losses based on hard example mining.
        
        Args:
            losses_per_sample: List of loss dictionaries for each sample
            batch_losses: Aggregated batch losses
            
        Returns:
            Modified batch losses with OHEM applied
        """
        if not losses_per_sample:
            return batch_losses
        
        # Extract total losses for each sample
        sample_losses = []
        for sample_loss_dict in losses_per_sample:
            total_loss = sample_loss_dict.get('total_loss', 0.0)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            sample_losses.append(total_loss)
        
        # Find hard examples
        hard_indices = self._find_hard_examples(sample_losses)
        
        if not hard_indices:
            return batch_losses
        
        # Apply mining strategy
        if self.weighted:
            return self._apply_weighted_mining(
                losses_per_sample, batch_losses, hard_indices
            )
        else:
            return self._apply_selection_mining(
                losses_per_sample, batch_losses, hard_indices
            )
    
    def _find_hard_examples(self, sample_losses: List[float]) -> List[int]:
        """
        Find indices of hard examples based on loss values.
        
        Args:
            sample_losses: List of loss values for each sample
            
        Returns:
            List of indices of hard examples
        """
        if not sample_losses:
            return []
        
        sample_losses = np.array(sample_losses)
        n_samples = len(sample_losses)
        
        if self.use_loss_threshold:
            # Use percentile-based threshold
            threshold = np.percentile(sample_losses, self.loss_threshold_percentile)
            hard_indices = np.where(sample_losses >= threshold)[0].tolist()
        else:
            # Use fraction-based selection
            n_hard = max(self.min_kept, int(n_samples * self.fraction))
            n_hard = min(n_hard, n_samples)  # Don't exceed total samples
            
            # Get indices of samples with highest losses
            hard_indices = np.argsort(sample_losses)[-n_hard:].tolist()
        
        # Update statistics
        self.hard_examples_count += len(hard_indices)
        self.total_examples_count += n_samples
        
        # Store loss statistics
        if sample_losses.size > 0:
            self.loss_history.append({
                'mean_loss': float(np.mean(sample_losses)),
                'max_loss': float(np.max(sample_losses)),
                'min_loss': float(np.min(sample_losses)),
                'hard_examples': len(hard_indices),
                'total_examples': n_samples
            })
        
        return hard_indices
    
    def _apply_weighted_mining(
        self,
        losses_per_sample: List[Dict[str, torch.Tensor]],
        batch_losses: Dict[str, torch.Tensor],
        hard_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply weighted mining - increase weights of hard examples.
        
        Args:
            losses_per_sample: Per-sample losses
            batch_losses: Original batch losses
            hard_indices: Indices of hard examples
            
        Returns:
            Modified batch losses
        """
        n_samples = len(losses_per_sample)
        
        # Create weight array
        weights = torch.ones(n_samples, device=batch_losses['total_loss'].device)
        weights[hard_indices] = self.hard_weight
        
        # Recompute weighted losses
        modified_losses = {}
        
        for loss_name in batch_losses.keys():
            if loss_name == 'total_loss':
                continue
            
            # Collect per-sample losses for this loss type
            sample_losses_tensor = []
            for i, sample_loss_dict in enumerate(losses_per_sample):
                if loss_name in sample_loss_dict:
                    loss_val = sample_loss_dict[loss_name]
                    if isinstance(loss_val, torch.Tensor):
                        sample_losses_tensor.append(loss_val)
                    else:
                        sample_losses_tensor.append(torch.tensor(loss_val, device=weights.device))
                else:
                    sample_losses_tensor.append(torch.tensor(0.0, device=weights.device))
            
            if sample_losses_tensor:
                sample_losses_tensor = torch.stack(sample_losses_tensor)
                weighted_loss = torch.sum(sample_losses_tensor * weights) / n_samples
                modified_losses[loss_name] = weighted_loss
        
        # Recompute total loss
        modified_losses['total_loss'] = sum(modified_losses.values())
        
        return modified_losses
    
    def _apply_selection_mining(
        self,
        losses_per_sample: List[Dict[str, torch.Tensor]],
        batch_losses: Dict[str, torch.Tensor],
        hard_indices: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply selection mining - only use hard examples for loss computation.
        
        Args:
            losses_per_sample: Per-sample losses
            batch_losses: Original batch losses
            hard_indices: Indices of hard examples
            
        Returns:
            Modified batch losses using only hard examples
        """
        if not hard_indices:
            return batch_losses
        
        # Recompute losses using only hard examples
        modified_losses = {}
        
        for loss_name in batch_losses.keys():
            if loss_name == 'total_loss':
                continue
            
            # Collect losses from hard examples only
            hard_losses = []
            for idx in hard_indices:
                if idx < len(losses_per_sample) and loss_name in losses_per_sample[idx]:
                    loss_val = losses_per_sample[idx][loss_name]
                    if isinstance(loss_val, torch.Tensor):
                        hard_losses.append(loss_val)
                    else:
                        hard_losses.append(torch.tensor(loss_val, device=batch_losses['total_loss'].device))
            
            if hard_losses:
                modified_losses[loss_name] = torch.stack(hard_losses).mean()
            else:
                modified_losses[loss_name] = batch_losses[loss_name]
        
        # Recompute total loss
        modified_losses['total_loss'] = sum(modified_losses.values())
        
        return modified_losses
    
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
        hard_weight: Optional[float] = None
    ) -> None:
        """
        Update OHEM parameters during training.
        
        Args:
            fraction: New fraction of hard examples
            hard_weight: New weight for hard examples
        """
        if fraction is not None:
            self.fraction = max(0.0, min(1.0, fraction))
            self.logger.info(f"Updated OHEM fraction to {self.fraction}")
        
        if hard_weight is not None:
            self.hard_weight = max(1.0, hard_weight)
            self.logger.info(f"Updated OHEM hard weight to {self.hard_weight}")


class AdaptiveOHEM(OnlineHardExampleMining):
    """
    Adaptive OHEM that adjusts parameters based on training progress.
    """
    
    def __init__(
        self,
        initial_fraction: float = 0.25,
        final_fraction: float = 0.1,
        initial_hard_weight: float = 2.0,
        final_hard_weight: float = 1.5,
        adaptation_epochs: int = 50,
        **kwargs
    ):
        """
        Initialize adaptive OHEM.
        
        Args:
            initial_fraction: Starting fraction of hard examples
            final_fraction: Final fraction of hard examples
            initial_hard_weight: Starting weight for hard examples
            final_hard_weight: Final weight for hard examples
            adaptation_epochs: Number of epochs over which to adapt
        """
        super().__init__(fraction=initial_fraction, hard_weight=initial_hard_weight, **kwargs)
        
        self.initial_fraction = initial_fraction
        self.final_fraction = final_fraction
        self.initial_hard_weight = initial_hard_weight
        self.final_hard_weight = final_hard_weight
        self.adaptation_epochs = adaptation_epochs
        
        self.current_epoch = 0
    
    def step_epoch(self, epoch: int) -> None:
        """
        Update OHEM parameters based on current epoch.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        if epoch >= self.adaptation_epochs:
            # Use final parameters
            new_fraction = self.final_fraction
            new_hard_weight = self.final_hard_weight
        else:
            # Linear interpolation between initial and final parameters
            progress = epoch / self.adaptation_epochs
            
            new_fraction = (
                self.initial_fraction + 
                progress * (self.final_fraction - self.initial_fraction)
            )
            
            new_hard_weight = (
                self.initial_hard_weight + 
                progress * (self.final_hard_weight - self.initial_hard_weight)
            )
        
        # Update parameters if they changed significantly
        if abs(new_fraction - self.fraction) > 0.01:
            self.fraction = new_fraction
            self.logger.info(f"Epoch {epoch}: Adapted OHEM fraction to {self.fraction:.3f}")
        
        if abs(new_hard_weight - self.hard_weight) > 0.05:
            self.hard_weight = new_hard_weight
            self.logger.info(f"Epoch {epoch}: Adapted OHEM hard weight to {self.hard_weight:.3f}")


class CurriculumOHEM(OnlineHardExampleMining):
    """
    Curriculum learning with OHEM - gradually increase difficulty.
    """
    
    def __init__(
        self,
        curriculum_schedule: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Initialize curriculum OHEM.
        
        Args:
            curriculum_schedule: List of curriculum stages with epoch ranges and parameters
                Example: [
                    {'epochs': (0, 20), 'fraction': 0.1, 'hard_weight': 1.2},
                    {'epochs': (20, 50), 'fraction': 0.25, 'hard_weight': 1.5},
                    {'epochs': (50, 100), 'fraction': 0.4, 'hard_weight': 2.0}
                ]
        """
        super().__init__(**kwargs)
        self.curriculum_schedule = curriculum_schedule
        self.current_epoch = 0
    
    def step_epoch(self, epoch: int) -> None:
        """Update parameters based on curriculum schedule."""
        self.current_epoch = epoch
        
        # Find current curriculum stage
        current_stage = None
        for stage in self.curriculum_schedule:
            start_epoch, end_epoch = stage['epochs']
            if start_epoch <= epoch < end_epoch:
                current_stage = stage
                break
        
        if current_stage is None:
            # Use last stage if beyond schedule
            current_stage = self.curriculum_schedule[-1]
        
        # Update parameters
        new_fraction = current_stage.get('fraction', self.fraction)
        new_hard_weight = current_stage.get('hard_weight', self.hard_weight)
        
        if abs(new_fraction - self.fraction) > 0.01:
            self.fraction = new_fraction
            self.logger.info(f"Epoch {epoch}: Curriculum updated OHEM fraction to {self.fraction:.3f}")
        
        if abs(new_hard_weight - self.hard_weight) > 0.05:
            self.hard_weight = new_hard_weight
            self.logger.info(f"Epoch {epoch}: Curriculum updated OHEM hard weight to {self.hard_weight:.3f}")


class MultiScaleOHEM:
    """
    Multi-scale OHEM for handling examples at different difficulty scales.
    """
    
    def __init__(
        self,
        scale_configs: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize multi-scale OHEM.
        
        Args:
            scale_configs: Dictionary mapping scale names to OHEM configurations
                Example: {
                    'easy': {'fraction': 0.1, 'hard_weight': 1.1},
                    'medium': {'fraction': 0.3, 'hard_weight': 1.5},
                    'hard': {'fraction': 0.6, 'hard_weight': 2.0}
                }
        """
        self.ohem_instances = {}
        
        for scale_name, config in scale_configs.items():
            self.ohem_instances[scale_name] = OnlineHardExampleMining(**config)
        
        self.logger = logging.getLogger(__name__)
    
    def apply_mining(
        self,
        losses_per_sample: List[Dict[str, torch.Tensor]],
        batch_losses: Dict[str, torch.Tensor],
        difficulty_scores: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multi-scale OHEM based on difficulty scores.
        
        Args:
            losses_per_sample: Per-sample losses
            batch_losses: Batch losses
            difficulty_scores: Difficulty score for each sample
            
        Returns:
            Modified batch losses
        """
        if not difficulty_scores or len(difficulty_scores) != len(losses_per_sample):
            # Fallback to regular OHEM if no difficulty scores
            if self.ohem_instances:
                first_ohem = next(iter(self.ohem_instances.values()))
                return first_ohem.apply_mining(losses_per_sample, batch_losses)
            return batch_losses
        
        # Categorize samples by difficulty
        difficulty_thresholds = self._compute_difficulty_thresholds(difficulty_scores)
        
        categorized_samples = {scale: [] for scale in self.ohem_instances.keys()}
        scale_names = list(self.ohem_instances.keys())
        
        for i, score in enumerate(difficulty_scores):
            # Assign to appropriate scale based on thresholds
            assigned_scale = scale_names[-1]  # Default to hardest scale
            
            for j, threshold in enumerate(difficulty_thresholds[:-1]):
                if score <= threshold:
                    assigned_scale = scale_names[j]
                    break
            
            categorized_samples[assigned_scale].append(i)
        
        # Apply OHEM for each scale and combine results
        combined_losses = {}
        total_samples = len(losses_per_sample)
        
        for scale_name, sample_indices in categorized_samples.items():
            if not sample_indices:
                continue
            
            # Extract losses for this scale
            scale_losses = [losses_per_sample[i] for i in sample_indices]
            
            # Create temporary batch losses for this scale
            scale_batch_losses = {}
            for loss_name, loss_value in batch_losses.items():
                # Weight by number of samples in this scale
                scale_weight = len(sample_indices) / total_samples
                scale_batch_losses[loss_name] = loss_value * scale_weight
            
            # Apply OHEM for this scale
            ohem_instance = self.ohem_instances[scale_name]
            modified_scale_losses = ohem_instance.apply_mining(scale_losses, scale_batch_losses)
            
            # Combine with overall losses
            for loss_name, loss_value in modified_scale_losses.items():
                if loss_name not in combined_losses:
                    combined_losses[loss_name] = loss_value
                else:
                    combined_losses[loss_name] += loss_value
        
        return combined_losses if combined_losses else batch_losses
    
    def _compute_difficulty_thresholds(self, difficulty_scores: List[float]) -> List[float]:
        """Compute thresholds for categorizing difficulty levels."""
        scores_array = np.array(difficulty_scores)
        n_scales = len(self.ohem_instances)
        
        # Use quantiles to define thresholds
        quantiles = np.linspace(0, 100, n_scales + 1)[1:]  # Exclude 0, include 100
        thresholds = [np.percentile(scores_array, q) for q in quantiles]
        
        return thresholds
    
    def get_mining_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all OHEM instances."""
        return {
            scale_name: ohem.get_mining_statistics()
            for scale_name, ohem in self.ohem_instances.items()
        }