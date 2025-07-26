"""
Base loss classes for the training framework.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions."""
    
    def __init__(self, weight: float = 1.0, name: str = None):
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def forward(self, predictions, targets, **kwargs):
        """Compute the loss."""
        pass
    
    def __call__(self, predictions, targets, **kwargs):
        loss = self.forward(predictions, targets, **kwargs)
        return loss * self.weight


class WeightedCombinedLoss(nn.Module):
    """Combines multiple losses with weights."""
    
    def __init__(self, losses_dict: dict):
        """
        Args:
            losses_dict: Dictionary of {'loss_name': (loss_instance, weight)}
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses_dict.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
    
    def forward(self, predictions, targets, **kwargs):
        total_loss = 0
        loss_components = {}
        
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(predictions, targets, **kwargs)
            weighted_loss = component_loss * self.weights[name]
            total_loss += weighted_loss
            loss_components[f"loss_{name}"] = component_loss.item()
        
        return total_loss, loss_components
    
    def get_loss_weights(self):
        return self.weights.copy()
    
    def update_weights(self, new_weights: dict):
        """Update loss weights dynamically."""
        for name, weight in new_weights.items():
            if name in self.weights:
                self.weights[name] = weight