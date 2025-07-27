"""
Abstract base trainer class for CellSAM training.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging

from segmentation.training.config.base_config import BaseTrainingConfig
from segmentation.training.utils.early_stopping import EarlyStopping
from segmentation.training.utils.checkpoint_manager import CheckpointManager
from segmentation.training.utils.gpu_monitoring import GPUMonitor
from segmentation.training.utils.metrics_tracker import MetricsTracker


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainingConfig,
        device: torch.device,
        output_dir: str,
        fold: int = 0
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.fold = fold

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Utilities
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        self.checkpoint_manager = CheckpointManager(output_dir, fold, self.device)
        self.gpu_monitor = GPUMonitor() if config.debug else None
        self.metrics_tracker = MetricsTracker(self.output_dir)
        
        # Logger
        self.logger = self._setup_logger()

        # Debugging
        self.debug = config.debug
        self.visualization_frequency = config.visualization_frequency
        
        # Optimizer and scheduler (to be set by subclasses)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this trainer."""
        logger = logging.getLogger(f"{self.__class__.__name__}_fold_{self.fold}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def setup_training_components(self) -> None:
        """Setup optimizer, scheduler, and criterion."""
        pass
    
    @abstractmethod
    def forward_pass(self, batch: Any) -> Tuple[Any, Any]:
        """
        Perform forward pass and return outputs and targets.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (outputs, targets)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """
        Compute loss from outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses (individual and total)
        """
        pass

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("Start of epoch")
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            outputs, targets = self.forward_pass(batch)

            # Compute loss
            loss_dict = self.compute_loss(outputs, targets)
            
            # Get total loss 
            total_loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.get_trainable_parameters(),
                    max_norm=self.config.grad_clip
                )
            
            self.optimizer.step()
            
            # Track losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()
            
            # Ensure total_loss is tracked
            if 'total_loss' not in epoch_losses:
                epoch_losses['total_loss'] = 0
            epoch_losses['total_loss'] += total_loss.item()
            
            # Debug logging
            if self.config.debug and batch_idx % 10 == 0:
                self.logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss.item():.4f}"
                )

            if self.debug and self.current_epoch % self.visualization_frequency == 0:
                self.visualize_predictions(batch[0], outputs, targets, self.current_epoch, batch_idx, "train")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("End of epoch")
            
        # Call post-step hook for subclass-specific logic
        self._post_step_hook(batch_idx, batch)
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Forward pass
                outputs, targets = self.forward_pass(batch)
                
                # Compute loss
                loss_dict = self.compute_loss(outputs, targets)
                total_loss = sum(loss_dict.values())
                
                # Track losses
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()
                
                if 'total_loss' not in epoch_losses:
                    epoch_losses['total_loss'] = 0
                epoch_losses['total_loss'] += total_loss.item()

                if self.debug and self.current_epoch % self.visualization_frequency == 0:
                    self.visualize_predictions(batch[0], outputs, targets, self.current_epoch, batch_idx, "val")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)
        
        return epoch_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training results dictionary
        """
        self.setup_training_components()
        
        self.logger.info(
            f"Starting training for {self.config.epochs} epochs"
        )
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_loader)
            self.metrics_tracker.log_epoch_metrics('train', epoch, train_losses)
            
            # Log training results
            train_loss_str = ", ".join([
                f"{k}: {v:.6f}" for k, v in train_losses.items()
            ])
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - Train - {train_loss_str}"
            )
            
            # Validation
            val_losses = {}
            if val_loader is not None:
                val_losses = self.validate_epoch(val_loader)
                self.metrics_tracker.log_epoch_metrics('val', epoch, val_losses)
                
                val_loss_str = ", ".join([
                    f"{k}: {v:.6f}" for k, v in val_losses.items()
                ])
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - Val - {val_loss_str}"
                )
                
                # Check for best model
                current_val_loss = val_losses.get('total_loss', float('inf'))
                if current_val_loss < self.best_loss:
                    self.best_loss = current_val_loss
                    self.best_epoch = epoch + 1
                    self.checkpoint_manager.save_best_model(
                        self.model, epoch, current_val_loss
                    )
                    self.logger.info(
                        f"New best model saved with val loss: {current_val_loss:.6f}"
                    )
                
                # Early stopping check
                if self.early_stopping.should_stop(current_val_loss):
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch+1}"
                    )
                    break
            else:
                # No validation - save model periodically
                if (epoch + 1) % 50 == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, epoch, train_losses['total_loss']
                    )
            
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_losses.get('total_loss', train_losses['total_loss'])
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        # Final save
        final_loss = val_losses.get('total_loss', train_losses['total_loss'])
        self.checkpoint_manager.save_final_model(self.model, self.current_epoch, final_loss)
        
        results = {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'final_epoch': self.current_epoch + 1,
            'final_loss': final_loss,
            'train_history': self.metrics_tracker.get_history('train'),
            'val_history': self.metrics_tracker.get_history('val') if val_loader else None
        }
        
        self.logger.info(
            f"Training completed. Best model at epoch {self.best_epoch} "
            f"with loss {self.best_loss:.6f}"
        )
        
        return results

    def _post_step_hook(self, batch_idx: int, batch: Any) -> None:
        """Hook called after each optimization step. Override in subclasses."""
        pass

    def visualize_predictions(self, batch: Any, outputs: Any, targets: Any, epoch: int, batch_idx: int, stage: str) -> None:
        """Create visualizations for debugging. Override in subclasses."""
        pass

    def freeze_parameters(self, module: nn.Module) -> None:
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self, module: nn.Module) -> None:
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self) -> list:
        """Get all trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def count_parameters(self, only_trainable: bool = True) -> int:
        """Count model parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())