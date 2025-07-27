"""
Checkpoint manager for saving and loading model states during training.
"""
import os
import torch
import json
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging


class CheckpointManager:
    """Manages model checkpointing during training."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 fold: int = 0,
                 device: torch.device = None,
                 experiment_name: str = "experiment",
                 keep_best_only: bool = False,
                 keep_last_n: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            fold: Current fold number
            experiment_name: Name of the experiment
            keep_best_only: Whether to keep only the best checkpoint
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.fold = fold
        self.device = device
        self.keep_best_only = keep_best_only
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory structure
        self.fold_dir = self.checkpoint_dir / f"fold_{fold + 1}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for different checkpoint types
        self.best_model_path = self.fold_dir / "best_model.pth"
        self.final_model_path = self.fold_dir / "final_model.pth"
        self.last_checkpoint_path = self.fold_dir / "last_checkpoint.pth"
        
        # Stage-specific paths
        self.best_s1_path = self.fold_dir / "best_model_s1.pth"
        self.best_s2_path = self.fold_dir / "best_model_s2.pth"
        
        # Tracking
        self.best_metric = float('inf')
        self.best_s1_metric = float('inf')
        self.best_s2_metric = float('inf')
        self.saved_checkpoints = []
        
        # Logger
        self.logger = logging.getLogger(f"CheckpointManager_fold_{fold}")
        
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       epoch: int,
                       metric_value: float,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       stage: Optional[str] = None,
                       metric_name: str = "loss",
                       is_best: bool = False,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metric_value: Value of the metric
            optimizer: Optimizer state to save (optional)
            scheduler: Scheduler state to save (optional) 
            stage: Training stage ('stage1' or 'stage2')
            metric_name: Name of the metric
            is_best: Whether this is the best checkpoint so far
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metric_value': metric_value,
            'metric_name': metric_name,
            'fold': self.fold,
            'experiment_name': self.experiment_name,
            'stage': stage
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)
        
        # Determine checkpoint path
        if is_best:
            if stage == 'stage1':
                checkpoint_path = self.best_s1_path
                self.best_s1_metric = metric_value
            elif stage == 'stage2':
                checkpoint_path = self.best_s2_path
                self.best_s2_metric = metric_value
            else:
                checkpoint_path = self.best_model_path
                self.best_metric = metric_value
        else:
            if stage:
                checkpoint_path = self.fold_dir / f"{stage}_epoch_{epoch}.pth"
            else:
                checkpoint_path = self.fold_dir / f"checkpoint_epoch_{epoch}.pth"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update last checkpoint path
        if checkpoint_path != self.last_checkpoint_path:
            shutil.copy2(checkpoint_path, self.last_checkpoint_path)
        
        # Track saved checkpoints
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'stage': stage,
            'metric_value': metric_value,
            'is_best': is_best
        }
        self.saved_checkpoints.append(checkpoint_info)
        
        # Clean up old checkpoints if needed
        if not self.keep_best_only:
            self._cleanup_checkpoints()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            self.logger.info(f"  -> New best {stage or 'model'} (epoch {epoch}, {metric_name}: {metric_value:.4f})")
        
        return str(checkpoint_path)
    
    def save_best_model(self,
                       model: torch.nn.Module,
                       epoch: int,
                       metric_value: float,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       stage: Optional[str] = None) -> str:
        """
        Save the best model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metric_value: Best metric value
            optimizer: Optimizer state (optional)
            stage: Training stage (optional)
            
        Returns:
            Path to saved checkpoint
        """
        return self.save_checkpoint(
            model=model,
            epoch=epoch,
            metric_value=metric_value,
            optimizer=optimizer,
            stage=stage,
            is_best=True
        )
    
    def save_final_model(self,
                        model: torch.nn.Module,
                        epoch: int,
                        metric_value: float,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        stage: Optional[str] = None) -> str:
        """
        Save the final model checkpoint.
        
        Args:
            model: Model to save
            epoch: Final epoch
            metric_value: Final metric value
            optimizer: Optimizer state (optional)
            stage: Training stage (optional)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metric_value': metric_value,
            'fold': self.fold,
            'experiment_name': self.experiment_name,
            'stage': stage,
            'is_final': True
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, self.final_model_path)
        self.logger.info(f"Final model saved: {self.final_model_path}")
        
        return str(self.final_model_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       stage: Optional[str] = None,
                       load_best: bool = True,
                       load_final: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            stage: Stage to load best checkpoint for ('stage1' or 'stage2')
            load_best: Whether to load the best checkpoint
            load_final: Whether to load the final checkpoint
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_final:
            path = self.final_model_path
        elif stage and load_best:
            if stage == 'stage1':
                path = self.best_s1_path
            elif stage == 'stage2':
                path = self.best_s2_path
            else:
                path = self.best_model_path
        elif load_best:
            path = self.best_model_path
        else:
            path = self.last_checkpoint_path
        
        if not path.exists():
            self.logger.warning(f"Checkpoint not found: {path}")
            return None
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.logger.info(f"Checkpoint loaded: {path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {path}: {e}")
            return None
    
    def load_model_state(self, 
                        model: torch.nn.Module,
                        checkpoint_path: Optional[str] = None,
                        stage: Optional[str] = None,
                        strict: bool = True,
                        load_best: bool = True) -> bool:
        """
        Load model state from checkpoint.
        
        Args:
            model: Model to load state into
            checkpoint_path: Specific checkpoint path to load
            stage: Stage to load best checkpoint for
            strict: Whether to strictly enforce key matching
            load_best: Whether to load the best checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint = self.load_checkpoint(checkpoint_path, stage, load_best)
        if checkpoint is None:
            return False
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            self.logger.info("Model state loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model state: {e}")
            return False
    
    def load_optimizer_state(self,
                           optimizer: torch.optim.Optimizer,
                           checkpoint_path: Optional[str] = None,
                           stage: Optional[str] = None) -> bool:
        """
        Load optimizer state from checkpoint.
        
        Args:
            optimizer: Optimizer to load state into
            checkpoint_path: Specific checkpoint path to load
            stage: Stage to load best checkpoint for
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint = self.load_checkpoint(checkpoint_path, stage)
        if checkpoint is None or 'optimizer_state_dict' not in checkpoint:
            return False
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading optimizer state: {e}")
            return False
    
    def get_best_model_path(self, stage: Optional[str] = None) -> str:
        """
        Get path to the best model checkpoint.
        
        Args:
            stage: Stage to get best model for ('stage1' or 'stage2')
            
        Returns:
            Path to best model checkpoint
        """
        if stage == 'stage1':
            return str(self.best_s1_path)
        elif stage == 'stage2':
            return str(self.best_s2_path)
        else:
            return str(self.best_model_path)
    
    def get_final_model_path(self) -> str:
        """Get path to the final model checkpoint."""
        return str(self.final_model_path)
    
    def get_best_checkpoint_info(self, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about the best checkpoint for a stage.
        
        Args:
            stage: Stage to get info for ('stage1' or 'stage2')
            
        Returns:
            Dictionary with checkpoint info or None
        """
        if stage == 'stage1':
            path = self.best_s1_path
            metric = self.best_s1_metric
        elif stage == 'stage2':
            path = self.best_s2_path
            metric = self.best_s2_metric
        else:
            path = self.best_model_path
            metric = self.best_metric
        
        if not path.exists():
            return None
        
        return {
            'path': str(path),
            'metric_value': metric,
            'exists': True
        }
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        if self.keep_best_only:
            # Keep only best checkpoints
            for checkpoint_info in self.saved_checkpoints:
                if not checkpoint_info['is_best']:
                    path = Path(checkpoint_info['path'])
                    if path.exists() and path != self.last_checkpoint_path:
                        try:
                            path.unlink()
                        except Exception as e:
                            self.logger.warning(f"Failed to remove checkpoint {path}: {e}")
        elif self.keep_last_n > 0:
            # Keep only last N checkpoints (excluding best ones)
            non_best_checkpoints = [
                info for info in self.saved_checkpoints 
                if not info['is_best']
            ]
            
            if len(non_best_checkpoints) > self.keep_last_n:
                to_remove = non_best_checkpoints[:-self.keep_last_n]
                for checkpoint_info in to_remove:
                    path = Path(checkpoint_info['path'])
                    if path.exists():
                        try:
                            path.unlink()
                            self.saved_checkpoints.remove(checkpoint_info)
                        except Exception as e:
                            self.logger.warning(f"Failed to remove checkpoint {path}: {e}")
    
    def save_training_info(self, info: Dict[str, Any]) -> None:
        """
        Save training information to JSON file.
        
        Args:
            info: Training information dictionary
        """
        info_path = self.fold_dir / "training_info.json"
        
        # Add checkpoint manager info
        info.update({
            'checkpoint_dir': str(self.checkpoint_dir),
            'experiment_name': self.experiment_name,
            'fold': self.fold,
            'best_metric': self.best_metric,
            'best_s1_metric': self.best_s1_metric,
            'best_s2_metric': self.best_s2_metric,
            'saved_checkpoints': self.saved_checkpoints
        })
        
        # Make serializable
        serializable_info = self._make_serializable(info)
        
        with open(info_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.fold_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'stage': checkpoint.get('stage', 'unknown'),
                    'metric_value': checkpoint.get('metric_value', 'unknown'),
                    'metric_name': checkpoint.get('metric_name', 'unknown'),
                    'is_best': checkpoint_file.name.startswith('best_'),
                    'is_final': checkpoint.get('is_final', False)
                })
            except Exception as e:
                self.logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get('epoch', 0))
    
    def has_checkpoint(self, stage: Optional[str] = None, checkpoint_type: str = 'best') -> bool:
        """
        Check if a checkpoint exists.
        
        Args:
            stage: Stage to check for ('stage1', 'stage2', or None)
            checkpoint_type: Type of checkpoint ('best', 'final', 'last')
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        if checkpoint_type == 'best':
            if stage == 'stage1':
                return self.best_s1_path.exists()
            elif stage == 'stage2':
                return self.best_s2_path.exists()
            else:
                return self.best_model_path.exists()
        elif checkpoint_type == 'final':
            return self.final_model_path.exists()
        elif checkpoint_type == 'last':
            return self.last_checkpoint_path.exists()
        else:
            return False
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all checkpoints.
        
        Returns:
            Summary dictionary
        """
        return {
            'fold': self.fold,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_metric': self.best_metric,
            'best_s1_metric': self.best_s1_metric,
            'best_s2_metric': self.best_s2_metric,
            'total_checkpoints': len(self.saved_checkpoints),
            'has_best_model': self.has_checkpoint(checkpoint_type='best'),
            'has_best_s1': self.has_checkpoint(stage='stage1', checkpoint_type='best'),
            'has_best_s2': self.has_checkpoint(stage='stage2', checkpoint_type='best'),
            'has_final_model': self.has_checkpoint(checkpoint_type='final'),
            'saved_checkpoints': self.saved_checkpoints
        }


class MultiStageCheckpointManager:
    """Manages checkpoints across multiple training stages with enhanced functionality."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 experiment_name: str = "multi_stage_experiment",
                 fold: int = 0):
        """
        Initialize multi-stage checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            experiment_name: Name of the experiment
            fold: Current fold number
        """
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, fold, experiment_name
        )
        self.stage_history = []
        self.logger = logging.getLogger(f"MultiStageCheckpointManager_fold_{fold}")
    
    def save_stage_checkpoint(self,
                            model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            epoch: int,
                            stage: str,
                            metric_value: float,
                            is_best: bool = False,
                            additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint for a specific stage.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            stage: Training stage
            metric_value: Metric value
            is_best: Whether this is the best for this stage
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        # Add stage history to additional info
        stage_info = {'stage_history': self.stage_history}
        if additional_info:
            stage_info.update(additional_info)
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=model,
            epoch=epoch,
            metric_value=metric_value,
            optimizer=optimizer,
            stage=stage,
            is_best=is_best,
            additional_info=stage_info
        )
        
        # Update stage history
        history_entry = {
            'stage': stage,
            'epoch': epoch,
            'metric_value': metric_value,
            'is_best': is_best,
            'checkpoint_path': checkpoint_path
        }
        self.stage_history.append(history_entry)
        
        return checkpoint_path
    
    def load_best_from_previous_stage(self,
                                    model: torch.nn.Module,
                                    previous_stage: str) -> bool:
        """
        Load the best checkpoint from a previous stage.
        
        Args:
            model: Model to load state into
            previous_stage: Name of the previous stage
            
        Returns:
            True if successful, False otherwise
        """
        success = self.checkpoint_manager.load_model_state(
            model=model,
            stage=previous_stage,
            strict=True
        )
        
        if success:
            self.logger.info(f"Loaded best model from {previous_stage}")
        else:
            self.logger.warning(f"Failed to load best model from {previous_stage}")
        
        return success
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stages.
        
        Returns:
            Summary dictionary
        """
        stages = set(info['stage'] for info in self.stage_history)
        
        return {
            'total_stages': len(stages),
            'stages': list(stages),
            'stage_history': self.stage_history,
            'best_s1_info': self.checkpoint_manager.get_best_checkpoint_info('stage1'),
            'best_s2_info': self.checkpoint_manager.get_best_checkpoint_info('stage2'),
            'checkpoint_summary': self.checkpoint_manager.get_checkpoint_summary()
        }
    
    def finalize_training(self, 
                         model: torch.nn.Module,
                         final_epoch: int,
                         final_metric: float) -> str:
        """
        Finalize multi-stage training by saving final model.
        
        Args:
            model: Final model to save
            final_epoch: Final epoch number
            final_metric: Final metric value
            
        Returns:
            Path to final model checkpoint
        """
        final_path = self.checkpoint_manager.save_final_model(
            model=model,
            epoch=final_epoch,
            metric_value=final_metric,
            stage='final'
        )
        
        # Save complete training info
        training_info = {
            'multi_stage_training': True,
            'stage_summary': self.get_stage_summary(),
            'final_epoch': final_epoch,
            'final_metric': final_metric,
            'final_model_path': final_path
        }
        
        self.checkpoint_manager.save_training_info(training_info)
        
        return final_path
    
    # Delegate methods to underlying checkpoint manager
    def get_best_model_path(self, stage: Optional[str] = None) -> str:
        """Get path to best model for given stage."""
        return self.checkpoint_manager.get_best_model_path(stage)
    
    def get_final_model_path(self) -> str:
        """Get path to final model."""
        return self.checkpoint_manager.get_final_model_path()
    
    def load_model_state(self, model: torch.nn.Module, stage: Optional[str] = None, **kwargs) -> bool:
        """Load model state from checkpoint."""
        return self.checkpoint_manager.load_model_state(model, stage=stage, **kwargs)
    
    def has_checkpoint(self, stage: Optional[str] = None, checkpoint_type: str = 'best') -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_manager.has_checkpoint(stage, checkpoint_type)