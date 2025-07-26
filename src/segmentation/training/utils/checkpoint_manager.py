"""
Checkpoint manager for saving and loading model states during training.
"""
import os
import torch
import json
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path


class CheckpointManager:
    """Manages model checkpointing during training."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 experiment_name: str,
                 fold: int = 0,
                 keep_best_only: bool = False,
                 keep_last_n: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            experiment_name: Name of the experiment
            fold: Current fold number
            keep_best_only: Whether to keep only the best checkpoint
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.fold = fold
        self.keep_best_only = keep_best_only
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory structure
        self.fold_dir = self.checkpoint_dir / experiment_name / f"fold_{fold + 1}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for different checkpoint types
        self.best_s1_path = self.fold_dir / "best_model_s1.pth"
        self.best_s2_path = self.fold_dir / "best_model_s2.pth"
        self.last_checkpoint_path = self.fold_dir / "last_checkpoint.pth"
        
        # Tracking
        self.best_s1_metric = float('inf')
        self.best_s2_metric = float('inf')
        self.saved_checkpoints = []
        
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       stage: str,
                       metric_value: float,
                       metric_name: str = "loss",
                       is_best: bool = False,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            stage: Training stage ('stage1' or 'stage2')
            metric_value: Value of the metric
            metric_name: Name of the metric
            is_best: Whether this is the best checkpoint so far
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'stage': stage,
            'metric_value': metric_value,
            'metric_name': metric_name,
            'fold': self.fold,
            'experiment_name': self.experiment_name
        }
        
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
                checkpoint_path = self.fold_dir / f"best_{stage}_epoch_{epoch}.pth"
        else:
            checkpoint_path = self.fold_dir / f"{stage}_epoch_{epoch}.pth"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update last checkpoint path
        if checkpoint_path != self.last_checkpoint_path:
            shutil.copy2(checkpoint_path, self.last_checkpoint_path)
        
        # Track saved checkpoints
        self.saved_checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'stage': stage,
            'metric_value': metric_value,
            'is_best': is_best
        })
        
        # Clean up old checkpoints if needed
        if not self.keep_best_only:
            self._cleanup_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            print(f"  -> New best {stage} model (epoch {epoch}, {metric_name}: {metric_value:.4f})")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       stage: Optional[str] = None,
                       load_best: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            stage: Stage to load best checkpoint for ('stage1' or 'stage2')
            load_best: Whether to load the best checkpoint
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif stage and load_best:
            if stage == 'stage1':
                path = self.best_s1_path
            elif stage == 'stage2':
                path = self.best_s2_path
            else:
                print(f"Unknown stage: {stage}")
                return None
        else:
            path = self.last_checkpoint_path
        
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return None
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            print(f"Checkpoint loaded: {path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}")
            return None
    
    def load_model_state(self, 
                        model: torch.nn.Module,
                        checkpoint_path: Optional[str] = None,
                        stage: Optional[str] = None,
                        strict: bool = True) -> bool:
        """
        Load model state from checkpoint.
        
        Args:
            model: Model to load state into
            checkpoint_path: Specific checkpoint path to load
            stage: Stage to load best checkpoint for
            strict: Whether to strictly enforce key matching
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint = self.load_checkpoint(checkpoint_path, stage)
        if checkpoint is None:
            return False
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print(f"Model state loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model state: {e}")
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
        if checkpoint is None:
            return False
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Optimizer state loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading optimizer state: {e}")
            return False
    
    def get_best_checkpoint_info(self, stage: str) -> Optional[Dict[str, Any]]:
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
            return None
        
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
                        path.unlink()
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
                        path.unlink()
                        self.saved_checkpoints.remove(checkpoint_info)
    
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
            'best_s1_metric': self.best_s1_metric,
            'best_s2_metric': self.best_s2_metric,
            'saved_checkpoints': self.saved_checkpoints
        })
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.fold_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'stage': checkpoint.get('stage', 'unknown'),
                    'metric_value': checkpoint.get('metric_value', 'unknown'),
                    'metric_name': checkpoint.get('metric_name', 'unknown')
                })
            except Exception as e:
                print(f"Error reading checkpoint {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get('epoch', 0))


class MultiStageCheckpointManager:
    """Manages checkpoints across multiple training stages."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 experiment_name: str,
                 fold: int = 0):
        """
        Initialize multi-stage checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            experiment_name: Name of the experiment
            fold: Current fold number
        """
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, experiment_name, fold
        )
        self.stage_history = []
    
    def save_stage_checkpoint(self,
                            model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            epoch: int,
                            stage: str,
                            metric_value: float,
                            is_best: bool = False) -> str:
        """
        Save checkpoint for a specific stage.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            stage: Training stage
            metric_value: Metric value
            is_best: Whether this is the best for this stage
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            stage=stage,
            metric_value=metric_value,
            is_best=is_best,
            additional_info={'stage_history': self.stage_history}
        )
        
        # Update stage history
        stage_info = {
            'stage': stage,
            'epoch': epoch,
            'metric_value': metric_value,
            'is_best': is_best,
            'checkpoint_path': checkpoint_path
        }
        self.stage_history.append(stage_info)
        
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
        return self.checkpoint_manager.load_model_state(
            model=model,
            stage=previous_stage,
            strict=True
        )
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stages.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_stages': len(set(info['stage'] for info in self.stage_history)),
            'stage_history': self.stage_history,
            'best_s1_info': self.checkpoint_manager.get_best_checkpoint_info('stage1'),
            'best_s2_info': self.checkpoint_manager.get_best_checkpoint_info('stage2')
        }