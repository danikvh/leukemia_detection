"""
Multi-stage trainer that orchestrates Stage 1 and Stage 2 training.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging

from segmentation.training.core.stage1_trainer import Stage1Trainer
from segmentation.training.core.stage2_trainer import Stage2Trainer
from segmentation.training.config.base_config import BaseTrainingConfig
from segmentation.training.config.stage1_config import Stage1Config
from segmentation.training.config.stage2_config import Stage2Config
from segmentation.training.utils.checkpoint_manager import CheckpointManager
from segmentation.training.utils.gpu_monitoring import GPUMonitor
from segmentation.training.utils.metrics_tracker import MetricsTracker


class MultiStageTrainer:
    """Orchestrates both Stage 1 and Stage 2 training."""
    
    def __init__(
        self,
        model: nn.Module,
        stage1_config: Stage1Config,
        stage2_config: Stage2Config,
        device: torch.device,
        output_dir: str,
        fold: int = 0
    ):
        self.model = model
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.device = device
        self.output_dir = output_dir
        self.fold = fold
        
        # Setup trainers
        self.stage1_trainer = Stage1Trainer(
            model, stage1_config, device, output_dir, fold
        )
        self.stage2_trainer = Stage2Trainer(
            model, stage2_config, device, output_dir, fold
        )
        
        # Utilities
        self.checkpoint_manager = CheckpointManager(output_dir, fold)
        self.gpu_monitor = GPUMonitor() if stage1_config.debug else None
        self.metrics_tracker = MetricsTracker(self.output_dir)
        
        # Logger
        self.logger = self._setup_logger()
        
        # Training results
        self.stage1_results = None
        self.stage2_results = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for multi-stage trainer."""
        logger = logging.getLogger(f"MultiStageTrainer_fold_{self.fold}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Execute both Stage 1 and Stage 2 training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Combined training results from both stages
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting Multi-Stage Training")
        self.logger.info("=" * 70)
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("Before multi-stage training")
        
        # Stage 1: CellFinder head and backbone training
        self.stage1_results = self._run_stage1(train_loader, val_loader)
        
        # Clear GPU cache between stages
        torch.cuda.empty_cache()
        
        # Stage 2: SAM neck fine-tuning
        self.stage2_results = self._run_stage2(train_loader, val_loader)
        
        # Finalize training
        results = self._finalize_training()
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("After multi-stage training")
        
        self.logger.info("Multi-stage training completed successfully")
        
        return results
    
    def _run_stage1(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader]
    ) -> Dict[str, Any]:
        """Run Stage 1 training."""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 1: CellFinder Head and Backbone Training")
        self.logger.info("=" * 50)
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("Before Stage 1")
        
        # Train Stage 1
        stage1_results = self.stage1_trainer.train(train_loader, val_loader)
        
        # Finalize Stage 1 (weight transfer, load best model, etc.)
        self.stage1_trainer.finalize_stage()
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("After Stage 1")
        
        self.logger.info(
            f"Stage 1 completed - Best epoch: {stage1_results['best_epoch']}, "
            f"Best loss: {stage1_results['best_loss']:.6f}"
        )
        
        return stage1_results
    
    def _run_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader]
    ) -> Dict[str, Any]:
        """Run Stage 2 training."""
        self.logger.info("=" * 50)
        self.logger.info("STAGE 2: SAM Neck Fine-tuning")
        self.logger.info("=" * 50)
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("Before Stage 2")
        
        # Train Stage 2
        stage2_results = self.stage2_trainer.train(train_loader, val_loader)
        
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("After Stage 2")
        
        self.logger.info(
            f"Stage 2 completed - Best epoch: {stage2_results['best_epoch']}, "
            f"Best loss: {stage2_results['best_loss']:.6f}"
        )
        
        return stage2_results
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize multi-stage training and prepare results."""
        # Load best Stage 2 model for final inference
        stage2_best_path = self.stage2_trainer.checkpoint_manager.get_best_model_path()

        if os.path.exists(stage2_best_path):
            self.logger.info(f"Loading best Stage 1 model from {stage2_best_path}")
            self.checkpoint_manager.load_model_state(self.model, stage2_best_path)
        else:
            self.logger.warning("No best Stage 1 model found, using current state")

        # Combine results
        combined_results = {
            'stage1': self.stage1_results,
            'stage2': self.stage2_results,
            'total_epochs': (
                self.stage1_results['final_epoch'] + 
                self.stage2_results['final_epoch']
            ),
            'final_model_path': stage2_best_path,
            'fold': self.fold,
            'device': str(self.device)
        }
        
        # Save combined results
        self._save_training_summary(combined_results)
        
        return combined_results
    
    def _save_training_summary(self, results: Dict[str, Any]) -> None:
        """Save training summary to file."""
        import json
        
        summary_dir = os.path.join(self.output_dir, f"fold_{self.fold+1}", "training_summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_path = os.path.join(summary_dir, "multi_stage_results.json")
        
        # Convert tensors to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"Training summary saved to {summary_path}")
    
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
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for saving or further processing."""
        return {
            'model_state_dict': self.model.state_dict(),
            'stage1_config': self.stage1_config.__dict__,
            'stage2_config': self.stage2_config.__dict__,
            'stage1_results': self.stage1_results,
            'stage2_results': self.stage2_results,
            'fold': self.fold
        }
    
    def load_model_state(self, state_dict_path: str) -> None:
        """Load model state from checkpoint."""
        checkpoint = torch.load(state_dict_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Model state loaded from {state_dict_path}")
        else:
            # Assume it's just the state dict
            self.model.load_state_dict(checkpoint)
            self.logger.info(f"Model state loaded from {state_dict_path}")
    
    def evaluate_model(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            eval_loader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating trained model...")
        
        self.model.eval()
        eval_losses = {}
        
        with torch.no_grad():
            for batch in eval_loader:
                # Use Stage 2 trainer for evaluation (final stage)
                outputs, targets = self.stage2_trainer.forward_pass(batch)
                loss_dict = self.stage2_trainer.compute_loss(outputs, targets)
                
                for key, value in loss_dict.items():
                    if key not in eval_losses:
                        eval_losses[key] = 0
                    eval_losses[key] += value.item()
        
        # Average losses
        for key in eval_losses:
            eval_losses[key] /= len(eval_loader)
        
        self.logger.info("Evaluation completed")
        for key, value in eval_losses.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        return eval_losses
    
    def print_training_summary(self) -> None:
        """Print a comprehensive training summary."""
        if not self.stage1_results or not self.stage2_results:
            self.logger.warning("Training not completed yet")
            return
        
        print("\n" + "=" * 80)
        print("MULTI-STAGE TRAINING SUMMARY")
        print("=" * 80)
        
        print(f"Fold: {self.fold + 1}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.output_dir}")
        
        print("\nSTAGE 1 RESULTS:")
        print("-" * 40)
        print(f"  Strategy: {self.stage1_config.training_strategy}")
        print(f"  Epochs: {self.stage1_results['final_epoch']}")
        print(f"  Best Epoch: {self.stage1_results['best_epoch']}")
        print(f"  Best Loss: {self.stage1_results['best_loss']:.6f}")
        
        print("\nSTAGE 2 RESULTS:")
        print("-" * 40)
        print(f"  Epochs: {self.stage2_results['final_epoch']}")
        print(f"  Best Epoch: {self.stage2_results['best_epoch']}")
        print(f"  Best Loss: {self.stage2_results['best_loss']:.6f}")
        
        total_epochs = self.stage1_results['final_epoch'] + self.stage2_results['final_epoch']
        print(f"\nTOTAL TRAINING EPOCHS: {total_epochs}")
        print("=" * 80)

    def finalize_with_evaluation(self, test_dataset, metrics_to_optimize=None, is_deep_model=False):
        """Perform final evaluation after training."""
        evaluation_config = {
            'evaluation_methods': ['deepcell', 'coco'],
            'iou_threshold': 0.5,
            'save_outputs': True
        }
        
        final_results = self.metrics_tracker.perform_final_evaluation(
            model=self.model,
            test_dataset=test_dataset,
            metrics_to_optimize=metrics_to_optimize or ['AP50'],
            is_deep_model=is_deep_model,
            evaluation_config=evaluation_config
        )
        
        return final_results