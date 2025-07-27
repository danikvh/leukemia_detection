"""
Base configuration classes for training pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import json
import yaml
from pathlib import Path
import logging
from common.base_config import BaseConfig


class BaseTrainingConfig(BaseConfig):
    """Base configuration class with common training parameters."""

    def _get_default_config(self) -> Dict[str, Any]:
        """Define default output configuration."""
        return {
            # Training basics
            "training_strategy": "train_val_split", # full_dataset, train_val_split, k_fold_cross_val
            "epochs": 100, 
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,

            # Optimization
            "optimizer": "adamw",  # adamw, adam, sgd
            "scheduler": "plateau",  # plateau, cosine, step, none
            "grad_clip": 1.0,

            # Early stopping
            "patience": 15,
            "min_delta": 1e-6,

            # Data
            "num_workers": 4,
            "pin_memory": True,

            # Logging and output
            "output_name": "cellsam_experiment",
            "output_dir": "../output",
            "debug": True,
            "save_checkpoints": True,
            "save_visualizations": True,
            "save_freq": 25,
            "visualization_frequency": 5,
            "log_freq": 10,
            "preeval": False,
                        
            # Checkpointing
            "keep_best_only": False,
            "keep_last_n": 3,
            
            # Reproducibility
            "random_seed": 42,
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.training_strategy not in ["full_dataset", "train_val_split", "k_fold_cross_val"]:
            raise ValueError(f"training strategy must be one of: full_dataset, train_val_split, k_fold_cross_val")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be between 0 and 1")
        
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        
        if self.optimizer not in ["adamw", "adam", "sgd"]:
            raise ValueError(f"optimizer must be one of: adamw, adam, sgd")
        
        if self.scheduler not in ["plateau", "cosine", "step", "none"]:
            raise ValueError(f"scheduler must be one of: plateau, cosine, step, none")