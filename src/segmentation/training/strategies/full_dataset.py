"""
Full dataset training strategy (no validation split).
"""
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset, Subset
import random

from .training_strategy import TrainingStrategy
from segmentation.datasets.dataset_factory import DatasetFactory


class FullDatasetStrategy(TrainingStrategy):
    """Training strategy using the full dataset without validation."""
    
    def __init__(self):
        super().__init__(name="full_dataset")
    
    def prepare_data_splits(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset,
                          **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Prepare full dataset for training (no actual split).
        
        Args:
            train_dataset: Training dataset (will be used fully)
            val_dataset: Validation dataset (used for final evaluation)
            **kwargs: Additional parameters
            
        Returns:
            List with single tuple of (full_train_dataset, val_subset_for_eval)
        """
        # Use full training dataset
        full_train = train_dataset
        
        # Create a random subset of validation dataset for final evaluation
        total_val_samples = len(val_dataset)
        indices = list(range(total_val_samples))
        eval_indices = random.sample(indices, total_val_samples)
        val_subset_for_eval = Subset(val_dataset, eval_indices)
        
        return [(full_train, val_subset_for_eval)]
    
    def execute_training(self, 
                        trainer,
                        data_splits: List[Tuple[Dataset, Dataset]],
                        config,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Execute full dataset training.
        
        Args:
            trainer: Multi-stage trainer instance
            data_splits: List with single (train, val) split
            config: Training configuration
            **kwargs: Additional parameters
            
        Returns:
            List with single training result
        """
        print("="*60)
        print("FULL DATASET TRAINING")
        print("Using full dataset for training. No validation during training.")
        print("="*60)
        
        if len(data_splits) != 1:
            raise ValueError("FullDatasetStrategy expects exactly one data split")
        
        train_dataset, val_dataset_for_eval = data_splits[0]
        
        # Create data loaders
        dataloaders = DatasetFactory().create_dataloaders(
            {'train': train_dataset}, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        train_loader = dataloaders['train']
        val_loader = None  # No validation during training
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(val_dataset_for_eval)}")
        
        # Execute training
        result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset_for_eval,
            fold_idx=0,
            config=config
        )
        
        self.results = [result]
        
        # Save results
        import os
        import json
        output_dir = f"../output/{config.output_name}/results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/full_training_results.json", "w") as f:
            json.dump(result, f, indent=4)
        
        print(f"Full dataset training completed. Results saved to {output_dir}")
        
        return self.results