"""
Train/validation split strategy.
"""
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset, Subset
import random

from segmentation.training.strategies.training_strategy import TrainingStrategy, DataSplitMixin
from segmentation.datasets.dataset_factory import DatasetFactory


class TrainValSplitStrategy(TrainingStrategy, DataSplitMixin):
    """Training strategy with a single train/validation split."""
    
    def __init__(
            self, 
            val_ratio: float = 0.3, 
            random_seed: int = 23,
            batch_size: int = 2,
            output_name: str = "results/train_val"):
        super().__init__(name="train_val_split")
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.output_name = output_name
    
    def prepare_data_splits(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset,
                          **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Prepare train/validation split.
        
        Args:
            train_dataset: Training dataset to be split
            val_dataset: Validation dataset (used for creating val split)
            **kwargs: Additional parameters
            
        Returns:
            List with single tuple of (train_split, val_split)
        """
        total_samples = len(train_dataset)
        val_size = int(round(total_samples * self.val_ratio))
        train_size = total_samples - val_size
        
        assert train_size + val_size == total_samples, "Split sizes do not match dataset size"
        
        indices = list(range(total_samples))
        random.seed(self.random_seed)
        
        print(f"Train/Validation split - Total: {total_samples}, Train: {train_size}, Val: {val_size}")
        
        split_indices = random.sample(indices, total_samples)
        val_idx = split_indices[:val_size]
        train_idx = split_indices[val_size:]
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        return [(train_subset, val_subset)]
    
    def execute_training(self, 
                        trainer,
                        data_splits: List[Tuple[Dataset, Dataset]],
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Execute train/val split training.
        
        Args:
            trainer: Multi-stage trainer instance
            data_splits: List with single (train, val) split
            config: Training configuration
            **kwargs: Additional parameters
            
        Returns:
            List with single training result
        """
        print("="*60)
        print("TRAIN/VALIDATION SPLIT TRAINING")
        print(f"Validation ratio: {self.val_ratio}")
        print(f"Random seed: {self.random_seed}")
        print("="*60)
        
        if len(data_splits) != 1:
            raise ValueError("TrainValSplitStrategy expects exactly one data split")
        
        train_subset, val_subset = data_splits[0]
        
        # Create data loaders     
        dataloaders = DatasetFactory().create_dataloaders(
            {'train': train_subset, 'val': val_subset}, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Execute training
        result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            #val_dataset=val_subset
        )
        
        self.results = [result]
        
        # Save results
        import os
        import json
        output_dir = self.output_name
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/train_val_results.json", "w") as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Train/val training completed. Results saved to {output_dir}")
        
        return self.results