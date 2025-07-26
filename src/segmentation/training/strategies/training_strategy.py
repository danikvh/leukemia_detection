"""
Abstract base class for training strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingStrategy(ABC):
    """Abstract base class for different training strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    @abstractmethod
    def prepare_data_splits(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset,
                          **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Prepare data splits for training.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional parameters
            
        Returns:
            List of (train_split, val_split) tuples
        """
        pass
    
    @abstractmethod
    def execute_training(self, 
                        trainer,
                        data_splits: List[Tuple[Dataset, Dataset]],
                        config,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the training strategy.
        
        Args:
            trainer: Trainer instance
            data_splits: List of (train, val) dataset splits
            config: Training configuration
            **kwargs: Additional parameters
            
        Returns:
            List of training results for each split/fold
        """
        pass
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get training results."""
        return self.results
    
    def save_results(self, save_path: str):
        """Save results to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump({
                'strategy': self.name,
                'results': self.results
            }, f, indent=2)
    
    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all folds/splits.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}
        
        # Aggregate metrics across folds
        aggregated = {}
        
        # Get metric keys from first result
        if self.results and len(self.results) > 0:
            first_result = self.results[0]
            
            for metric_type in ['best_f1', 'best_dice', 'best_ap50']:
                if metric_type in first_result:
                    aggregated[metric_type] = {}
                    
                    for metric_name in first_result[metric_type].keys():
                        # Collect values across all folds
                        values = []
                        for result in self.results:
                            if metric_type in result and metric_name in result[metric_type]:
                                values.append(result[metric_type][metric_name])
                        
                        if values:
                            import numpy as np
                            values_array = np.array(values)
                            aggregated[metric_type][metric_name] = {
                                'mean': np.mean(values_array, axis=0).tolist() if values_array.ndim > 1 else float(np.mean(values_array)),
                                'std': np.std(values_array, axis=0).tolist() if values_array.ndim > 1 else float(np.std(values_array)),
                                'min': np.min(values_array, axis=0).tolist() if values_array.ndim > 1 else float(np.min(values_array)),
                                'max': np.max(values_array, axis=0).tolist() if values_array.ndim > 1 else float(np.max(values_array))
                            }
        
        return aggregated
    
    def print_summary(self):
        """Print a summary of the training results."""
        print(f"\n{'='*50}")
        print(f"Training Strategy: {self.name}")
        print(f"Number of folds/runs: {len(self.results)}")
        print(f"{'='*50}")
        
        aggregated = self.aggregate_results()
        
        for metric_type in ['best_f1', 'best_dice', 'best_ap50']:
            if metric_type in aggregated:
                print(f"\n{metric_type.upper()} Results:")
                print("-" * 30)
                
                for metric_name, stats in aggregated[metric_type].items():
                    if isinstance(stats['mean'], list):
                        mean_val = sum(stats['mean']) / len(stats['mean'])
                        std_val = sum(stats['std']) / len(stats['std'])
                    else:
                        mean_val = stats['mean']
                        std_val = stats['std']
                    
                    print(f"{metric_name:15s}: {mean_val:.4f} ± {std_val:.4f}")


class DataSplitMixin:
    """Mixin class providing common data splitting functionality."""
    
    @staticmethod
    def create_random_split(dataset: Dataset, 
                          train_ratio: float = 0.7,
                          random_seed: int = 42) -> Tuple[Dataset, Dataset]:
        """Create a random train/val split."""
        from torch.utils.data import Subset
        import random
        
        total_samples = len(dataset)
        train_size = int(total_samples * train_ratio)
        val_size = total_samples - train_size
        
        indices = list(range(total_samples))
        random.seed(random_seed)
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        return train_subset, val_subset
    
    @staticmethod
    def create_k_fold_splits(dataset: Dataset, 
                           k: int = 5,
                           random_seed: int = 42) -> List[Tuple[Dataset, Dataset]]:
        """Create k-fold cross-validation splits."""
        from torch.utils.data import Subset
        from sklearn.model_selection import KFold
        import numpy as np
        
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        indices = np.arange(len(dataset))
        
        splits = []
        for train_idx, val_idx in kf.split(indices):
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())
            splits.append((train_subset, val_subset))
        
        return splits