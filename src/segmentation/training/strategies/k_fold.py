"""
K-fold cross-validation strategy.
"""
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import random

from .training_strategy import TrainingStrategy, DataSplitMixin


class KFoldStrategy(TrainingStrategy, DataSplitMixin):
    """K-fold cross-validation training strategy."""
    
    def __init__(self, 
                 k: int = 5, 
                 val_ratio: float = 0.3, 
                 random_seed: int = 23,
                 use_multiple_datasets: bool = False):
        super().__init__(name=f"{k}_fold_cv")
        self.k = k
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.use_multiple_datasets = use_multiple_datasets
    
    def prepare_data_splits(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset,
                          **kwargs) -> List[Tuple[Dataset, Dataset]]:
        """
        Prepare k-fold cross-validation splits.
        
        Args:
            train_dataset: Training dataset to be split
            val_dataset: Validation dataset
            **kwargs: Additional parameters
            
        Returns:
            List of k (train_split, val_split) tuples
        """
        total_samples = len(train_dataset)
        val_size = int(round(total_samples * self.val_ratio))
        train_size = total_samples - val_size
        
        assert train_size + val_size == total_samples, "Split sizes do not match dataset size"
        
        indices = list(range(total_samples))
        random.seed(self.random_seed)
        
        print(f"K-Fold CV (k={self.k}) - Total: {total_samples}, Train: {train_size}, Val: {val_size}")
        
        splits = []
        
        # Generate k different random splits
        for fold in range(self.k):
            split_indices = random.sample(indices, total_samples)
            val_idx = split_indices[:val_size]
            train_idx = split_indices[val_size:]
            
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)
            
            splits.append((train_subset, val_subset))
        
        return splits
    
    def execute_training(self, 
                        trainer,
                        data_splits: List[Tuple[Dataset, Dataset]],
                        config,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Execute k-fold cross-validation training.
        
        Args:
            trainer: Multi-stage trainer instance
            data_splits: List of k (train, val) splits
            config: Training configuration
            **kwargs: Additional parameters
            
        Returns:
            List of k training results
        """
        print("="*60)
        print(f"K-FOLD CROSS-VALIDATION TRAINING (k={self.k})")
        print(f"Validation ratio: {self.val_ratio}")
        print(f"Random seed: {self.random_seed}")
        print("="*60)
        
        if len(data_splits) != self.k:
            raise ValueError(f"Expected {self.k} data splits, got {len(data_splits)}")
        
        results = []
        
        for fold_idx, (train_subset, val_subset) in enumerate(data_splits):
            print(f"\n{'#'*20} FOLD {fold_idx + 1}/{self.k} {'#'*20}")
            
            current_train_subset = train_subset
            
            # Optionally add additional datasets
            if self.use_multiple_datasets:
                print("Adding additional TNBC dataset...")
                try:
                    from datasets.datasets import get_datasets
                    from datasets.transforms import FullTransform
                    
                    # Create transform (you might want to pass this from config)
                    transform = FullTransform(
                        normalize=config.normalize,
                        rgb_transform=config.rgb_transform,
                        stain_transform=config.stain_transform,
                        eosin=config.eosin,
                        dab=config.dab,
                        inversion=config.inversion,
                        only_nuclei=config.only_nuclei,
                        gamma=config.gamma,
                        debug=False
                    )
                    
                    additional_dataset = get_datasets(
                        img_folder_path="../../data/TNBC_dataset/images",
                        mask_folder_path="../../data/TNBC_dataset/masks", 
                        transform=transform,
                        do_augmentation=config.augmentation,
                        complex_augmentation=config.complex_augmentation
                    )
                    
                    current_train_subset = ConcatDataset([additional_dataset, train_subset])
                    print(f"Combined training set size: {len(current_train_subset)}")
                    
                except Exception as e:
                    print(f"Warning: Could not load additional dataset: {e}")
                    current_train_subset = train_subset
            
            # Create data loaders
            from datasets.cellsam_datasets import get_cellsam_dataloaders
            
            dataloaders = get_cellsam_dataloaders(
                {'train': current_train_subset, 'val': val_subset}, 
                batch_size=config.batch_size,
                shuffle=True
            )
            
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
            
            print(f"Fold {fold_idx + 1} - Training samples: {len(current_train_subset)}")
            print(f"Fold {fold_idx + 1} - Validation samples: {len(val_subset)}")
            
            # Execute training for this fold
            fold_result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                val_dataset=val_subset,
                fold_idx=fold_idx,
                config=config
            )
            
            results.append(fold_result)
            
            # Clean up memory
            del train_loader, val_loader
            torch.cuda.empty_cache()
        
        self.results = results
        
        # Save individual fold results
        import os
        import json
        output_dir = f"../output/{config.output_name}/results"
        os.makedirs(output_dir, exist_ok=True)
        
        for fold_idx, result in enumerate(results):
            with open(f"{output_dir}/fold_{fold_idx + 1}_results.json", "w") as f:
                json.dump(result, f, indent=4)
        
        # Save aggregated results
        aggregated = self.aggregate_results()
        with open(f"{output_dir}/k_fold_aggregated_results.json", "w") as f:
            json.dump(aggregated, f, indent=4)
        
        # Print summary
        self.print_summary()
        
        print(f"K-fold training completed. Results saved to {output_dir}")
        
        return self.results