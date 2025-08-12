import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging

from classification.transforms import get_classification_transforms # Assuming relative import
from classification.utils import (
    load_classification_labels, get_image_paths_and_labels, split_data,
    calculate_pos_weight, calculate_class_weights, visualize_class_distribution
)

logger = logging.getLogger(__name__)

class CellClassificationDataset(Dataset):
    """Dataset for cell classification tasks (binary: cancerous/non-cancerous, ternary: +false-positive)."""

    def __init__(
        self,
        data_samples: List[Tuple[Path, int]], # List of (image_path, label)
        image_size: int,
        mean: Union[Tuple[float, ...], List[float]], # Use Union for flexibility
        std: Union[Tuple[float, ...], List[float]],
        classification_mode: str = "binary",  # NEW: "binary" or "ternary"
        is_train: bool = True
    ):
        """
        Args:
            data_samples (List[Tuple[Path, int]]): A list of tuples, where each tuple
                                                   contains (image_path, label).
            image_size (int): The target size for images (e.g., 96 for 96x96).
            mean (Union[Tuple[float, ...], List[float]]): Mean for normalization.
            std (Union[Tuple[float, ...], List[float]]): Standard deviation for normalization.
            classification_mode (str): Either "binary" or "ternary".
            is_train (bool): Whether this is a training dataset (applies augmentation).
        """
        if not data_samples:
            raise ValueError("data_samples cannot be empty.")

        self.data_samples = data_samples
        self.classification_mode = classification_mode
        
        # Validate labels based on classification mode
        labels = [sample[1] for sample in data_samples]
        unique_labels = set(labels)
        
        if classification_mode == "binary":
            expected_labels = {0, 1}
            self.num_classes = 2
        else:  # ternary
            expected_labels = {0, 1, 2}
            self.num_classes = 3
            
        if not unique_labels.issubset(expected_labels):
            invalid_labels = unique_labels - expected_labels
            raise ValueError(f"Invalid labels for {classification_mode} mode: {invalid_labels}. "
                           f"Expected: {expected_labels}")
        
        # Ensure mean and std are tuples for consistency with torchvision.transforms.Normalize
        self.transform = get_classification_transforms(image_size, tuple(mean), tuple(std), is_train)
        logger.info(f"Initialized CellClassificationDataset with {len(data_samples)} samples in {classification_mode} mode. "
                    f"Augmentation {'ENABLED' if is_train else 'DISABLED'}.")

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: A tuple containing the transformed
            image tensor, the label tensor, and the original filename stem.
        """
        img_path, label = self.data_samples[idx]
        filename_stem = img_path.stem

        try:
            # Load image. Handle .npy or common image formats.
            if img_path.suffix.lower() == '.npy':
                img_np = np.load(img_path)
                # Ensure numpy array is uint8 for PIL conversion
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0: # Assume float [0,1]
                        img_np = (img_np * 255).astype(np.uint8)
                    else: # Assume int or other range, scale to 0-255 if needed
                        img_np = (img_np / img_np.max() * 255).astype(np.uint8)

                if img_np.ndim == 2: # Convert grayscale to RGB for consistency with ImageNet models
                    img_np = np.stack([img_np]*3, axis=-1)
                elif img_np.ndim == 3 and img_np.shape[2] == 4: # Handle RGBA
                    img_np = img_np[:, :, :3] # Discard alpha channel
                elif img_np.ndim == 3 and img_np.shape[2] not in [1, 3]: # Handle potentially (H, W, C) where C is not 1 or 3
                     raise ValueError(f"Unsupported number of channels for .npy image: {img_np.shape[2]} at {img_path}")
                elif img_np.ndim == 3 and img_np.shape[2] == 1: # Convert grayscale (H, W, 1) to RGB
                    img_np = np.repeat(img_np, 3, axis=2)

                img = Image.fromarray(img_np)
            else:
                img = Image.open(img_path).convert("RGB") # Ensure RGB for consistency
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            # Prepare label tensor based on classification mode
            if self.classification_mode == "binary":
                # For binary classification with BCEWithLogitsLoss
                label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            else:
                # For ternary classification with CrossEntropyLoss
                label_tensor = torch.tensor(label, dtype=torch.long)

            return img_tensor, label_tensor, filename_stem

        except Exception as e:
            logger.error(f"Error loading or processing image {img_path.name}: {e}")
            # Instead of re-raising, return a placeholder or handle gracefully in production
            # For this example, we re-raise to quickly catch issues.
            raise

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Returns the class distribution of the dataset.
        
        Returns:
            Dict[int, int]: Dictionary mapping class labels to their counts.
        """
        labels = [sample[1] for sample in self.data_samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    def get_class_names(self) -> List[str]:
        """
        Returns the class names based on classification mode.
        
        Returns:
            List[str]: List of class names.
        """
        if self.classification_mode == "binary":
            return ["non-cancerous", "cancerous"]
        else:  # ternary
            return ["non-cancerous", "cancerous", "false-positive"]

    @staticmethod
    def visualize_class_distribution(data_samples: List[Tuple[Path, int]], 
                                   classification_mode: str = "binary",
                                   title: str = "Class Distribution",
                                   save_path: Optional[Path] = None,
                                   show_plot: bool = True) -> Dict[int, int]:
        """
        Visualize the class distribution of the dataset.
        
        Args:
            data_samples: List of (image_path, label) tuples
            classification_mode: Either "binary" or "ternary"
            title: Title for the plot
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Dict[int, int]: Dictionary mapping class labels to their counts
        """
        labels = [sample[1] for sample in data_samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        
        # Get class names
        if classification_mode == "binary":
            class_names = ["Non-Cancerous", "Cancerous"]
        else:  # ternary
            class_names = ["Non-Cancerous", "Cancerous", "False-Positive"]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        plot_labels = []
        plot_counts = []
        
        for i, name in enumerate(class_names):
            if i in class_distribution:
                plot_labels.append(f"{name}\n({class_distribution[i]} samples)")
                plot_counts.append(class_distribution[i])
            else:
                plot_labels.append(f"{name}\n(0 samples)")
                plot_counts.append(0)
        
        bars = plt.bar(range(len(plot_labels)), plot_counts, 
                      color=['skyblue', 'lightcoral', 'lightgreen'][:len(plot_labels)])
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(range(len(plot_labels)), plot_labels)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Log distribution info
        total_samples = sum(class_distribution.values())
        logger.info(f"Class distribution for {classification_mode} mode:")
        for i, name in enumerate(class_names):
            count = class_distribution.get(i, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.info(f"  {name}: {count} samples ({percentage:.1f}%)")
        
        return class_distribution


class DataLoaderManager:
    """
    Manager class for creating and handling data loaders for cell classification.
    Centralizes data loading logic that was scattered in the main script.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with data loading parameters
        """
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pos_weight = None
        self.class_weights = None
        
    
    def setup_data_loaders(self, data_samples: List[Tuple[Path, int]]) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Set up data loaders for training, validation, and testing.
        
        Args:
            data_samples: List of (image_path, label) tuples
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Visualize overall class distribution
        CellClassificationDataset.visualize_class_distribution(
            data_samples, 
            classification_mode=self.config.classification_mode,
            title=f"Overall Class Distribution ({self.config.classification_mode.title()} Mode)"
        )
        
        # Check if we're in train-only mode
        is_train_only = (self.config.train_split == 1.0 and 
                        self.config.val_split == 0.0 and 
                        self.config.test_split == 0.0)
        
        if is_train_only:
            logger.info("TRAIN-ONLY MODE: Using all data for training (no validation or test sets)")
            self.train_data = data_samples
            self.val_data = []
            self.test_data = []
        else:
            # Split data normally (assuming split_data function is available from utils)
            from classification.utils import split_data
            self.train_data, self.val_data, self.test_data = split_data(
                data_samples,
                self.config.train_split,
                self.config.val_split,
                self.config.test_split,
                self.config.random_seed
            )
        
        logger.info(f"Data splits - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # Calculate class weights
        self._calculate_weights()
        
        # Create datasets
        train_dataset = CellClassificationDataset(
            self.train_data, self.config.image_size, self.config.normalize_mean, self.config.normalize_std,
            classification_mode=self.config.classification_mode, is_train=True
        )
        
        # Only create validation dataset if we have validation data
        val_dataset = None
        if self.val_data:
            val_dataset = CellClassificationDataset(
                self.val_data, self.config.image_size, self.config.normalize_mean, self.config.normalize_std,
                classification_mode=self.config.classification_mode, is_train=False
            )
        
        # Only create test dataset if we have test data
        test_dataset = None
        if self.test_data:
            test_dataset = CellClassificationDataset(
                self.test_data, self.config.image_size, self.config.normalize_mean, self.config.normalize_std,
                classification_mode=self.config.classification_mode, is_train=False
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False,
                num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
            )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False,
                num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
            )
        
        return train_loader, val_loader, test_loader
    
    def _calculate_weights(self):
        """Calculate class weights for handling imbalanced datasets."""
        self.pos_weight = None
        self.class_weights = None
        
        if self.config.use_class_weights and self.train_data:
            if self.config.classification_mode == "binary":
                self.pos_weight = calculate_pos_weight(self.train_data)
            else:  # ternary
                self.class_weights = calculate_class_weights(
                    self.train_data, self.config.num_classes
                )
    
    def get_data_splits(self) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """
        Get the data splits.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.train_data is None:
            raise ValueError("Data loaders must be set up first using setup_data_loaders()")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_class_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get calculated class weights.
        
        Returns:
            Tuple of (pos_weight, class_weights)
        """
        return self.pos_weight, self.class_weights