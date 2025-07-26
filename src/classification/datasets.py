import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from PIL import Image
import numpy as np
import logging

from classification.transforms import get_classification_transforms # Assuming relative import

logger = logging.getLogger(__name__)

class CellClassificationDataset(Dataset):
    """Dataset for cell classification tasks (cancerous/non-cancerous)."""

    def __init__(
        self,
        data_samples: List[Tuple[Path, int]], # List of (image_path, label)
        image_size: int,
        mean: Union[Tuple[float, ...], List[float]], # Use Union for flexibility
        std: Union[Tuple[float, ...], List[float]],
        is_train: bool = True
    ):
        """
        Args:
            data_samples (List[Tuple[Path, int]]): A list of tuples, where each tuple
                                                   contains (image_path, label).
            image_size (int): The target size for images (e.g., 96 for 96x96).
            mean (Tuple[float, float, float]): Mean for normalization.
            std (Tuple[float, float, float]): Standard deviation for normalization.
            is_train (bool): Whether this is a training dataset (applies augmentation).
        """
        if not data_samples:
            raise ValueError("data_samples cannot be empty.")

        self.data_samples = data_samples
        
        # Ensure mean and std are tuples for consistency with torchvision.transforms.Normalize
        self.transform = get_classification_transforms(image_size, tuple(mean), tuple(std), is_train)
        logger.info(f"Initialized CellClassificationDataset with {len(data_samples)} samples. "
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
            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0) # For BCEWithLogitsLoss

            return img_tensor, label_tensor, filename_stem

        except Exception as e:
            logger.error(f"Error loading or processing image {img_path.name}: {e}")
            # Instead of re-raising, return a placeholder or handle gracefully in production
            # For this example, we re-raise to quickly catch issues.
            raise