from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import torch
from skimage import io
from scipy.ndimage import label
import logging

logger = logging.getLogger(__name__)

class UnifiedDataLoader:
    """Unified data loader for various file formats."""
    
    SUPPORTED_EXTENSIONS = {'.npy', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    
    @classmethod
    def load(cls, path: Path) -> np.ndarray:
        """Load data from file with unified interface."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        try:
            if suffix == '.npy':
                return np.load(path)
            else:
                return io.imread(path)
        except Exception as e:
            raise IOError(f"Error loading file {path}: {str(e)}")

class UnifiedDataProcessor:
    """Unified data processor for images and masks."""
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to appropriate format."""
        if img.dtype in [np.float32, np.float64]:
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            else:
                return np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            return np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    @staticmethod
    def process_mask(mask: np.ndarray, convert_binary: bool = True) -> np.ndarray:
        """Process mask with optional binary-to-labeled conversion."""
        if not convert_binary:
            return mask
            
        unique_vals = np.unique(mask)
        
        # Check if it's a binary mask
        if len(unique_vals) == 2 and 0 in unique_vals:
            logger.debug("Converting binary mask to labeled mask")
            fg_val = unique_vals[unique_vals != 0][0]
            binary_mask = (mask == fg_val).astype(np.uint8)
            labeled_mask, _ = label(binary_mask)
            return labeled_mask
        
        return mask
    
    @staticmethod
    def to_tensor(
        data: np.ndarray, 
        dtype: torch.dtype = torch.float32,
        add_channel_dim: bool = True,
        permute_channels: bool = True
    ) -> torch.Tensor:
        """Convert numpy array to tensor with proper dimensions."""
        tensor = torch.from_numpy(data).to(dtype)
        
        # Handle dimensions
        if tensor.ndim == 2 and add_channel_dim:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in [3, 4] and permute_channels:
            tensor = tensor.permute(2, 0, 1)
            
            # Handle alpha channel
            if tensor.shape[0] == 4 and torch.all(tensor[3] == tensor[3][0, 0]):
                tensor = tensor[:3, :, :]
                logger.debug("Removed alpha channel")
        
        return tensor
    
    def process_image_mask_pair(
        self, 
        img_path: Path, 
        mask_path: Path,
        normalize_image: bool = True,
        convert_binary_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process image-mask pair with unified pipeline."""
        # Load data
        img = UnifiedDataLoader.load(img_path)
        mask = UnifiedDataLoader.load(mask_path)
        
        # Process image
        if normalize_image:
            img = self.normalize_image(img)
        img_tensor = self.to_tensor(img, torch.float32)
        
        # Process mask
        if convert_binary_mask:
            mask = self.process_mask(mask)
        mask_tensor = self.to_tensor(mask, torch.long, permute_channels=False)
        
        return img_tensor, mask_tensor