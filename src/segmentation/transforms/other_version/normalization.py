import torch
import numpy as np
from skimage import exposure
from typing import Tuple, Optional
from .base import ImageOnlyTransform, ParameterizedTransform
import logging

logger = logging.getLogger(__name__)

class ImageNormalizer(ImageOnlyTransform, ParameterizedTransform):
    """
    Normalizes image values to a specific target range and converts to target dtype.
    Assumes input values are typically in [0, 255] or [0, 1].
    """
    def __init__(self, target_range: Tuple[float, float] = (0.0, 1.0), target_dtype: torch.dtype = torch.float32):
        super().__init__(target_range=target_range, target_dtype=target_dtype)
        self.target_range = target_range
        self.target_dtype = target_dtype

    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the image to the target range and converts to target dtype.
        """
        img_np = image.cpu().numpy().astype(np.float32) # Ensure float for calculations

        current_min, current_max = img_np.min(), img_np.max()
        target_min, target_max = self.target_range
        
        if current_max - current_min > 1e-6: # Avoid division by zero
            normalized_np = (img_np - current_min) * ((target_max - target_min) / (current_max - current_min)) + target_min
        else:
            # If image is constant, fill with target_min
            normalized_np = np.full_like(img_np, target_min) 

        normalized_tensor = torch.from_numpy(normalized_np).to(self.target_dtype).to(image.device)
        return normalized_tensor

class ContrastEnhancer(ImageOnlyTransform, ParameterizedTransform):
    """
    Applies contrast enhancement (e.g., adaptive histogram equalization or gamma correction).
    Image values are expected to be in [0, 1] for skimage processing, and will be converted back to original range/dtype.
    """
    def __init__(
        self, 
        method: str = 'adapthist', 
        gamma: Optional[float] = None, 
        clip_limit: float = 0.01, 
        kernel_size: Optional[int] = None
    ):
        super().__init__(
            method=method, gamma=gamma, clip_limit=clip_limit, kernel_size=kernel_size
        )
        self.method = method
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

        if self.method not in ['adapthist', 'gamma']:
            raise ValueError(f"Method must be 'adapthist' or 'gamma', got '{method}'.")
        if self.method == 'gamma' and self.gamma is None:
            raise ValueError("Gamma value must be provided for 'gamma' method.")

    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies contrast enhancement to the image.
        Image is expected to be C, H, W and values are converted to [0,1] float for skimage,
        then converted back to original range/dtype.
        """
        original_dtype = image.dtype
        original_max_val = image.max()
        
        # Convert to numpy HWC and float [0,1] for skimage
        img_np = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if original_max_val > 1.0 + 1e-6: # If originally in [0,255] range
            img_np = img_np / 255.0

        enhanced_np = np.zeros_like(img_np)
        for i in range(img_np.shape[-1]): # Process each channel independently
            if self.method == 'adapthist':
                enhanced_np[:, :, i] = exposure.equalize_adapthist(
                    img_np[:, :, i], 
                    kernel_size=self.kernel_size, 
                    clip_limit=self.clip_limit
                )
            elif self.method == 'gamma':
                enhanced_np[:, :, i] = exposure.adjust_gamma(img_np[:, :, i], gamma=self.gamma)
        
        # Convert back to original tensor format (CHW) and dtype/range
        if original_max_val > 1.0 + 1e-6: # If original was [0,255] float/uint8, scale back
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255)
            
        enhanced_tensor = torch.from_numpy(enhanced_np).permute(2, 0, 1).to(image.device, original_dtype)
        return enhanced_tensor