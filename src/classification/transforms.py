from typing import Tuple, Union
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PadToSquare:
    """Pads an image (PIL Image) to a square shape, preserving aspect ratio."""
    def __init__(self, fill: Union[int, Tuple[int, int, int]] = 0):
        """
        Args:
            fill (int or tuple): Pixel fill value for padded areas. Default is 0 (black).
        """
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        max_dim = max(width, height)
        
        # Ensure fill matches image mode
        if isinstance(self.fill, int) and img.mode in ("RGB", "RGBA"):
            fill_value = (self.fill, self.fill, self.fill)
            if img.mode == "RGBA":
                fill_value = (*fill_value, 255) # Opaque alpha channel
        elif isinstance(self.fill, tuple) and len(self.fill) == 3 and img.mode == "RGB":
            fill_value = self.fill
        elif isinstance(self.fill, tuple) and len(self.fill) == 4 and img.mode == "RGBA":
            fill_value = self.fill
        elif isinstance(self.fill, int) and img.mode == "L": # Grayscale
            fill_value = self.fill
        else:
            logger.warning(f"Unsupported fill value {self.fill} for image mode {img.mode}. Using default black.")
            fill_value = 0 # Default to black for safety
            if img.mode == "RGB": fill_value = (0,0,0)
            if img.mode == "RGBA": fill_value = (0,0,0,255)

        new_img = Image.new(img.mode, (max_dim, max_dim), fill_value)
        new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        return new_img

def get_classification_transforms(
    image_size: int, 
    mean: Tuple[float, float, float], 
    std: Tuple[float, float, float],
    is_train: bool = True
) -> transforms.Compose:
    """
    Returns a composed set of transformations for cell classification.
    
    Args:
        image_size (int): Target size (e.g., 96) for the square image.
        mean (Tuple[float, float, float]): Mean for normalization (e.g., ImageNet mean).
        std (Tuple[float, float, float]): Standard deviation for normalization (e.g., ImageNet std).
        is_train (bool): If True, apply data augmentation for training.
        
    Returns:
        transforms.Compose: Composed transformations.
    """
    common_transforms = [
        PadToSquare(fill=0), # Pad to square with black pixels
        transforms.Resize((image_size, image_size)), # Resize to target square size
        transforms.ToTensor(), # Converts PIL Image to tensor (H, W, C) to (C, H, W) and scales to [0, 1]
        transforms.Normalize(mean=mean, std=std)
    ]

    if is_train:
        # Augmentations for training
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15), # Rotate by +/- 15 degrees
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            *common_transforms # Apply common transforms after augmentations
        ])
    else:
        # No augmentation for validation/test
        return transforms.Compose(common_transforms)