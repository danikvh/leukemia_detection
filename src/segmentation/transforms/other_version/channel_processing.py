import torch
import torch.nn.functional as F
from typing import Tuple
from .base import ImageOnlyTransform, ParameterizedTransform
import logging

logger = logging.getLogger(__name__)

class ChannelSelector(ImageOnlyTransform, ParameterizedTransform):
    """
    Selects and re-arranges channels for specific cell segmentation tasks.
    Assumes input image is RGB (C=3) and values are in [0, 255] float/uint8.
    """
    def __init__(
        self, 
        nuclear_channel_input: str = "G", # R, G, B for input index 0, 1, 2
        whole_cell_channel_input: str = "B", 
        output_channels: int = 3 # 2 or 3 channels in output (e.g., Cellpose expects 2 or 3)
    ):
        super().__init__(
            nuclear_channel_input=nuclear_channel_input, 
            whole_cell_channel_input=whole_cell_channel_input,
            output_channels=output_channels
        )
        self.channel_map = {"R": 0, "G": 1, "B": 2}
        
        if nuclear_channel_input.upper() not in self.channel_map or \
           whole_cell_channel_input.upper() not in self.channel_map:
            raise ValueError("Nuclear and whole cell input channels must be 'R', 'G', or 'B'.")
        
        self.nuclear_idx = self.channel_map[nuclear_channel_input.upper()]
        self.whole_cell_idx = self.channel_map[whole_cell_channel_input.upper()]
        self.output_channels = output_channels
        
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input image as Tensor with shape (C, H, W).
                   It's assumed to be RGB (C=3) or convertible to it.

        Returns:
            Transformed image tensor with shape (self.output_channels, H, W).
        """
        if image.shape[0] < 3:
            logger.warning(
                f"Input image has {image.shape[0]} channels, expected at least 3 for RGB "
                "channel selection. Attempting to process, but result may be unexpected."
            )
            # Pad with zeros if less than 3 channels, to avoid index errors in indexing self.nuclear_idx
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1) # Grayscale to RGB
            elif image.shape[0] == 2:
                image = F.pad(image, (0, 0, 0, 0, 0, 1), mode='constant', value=0) # Add a black channel

        _, H, W = image.shape
        
        if self.output_channels == 3:
            rgb_output = torch.zeros((3, H, W), dtype=image.dtype, device=image.device)
            rgb_output[1] = image[self.nuclear_idx] # Green channel in output for nuclei
            rgb_output[2] = image[self.whole_cell_idx] # Blue channel in output for whole cell
            return rgb_output
        elif self.output_channels == 2:
            # Cellpose expects (C, H, W) where C=2 for 2-channel images, typically (nuclear, whole_cell)
            output = torch.zeros((2, H, W), dtype=image.dtype, device=image.device)
            output[0] = image[self.nuclear_idx] # First channel for nuclear
            output[1] = image[self.whole_cell_idx] # Second channel for whole cell
            return output
        else:
            raise ValueError(f"Unsupported output_channels: {self.output_channels}. Must be 2 or 3.")

class RGBToBlue(ImageOnlyTransform):
    """
    Transforms an RGB image by setting Red and Green channels to 0, making it blue.
    """
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input image as Tensor with shape (C, H, W).
                   It's assumed to have at least 3 channels.

        Returns:
            Transformed image tensor with Red and Green channels set to 0.
        """
        if image.shape[0] < 3:
            logger.warning(f"Image has {image.shape[0]} channels, RGBToBlue expects at least 3. Returning original.")
            return image
            
        transformed_image = image.clone()
        transformed_image[0] = 0 # Red channel
        transformed_image[1] = 0 # Green channel
        return transformed_image