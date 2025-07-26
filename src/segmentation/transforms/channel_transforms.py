import torch
import matplotlib.pyplot as plt # For debug
from typing import Tuple

class ChannelTransform:
    def __init__(self, debug: bool = False, nuclear_channel: str = "R", whole_cell_channel: str = "G"):
        self.debug = debug
        self.nuclear_channel = nuclear_channel
        self.whole_cell_channel = whole_cell_channel

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms input channels to a specific 3-channel RGB format for Cellpose.
        - Nuclear channel mapped to Green (index 1).
        - Whole cell channel mapped to Blue (index 2).
        - Red channel (index 0) is set to zero.

        Args:
            img (torch.Tensor): Input image tensor (C, H, W).
            mask (torch.Tensor): Corresponding mask tensor.
        """
        channels = {"R": 0, "G": 1, "B": 2}
        
        if img.ndim == 2:
            img = img.unsqueeze(0)
            
        C, H, W = img.shape
        rgb_output = torch.zeros((3, H, W), dtype=img.dtype, device=img.device)

        if self.nuclear_channel in channels and channels[self.nuclear_channel] < C:
            rgb_output[1] = img[channels[self.nuclear_channel]]

        if self.whole_cell_channel in channels and channels[self.whole_cell_channel] < C:
            rgb_output[2] = img[channels[self.whole_cell_channel]]

        img = rgb_output
       
        if self.debug:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img_np)
            plt.title("Debug: Adjusted RGB Image (ChannelTransform)")
            plt.axis("off")
            plt.show()

        return img, mask

class RGBTransform:
    def __call__(self, img: torch.Tensor, mask: torch.Tensor, blue: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms a multi-channel image to primarily blue.
        Sets the Red and Green channels to zero, effectively turning the image blue.
        Creates a 3-channel image if the input has fewer channels.
        """
        if blue:
            C, H, W = img.shape
            if C < 3:
                new_img = torch.zeros((3, H, W), dtype=img.dtype, device=img.device)
                if C >= 1:
                    new_img[2] = img[0] # Copy first channel to blue
                img = new_img
            else:
                img[0] = 0 # Red
                img[1] = 0 # Green
        return img, mask