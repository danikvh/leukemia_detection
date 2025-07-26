from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
from .base import PairedTransform, ParameterizedTransform


class PadToSize(PairedTransform, ParameterizedTransform):
    """Pad image and mask to a specific size."""
    
    def __init__(self, target_size: int = 512, mode: str = 'constant', value: float = 0):
        super().__init__(target_size=target_size, mode=mode, value=value)
        self.target_size = target_size
        self.mode = mode
        self.value = value
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape
        
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)
        
        # Center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        image_padded = F.pad(image, padding, mode=self.mode, value=self.value)
        mask_padded = F.pad(mask, padding, mode='constant', value=0)
        
        return image_padded, mask_padded


class TileExtractor(PairedTransform, ParameterizedTransform):
    """Extract overlapping tiles from image and mask."""
    
    def __init__(self, tile_size: int = 512, overlap_ratio: float = 0.25):
        super().__init__(tile_size=tile_size, overlap_ratio=overlap_ratio)
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap_ratio))
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return lists of image tiles and mask tiles."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        _, H, W = image.shape
        
        # Calculate padding needed
        pad_h_total = self._calculate_padding(H, self.tile_size, self.stride)
        pad_w_total = self._calculate_padding(W, self.tile_size, self.stride)
        
        # Apply symmetric padding
        image, mask = self._apply_symmetric_padding(image, mask, pad_h_total, pad_w_total)
        
        # Extract tiles
        image_tiles, mask_tiles = self._extract_tiles(image, mask)
        
        return image_tiles, mask_tiles
    
    def _calculate_padding(self, dim_size: int, tile_size: int, stride: int) -> int:
        """Calculate total padding needed for a dimension."""
        if dim_size <= tile_size:
            return 0
        return (stride - (dim_size - tile_size) % stride) % stride
    
    def _apply_symmetric_padding(self, image: torch.Tensor, mask: torch.Tensor, 
                                pad_h_total: int, pad_w_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply symmetric padding to center the image."""
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        image = F.pad(image, padding, mode='constant', value=0)
        mask = F.pad(mask, padding, mode='constant', value=0)
        
        return image, mask
    
    def _extract_tiles(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract tiles from padded image and mask."""
        _, padded_H, padded_W = image.shape
        image_tiles, mask_tiles = [], []
        
        for y in range(0, padded_H - self.tile_size + 1, self.stride):
            for x in range(0, padded_W - self.tile_size + 1, self.stride):
                img_tile = image[:, y:y + self.tile_size, x:x + self.tile_size]
                mask_tile = mask[:, y:y + self.tile_size, x:x + self.tile_size]
                image_tiles.append(img_tile)
                mask_tiles.append(mask_tile)
        
        return image_tiles, mask_tiles


class Resize(PairedTransform, ParameterizedTransform):
    """Resize image and mask to target size."""
    
    def __init__(self, size: Tuple[int, int], interpolation_mode: str = 'bilinear'):
        super().__init__(size=size, interpolation_mode=interpolation_mode)
        self.size = size
        self.interpolation_mode = interpolation_mode
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add batch dimension if not present
        if image.ndim == 3:
            image = image.unsqueeze(0)
            remove_batch_dim = True
        else:
            remove_batch_dim = False
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        
        # Resize
        resized_image = F.interpolate(
            image, size=self.size, mode=self.interpolation_mode, align_corners=False
        )
        resized_mask = F.interpolate(
            mask.float(), size=self.size, mode='nearest'
        ).long()
        
        # Remove batch dimension if added
        if remove_batch_dim:
            resized_image = resized_image.squeeze(0)
            resized_mask = resized_mask.squeeze(0)
        
        return resized_image, resized_mask


class CropToContent(PairedTransform):
    """Crop image and mask to remove empty borders."""
    
    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)
        self.margin = margin
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find bounding box of non-zero content
        if mask.ndim == 3:
            mask_2d = mask.squeeze(0)
        else:
            mask_2d = mask
        
        # Find coordinates of non-zero pixels
        nonzero_coords = torch.nonzero(mask_2d, as_tuple=False)
        
        if len(nonzero_coords) == 0:
            # No content, return original
            return image, mask
        
        min_y, min_x = nonzero_coords.min(dim=0)[0]
        max_y, max_x = nonzero_coords.max(dim=0)[0]
        
        # Add margin
        h, w = mask_2d.shape
        min_y = max(0, min_y - self.margin)
        min_x = max(0, min_x - self.margin)
        max_y = min(h, max_y + self.margin + 1)
        max_x = min(w, max_x + self.margin + 1)
        
        # Crop
        cropped_image = image[:, min_y:max_y, min_x:max_x]
        cropped_mask = mask[:, min_y:max_y, min_x:max_x] if mask.ndim == 3 else mask[min_y:max_y, min_x:max_x]
        
        return cropped_image, cropped_mask