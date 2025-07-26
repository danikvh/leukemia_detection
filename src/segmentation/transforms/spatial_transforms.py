import torch
import torch.nn.functional as F
from typing import Tuple

class TileTransform:
    def __init__(self, tile_size: int = 512, overlap_ratio: float = 0.25):
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap_ratio))

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[list, list]:
        """
        Args:
            image (Tensor): (C, H, W)
            mask (Tensor): (1, H, W) or (H, W)

        Returns:
            List of image tiles and corresponding mask tiles, each of shape (C, tile_size, tile_size)
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        _, H, W = image.shape
        tile_size = self.tile_size
        stride = self.stride

        pad_h_total = (stride - (H - tile_size) % stride) % stride if H > tile_size else max(0, tile_size - H)
        pad_w_total = (stride - (W - tile_size) % stride) % stride if W > tile_size else max(0, tile_size - W)

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        _, padded_H, padded_W = image.shape
        image_tiles, mask_tiles = [], []

        for y in range(0, padded_H - tile_size + 1, stride):
            for x in range(0, padded_W - tile_size + 1, stride):
                img_tile = image[:, y:y + tile_size, x:x + tile_size]
                mask_tile = mask[:, y:y + tile_size, x:x + tile_size]
                image_tiles.append(img_tile)
                mask_tiles.append(mask_tile)

        return image_tiles, mask_tiles
    
class PadTo512Both:
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads an image and mask symmetrically to a minimum size of 512x512.
        If dimensions are already >= 512, no padding is applied.
        """
        _, h, w = img.shape

        pad_h = max(0, 512 - h)
        pad_w = max(0, 512 - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (pad_left, pad_right, pad_top, pad_bottom)

        img_padded = F.pad(img, padding, mode="constant", value=0)
        mask_padded = F.pad(mask, padding, mode="constant", value=0)

        return img_padded, mask_padded