import torch
import numpy as np
from typing import List, Tuple
from cellSAM.utils import normalize_image

from .spatial_transforms import TileTransform, PadTo512Both
from .channel_transforms import ChannelTransform, RGBTransform
from .stain_transforms import StainTransform

class FullTransform:
    """
    Orchestrates a comprehensive image transformation pipeline, including
    channel remapping, stain deconvolution, padding, and tiling.
    """
    def __init__(self, use_channel_transform: bool = False, normalize: bool = False, rgb_transform: bool = False,
                 stain_transform: bool = False, inversion: bool = False, only_nuclei: bool = False,
                 dab: float = 1.0, eosin: float = 0.0, gamma: float = 1.0,
                 nuclear_channel: str = "R", whole_cell_channel: str = "G",
                 tile_size: int = 512, overlap_ratio: float = 0.25, debug: bool = False):
        self.use_channel_transform = use_channel_transform
        self.normalize = normalize
        self.rgb_transform = rgb_transform
        self.stain_transform = stain_transform
        self.eosin = eosin
        self.dab = dab
        self.inversion = inversion
        self.only_nuclei = only_nuclei
        self.gamma = gamma
        self.debug = debug

        self.channel_transform = ChannelTransform(debug=debug, nuclear_channel=nuclear_channel,
                                                  whole_cell_channel=whole_cell_channel)
        self.rgb_transform_instance = RGBTransform()
        self.stain_transform_instance = StainTransform()
        self.pad = PadTo512Both()
        self.tile = TileTransform(tile_size=tile_size, overlap_ratio=overlap_ratio)

    def _normalize_and_convert_tile(self, img_tile: torch.Tensor) -> torch.Tensor:
        """
        Applies CellSAM-like normalization and converts the tile back to [0, 255] uint8 equivalent.
        """
        img_np = img_tile.permute(1, 2, 0).cpu().numpy()
        
        if img_np.max() > 1.0 and img_np.dtype != np.uint8:
            img_np = img_np / 255.0

        img_normalized = normalize_image(img_np)
        
        img_final_np = np.clip(img_normalized * 255.0, 0, 255).astype(np.uint8)
        
        return torch.from_numpy(img_final_np.transpose(2, 0, 1))

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Applies a sequence of transformations to an image and its mask.
        """
        current_img, current_mask = img, mask

        if self.use_channel_transform:
            current_img, current_mask = self.channel_transform(current_img, current_mask)

        if self.rgb_transform:
            current_img, current_mask = self.rgb_transform_instance(current_img, current_mask)

        if self.stain_transform:
            current_img, current_mask = self.stain_transform_instance(
                current_img, current_mask, dab=self.dab, eosin=self.eosin,
                inversion=self.inversion, only_nuclei=self.only_nuclei,
                gamma_value=self.gamma, debug=self.debug
            )
            
        if current_mask.ndim == 2:
            current_mask = current_mask.unsqueeze(0)

        img_padded, mask_padded = self.pad(current_img, current_mask)
        img_tiles, mask_tiles = self.tile(img_padded, mask_padded)

        if self.normalize:
            img_tiles = [self._normalize_and_convert_tile(tile) for tile in img_tiles]
            
        mask_tiles = [tile.long() if tile.dtype != torch.long else tile for tile in mask_tiles]
        mask_tiles = [tile.unsqueeze(0) if tile.ndim == 2 else tile for tile in mask_tiles]

        return img_tiles, mask_tiles