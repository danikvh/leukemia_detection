from pathlib import Path
from typing import List, Tuple, Optional, Any, Union
import torch
import logging
from .base_dataset import BaseDataset
from .data_handlers import UnifiedDataProcessor
from ..transforms.augmentations import get_cell_augmentations, augment_image_and_mask

logger = logging.getLogger(__name__)

class CellDataset(BaseDataset):
    """Unified cell segmentation dataset."""
    
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        filenames: List[str],
        transform: Optional[Any] = None,
        do_augmentation: bool = False,
        complex_augmentation: bool = False,
        precompute: bool = True,
        **kwargs
    ):
        self.do_augmentation = do_augmentation
        self.complex_augmentation = complex_augmentation
        self.augmentation_pipeline = None
        
        # Initialize data processor
        data_processor = UnifiedDataProcessor()
        
        super().__init__(
            image_paths=image_paths,
            mask_paths=mask_paths,
            filenames=filenames,
            data_processor=data_processor,
            transform=transform,
            precompute=precompute,
            **kwargs
        )
        
        self._setup_augmentation()
    
    def _setup_augmentation(self) -> None:
        """Setup augmentation pipeline."""
        if self.do_augmentation:
            self.augmentation_pipeline = get_cell_augmentations(self.complex_augmentation)
            logger.info("Data augmentation enabled")
        else:
            logger.info("Data augmentation disabled")
    
    def _process_single_sample(
        self, 
        img_path: Path, 
        mask_path: Path, 
        filename: str, 
        index: int
    ) -> Union[Tuple, List[Tuple]]:
        """Process a single image-mask pair."""
        img_tensor, mask_tensor = self.data_processor.process_image_mask_pair(
            img_path, mask_path
        )
        
        if self.transform:
            # Transform returns multiple tiles
            img_tiles, mask_tiles = self.transform(img_tensor, mask_tensor)
            return [
                (img_tile, mask_tile, f"{filename}_tile{j}")
                for j, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles))
            ]
        else:
            return (img_tensor, mask_tensor, filename)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.precompute:
            if idx >= len(self._samples):
                raise IndexError(f"Index {idx} out of range")
            
            img_tile, mask_tile, filename = self._samples[idx]
        else:
            # Dynamic loading
            file_idx = idx
            img_path = self.image_paths[file_idx]
            mask_path = self.mask_paths[file_idx]
            filename = self.filenames[file_idx]
            
            img_tile, mask_tile = self.data_processor.process_image_mask_pair(
                img_path, mask_path
            )
        
        # Apply augmentation
        if self.do_augmentation and self.augmentation_pipeline is not None:
            try:
                img_tile, mask_tile = augment_image_and_mask(
                    img_tile, mask_tile, self.augmentation_pipeline
                )
            except Exception as e:
                logger.warning(f"Augmentation failed for {filename}: {str(e)}")
        
        return img_tile, mask_tile, filename