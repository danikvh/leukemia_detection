from pathlib import Path
from typing import Union, Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader
import logging
from segmentation.datasets.cell_dataset import CellDataset
from .utils import PathUtils

logger = logging.getLogger(__name__)

class DatasetFactory:
    """Simplified factory for dataset creation."""
    
    @staticmethod
    def create_dataset(
        data_source: Union[str, Path, Tuple[Path, Path]],
        dataset_type: str = "cell",
        mask_suffix: str = "_masks",
        **kwargs
    ) -> CellDataset:
        """Create dataset from various data sources."""
        
        if isinstance(data_source, (str, Path)):
            # Single folder with paired files
            image_paths, mask_paths, filenames = PathUtils.find_paired_files(
                Path(data_source), mask_suffix
            )
        elif isinstance(data_source, tuple) and len(data_source) == 2:
            # Separate image and mask folders
            image_paths, mask_paths, filenames = PathUtils.find_separated_files(
                data_source[0], data_source[1]
            )
        else:
            raise ValueError("Invalid data_source format")
        
        return CellDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            filenames=filenames,
            **kwargs
        )
    
    @staticmethod
    def create_dataloaders(
        datasets: Dict[str, CellDataset], 
        batch_size: int = 1, 
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
        """Create dataloaders from datasets."""
        return {
            key: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if key == 'train' else False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            for key, dataset in datasets.items()
        }