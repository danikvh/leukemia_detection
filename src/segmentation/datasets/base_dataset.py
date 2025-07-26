from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any, Protocol
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class DataLoader(Protocol):
    """Protocol for data loading operations."""
    def load(self, path: Path) -> Any: ...

class DataProcessor(Protocol):
    """Protocol for data processing operations."""
    def process(self, data: Any, **kwargs) -> Any: ...

class BaseDataset(Dataset, ABC):
    """Enhanced abstract base class for all datasets."""
    
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        filenames: List[str],
        data_loader: Optional[DataLoader] = None,
        data_processor: Optional[DataProcessor] = None,
        transform: Optional[Any] = None,
        precompute: bool = True,
        **kwargs
    ):
        self._validate_inputs(image_paths, mask_paths, filenames)
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.filenames = filenames
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.transform = transform
        self.precompute = precompute
        self._samples = None
        
        if precompute:
            self._precompute_samples()
    
    @staticmethod
    def _validate_inputs(
        image_paths: List[Path], 
        mask_paths: List[Path], 
        filenames: List[str]
    ) -> None:
        """Validate input parameters with enhanced error messages."""
        if not image_paths:
            raise ValueError("Image paths list cannot be empty")
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch between image paths ({len(image_paths)}) "
                f"and mask paths ({len(mask_paths)})"
            )
        
        if len(image_paths) != len(filenames):
            raise ValueError(
                f"Mismatch between image paths ({len(image_paths)}) "
                f"and filenames ({len(filenames)})"
            )
        
        # Validate file existence
        missing_files = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            if not img_path.exists():
                missing_files.append(f"Image: {img_path}")
            if not mask_path.exists():
                missing_files.append(f"Mask: {mask_path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing_files))
    
    def _precompute_samples(self) -> None:
        """Precompute samples if enabled."""
        if not self.precompute:
            return
            
        self._samples = []
        for i, (img_path, mask_path, filename) in enumerate(
            zip(self.image_paths, self.mask_paths, self.filenames)
        ):
            try:
                sample_data = self._process_single_sample(img_path, mask_path, filename, i)
                if isinstance(sample_data, list):
                    self._samples.extend(sample_data)
                else:
                    self._samples.append(sample_data)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                raise
        
        logger.info(f"Precomputed {len(self._samples)} samples")
    
    @abstractmethod
    def _process_single_sample(
        self, 
        img_path: Path, 
        mask_path: Path, 
        filename: str, 
        index: int
    ) -> Union[Tuple, List[Tuple]]:
        """Process a single image-mask pair into dataset samples."""
        pass
    
    def __len__(self) -> int:
        if self.precompute:
            return len(self._samples) if self._samples else 0
        return len(self.image_paths)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get item by index."""
        pass