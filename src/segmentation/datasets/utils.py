from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class PathUtils:
    """Utility functions for path operations."""
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.npy')
    
    @classmethod
    def find_paired_files(
        cls,
        folder_path: Path,
        mask_suffix: str = "_masks"
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """Find image-mask pairs in a single folder."""
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_paths = []
        mask_paths = []
        filenames = []
        
        # Get all supported files
        all_files = [
            f for f in folder_path.iterdir() 
            if f.is_file() and f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ]
        
        for img_file in sorted(all_files):
            if mask_suffix in img_file.stem:
                continue
            
            # Look for corresponding mask
            mask_name = f"{img_file.stem}{mask_suffix}{img_file.suffix}"
            mask_path = folder_path / mask_name
            
            if mask_path.exists():
                image_paths.append(img_file)
                mask_paths.append(mask_path)
                filenames.append(img_file.stem)
            else:
                logger.warning(f"No mask found for image: {img_file}")
        
        if not image_paths:
            raise ValueError(f"No valid image-mask pairs found in {folder_path}")
        
        logger.info(f"Found {len(image_paths)} image-mask pairs")
        return image_paths, mask_paths, filenames
    
    @classmethod
    def find_separated_files(
        cls,
        img_folder_path: Path,
        mask_folder_path: Path
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """Find files in separate image and mask folders."""
        img_folder_path = Path(img_folder_path)
        mask_folder_path = Path(mask_folder_path)
        
        if not img_folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {img_folder_path}")
        if not mask_folder_path.exists():
            raise FileNotFoundError(f"Mask folder not found: {mask_folder_path}")
        
        img_files = sorted([
            f for f in img_folder_path.iterdir() 
            if f.is_file() and f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ])
        mask_files = sorted([
            f for f in mask_folder_path.iterdir() 
            if f.is_file() and f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ])
        
        if len(img_files) != len(mask_files):
            raise ValueError(
                f"Mismatch: {len(img_files)} images vs {len(mask_files)} masks"
            )
        
        filenames = [img.stem for img in img_files]
        
        logger.info(f"Found {len(img_files)} image-mask pairs in separate folders")
        return img_files, mask_files, filenames