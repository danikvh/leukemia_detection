"""
WSI Processor Module

This module handles processing of Whole Slide Images (WSI) for patch extraction
and annotation processing.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, PngImagePlugin
from shapely.geometry import Polygon
import torch

from data.config import DataProcessingConfig
from common.utilities import ensure_dir
from data.annotation_extractor import AnnotationExtractor

logger = logging.getLogger(__name__)


class WSIProcessor:
    """Processes WSI files for patch extraction and annotation handling."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the WSI processor.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config or DataProcessingConfig()
        self.annotation_processor = AnnotationExtractor()
        
    def extract_patches(
        self,
        svs_file: str,
        output_dir: str,
        qp_img: Any,
        patch_size: int = 512,
        stride: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        bbox_threshold: float = 0.4,
        generate_mask: bool = False,
        use_rois: bool = False,
        save_masks: bool = True
    ) -> None:
        """
        Extract patches from a WSI file.
        
        Args:
            svs_file: Path to the SVS file
            output_dir: Directory to save extracted patches
            qp_img: QuPath image entry
            patch_size: Size of each patch
            stride: Stride between patches (defaults to patch_size)
            model: Model for mask generation (if generate_mask=True)
            bbox_threshold: Threshold for bounding box detection
            generate_mask: Whether to generate masks using model
            use_rois: Whether to extract patches centered on ROIs
            save_masks: Whether to save mask outputs
        """
        try:
            stride = stride or patch_size
            self._setup_output_directories(output_dir, save_masks)
            
            slide = openslide.OpenSlide(svs_file)
            file_name = self._get_filename(svs_file)
            
            logger.info(f"Processing {file_name}")
            
            # Get patch coordinates
            coordinates = self._get_patch_coordinates(
                slide, qp_img, patch_size, stride, use_rois
            )
            
            # Process each patch
            for idx, (x, y) in enumerate(coordinates):
                self._process_patch(
                    slide, qp_img, x, y, patch_size, output_dir, file_name,
                    model, bbox_threshold, generate_mask, save_masks
                )
            
            slide.close()
            logger.info(f"Completed processing {len(coordinates)} patches")
            
        except Exception as e:
            logger.error(f"Error processing WSI: {e}")
            raise Exception(f"Failed to process WSI: {e}")
    
    def _setup_output_directories(self, output_dir: str, save_masks: bool) -> None:
        """Create necessary output directories."""
        directories = ["images"]
        
        if save_masks:
            directories.extend(["overlay", "masks/img", "masks/data"])
        
        for dir_name in directories:
            ensure_dir(os.path.join(output_dir, dir_name))
    
    def _get_filename(self, svs_file: str) -> str:
        """Extract clean filename from SVS path."""
        return os.path.basename(svs_file).replace(".svs", "").replace(" ", "_")
    
    def _get_patch_coordinates(
        self,
        slide: openslide.OpenSlide,
        qp_img: Any,
        patch_size: int,
        stride: int,
        use_rois: bool
    ) -> List[Tuple[int, int]]:
        """Get coordinates for patch extraction."""
        width, height = slide.dimensions
        
        if use_rois:
            return self._get_roi_patch_coords(qp_img, patch_size, width, height)
        else:
            return self._get_grid_patch_coords(width, height, patch_size, stride)
    
    def _get_grid_patch_coords(
        self,
        width: int,
        height: int,
        patch_size: int,
        stride: int
    ) -> List[Tuple[int, int]]:
        """Get grid-based patch coordinates."""
        coordinates = []
        
        # Calculate padding if stride < patch_size
        if stride < patch_size:
            padding = (patch_size - stride) // 2
            padded_width = width + 2 * padding
            padded_height = height + 2 * padding
        else:
            padding = 0
            padded_width, padded_height = width, height
        
        for i in range((padded_width - patch_size) // stride + 1):
            for j in range((padded_height - patch_size) // stride + 1):
                x = i * stride - padding
                y = j * stride - padding
                coordinates.append((x, y))
        
        return coordinates
    
    def _get_roi_patch_coords(
        self,
        qp_img: Any,
        patch_size: int,
        image_width: int,
        image_height: int
    ) -> List[Tuple[int, int]]:
        """Get ROI-centered patch coordinates."""
        coordinates = []
        
        for annotation in qp_img.hierarchy.annotations:
            roi = annotation.roi
            if not isinstance(roi, Polygon):
                continue
                
            centroid = roi.centroid
            x = int(centroid.x - patch_size // 2)
            y = int(centroid.y - patch_size // 2)
            
            # Ensure patch stays within bounds
            x = max(0, min(x, image_width - patch_size))
            y = max(0, min(y, image_height - patch_size))
            
            coordinates.append((x, y))
        
        return coordinates
    
    def _process_patch(
        self,
        slide: openslide.OpenSlide,
        qp_img: Any,
        x: int,
        y: int,
        patch_size: int,
        output_dir: str,
        file_name: str,
        model: Optional[torch.nn.Module],
        bbox_threshold: float,
        generate_mask: bool,
        save_masks: bool
    ) -> None:
        """Process a single patch."""
        out_filename = os.path.join(output_dir, f"images/{file_name}_{x}_{y}.png")
        
        # Skip if already exists
        if os.path.exists(out_filename):
            logger.debug(f"File exists, skipping: {out_filename}")
            return
        
        # Extract patch
        img = slide.read_region((max(0, x), max(0, y)), 0, (patch_size, patch_size))
        img_array = np.array(img)
        
        # Skip if mostly white/empty
        if np.mean(img_array) > 240:
            logger.debug(f"Skipping patch {x}_{y} (mostly white)")
            return
        
        # Generate or extract masks
        if generate_mask and model is not None:
            mask, mask_np = self._generate_mask_with_model(
                img, model, bbox_threshold, f"{file_name}_{x}_{y}"
            )
            if mask_np is None:
                logger.debug(f"No mask generated for patch {x}_{y}")
                return
        else:
            mask, mask_np = self.annotation_processor.extract_annotations(
                qp_img, x, y, patch_size, patch_size
            )
        
        # Skip if no annotations/masks
        if not save_masks and np.mean(mask_np) == 0:
            logger.debug(f"Skipping patch {x}_{y} (no annotations)")
            return
        
        # Save patch
        self._save_patch(
            img, mask, mask_np, out_filename, output_dir, 
            file_name, x, y, save_masks
        )
    
    def _generate_mask_with_model(
        self,
        img: Image.Image,
        model: torch.nn.Module,
        bbox_threshold: float,
        filename: str
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Generate mask using a trained model."""
        try:
            from segmentation.inference.inference_engine import create_inference_engine, InferenceEngine
            from segmentation.transforms.composed_transforms import FullTransform
            
            # Apply transforms
            transform = FullTransform(
                normalize=False, rgb_transform=False,
                stain_transform=True, eosin=0.0, dab=1.0,
                inversion=False, only_nuclei=False, gamma=2.1,
                debug=False
            )
            
            image_tensor = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)
            tiles, _ = transform(image_tensor, image_tensor)
            image = tiles[0]
            
            # Run inference
            inference_engine = create_inference_engine(model, bbox_threshold)
            result = inference_engine.segment_single_image(image, filename)
            mask_np = result.mask

            if result.success is False:
                return None, None
            
            mask = Image.fromarray((mask_np.squeeze() * 255).astype(np.uint8)).convert("RGB")
            return mask, mask_np
            
        except Exception as e:
            logger.error(f"Error generating mask: {e}")
            return None, None
    
    def _save_patch(
        self,
        img: Image.Image,
        mask: Image.Image,
        mask_np: np.ndarray,
        out_filename: str,
        output_dir: str,
        file_name: str,
        x: int,
        y: int,
        save_masks: bool
    ) -> None:
        """Save patch and associated data."""
        # Create metadata
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("x", str(x))
        metadata.add_text("y", str(y))
        
        # Convert and save image
        img_rgb = img.convert("RGB")
        img_p = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=256)
        img_p.save(out_filename, pnginfo=metadata)
        
        if save_masks:
            # Save overlay
            overlay = Image.blend(img_rgb, mask, alpha=0.3)
            overlay.save(
                os.path.join(output_dir, f"overlay/{file_name}_{x}_{y}.png"),
                pnginfo=metadata
            )
            
            # Save mask
            mask.save(
                os.path.join(output_dir, f"masks/img/{file_name}_{x}_{y}.png"),
                pnginfo=metadata
            )
            
            # Save individual label masks if 3D
            if mask_np.ndim == 3:
                self._save_label_masks(mask_np, output_dir, file_name, x, y)
            
            # Save numpy mask
            np.save(
                os.path.join(output_dir, f"masks/data/{file_name}_{x}_{y}.npy"),
                mask_np
            )
        
        logger.debug(f"Saved patch {file_name}_{x}_{y}")
    
    def _save_label_masks(
        self,
        mask_np: np.ndarray,
        output_dir: str,
        file_name: str,
        x: int,
        y: int
    ) -> None:
        """Save individual label masks."""
        n_labels, height, width = mask_np.shape
        
        for lab in range(n_labels):
            if np.mean(mask_np[lab]) == 0:
                continue
            
            img = Image.fromarray(mask_np[lab] * 255)
            img.save(
                os.path.join(
                    output_dir, 
                    f"masks/img/{file_name}_{x}_{y}_label_{lab}.png"
                )
            )


class PatchQualityFilter:
    """Filters patches based on quality metrics."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
    
    def is_patch_valid(self, patch: np.ndarray) -> bool:
        """
        Check if a patch meets quality criteria.
        
        Args:
            patch: Patch image as numpy array
            
        Returns:
            True if patch is valid, False otherwise
        """
        # Check if mostly white/empty
        if np.mean(patch) > self.config.white_threshold:
            return False
        
        # Check variance (too uniform)
        if np.var(patch) < self.config.variance_threshold:
            return False
        
        # Additional quality checks can be added here
        return True
    
    def filter_patches(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        """Filter a list of patches based on quality."""
        return [patch for patch in patches if self.is_patch_valid(patch)]


class WSIMetadataExtractor:
    """Extracts metadata from WSI files."""
    
    @staticmethod
    def extract_metadata(svs_file: str) -> Dict[str, Any]:
        """
        Extract metadata from WSI file.
        
        Args:
            svs_file: Path to SVS file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            slide = openslide.OpenSlide(svs_file)
            
            metadata = {
                'filename': os.path.basename(svs_file),
                'dimensions': slide.dimensions,
                'level_count': slide.level_count,
                'level_dimensions': slide.level_dimensions,
                'level_downsamples': slide.level_downsamples,
                'properties': dict(slide.properties)
            }
            
            slide.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    @staticmethod
    def get_optimal_level(
        svs_file: str, 
        target_mpp: float = 0.25
    ) -> Tuple[int, float]:
        """
        Get optimal pyramid level for target microns per pixel.
        
        Args:
            svs_file: Path to SVS file
            target_mpp: Target microns per pixel
            
        Returns:
            Tuple of (level, actual_mpp)
        """
        try:
            slide = openslide.OpenSlide(svs_file)
            
            # Get base MPP
            try:
                base_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            except (KeyError, ValueError):
                logger.warning("MPP not found in slide properties, using default")
                base_mpp = 0.25
            
            # Find best level
            best_level = 0
            best_diff = float('inf')
            
            for level in range(slide.level_count):
                level_mpp = base_mpp * slide.level_downsamples[level]
                diff = abs(level_mpp - target_mpp)
                
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
            
            actual_mpp = base_mpp * slide.level_downsamples[best_level]
            slide.close()
            
            return best_level, actual_mpp
            
        except Exception as e:
            logger.error(f"Error finding optimal level: {e}")
            return 0, 0.25