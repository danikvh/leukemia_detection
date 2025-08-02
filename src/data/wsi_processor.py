"""
Enhanced WSI Processor Module

This module handles processing of Whole Slide Images (WSI) for patch extraction
and individual cell extraction from generated masks.
"""

import os
import csv
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, PngImagePlugin
from shapely.geometry import Polygon
import torch
import cv2
import shapely.affinity

from data.config import DataProcessingConfig
from common.utilities import ensure_dir
from data.annotation_extractor import AnnotationExtractor

logger = logging.getLogger(__name__)


class WSIProcessor:
    """Enhanced WSI processor that extracts patches and individual cells."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the enhanced WSI processor.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config or DataProcessingConfig()
        self.annotation_processor = AnnotationExtractor()
        
    def extract_patches_and_cells(
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
        save_masks: bool = True,
        cell_crop_padding: int = 10,
        use_mask_shape: bool = True,
        background_value: int = 255,
        min_cell_area: float = 50.0
    ) -> None:
        """
        Extract patches from a WSI file and individual cells from masks.
        
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
            cell_crop_padding: Padding around cell boundary for context
            use_mask_shape: If True, extract cells with exact mask shape
            background_value: Value for background pixels when use_mask_shape=True
            min_cell_area: Minimum cell area to be considered valid
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
            
            # Initialize cell labels list for CSV
            all_cell_labels = []
            
            # Process each patch
            for idx, (x, y) in enumerate(coordinates):
                patch_cell_labels = self._process_patch_with_cells(
                    slide, qp_img, x, y, patch_size, output_dir, file_name,
                    model, bbox_threshold, generate_mask, save_masks,
                    cell_crop_padding, use_mask_shape,
                    background_value, min_cell_area, idx
                )
                
                if patch_cell_labels:
                    all_cell_labels.extend(patch_cell_labels)
            
            # Save consolidated cell labels CSV
            if all_cell_labels:
                self._save_all_cell_labels(all_cell_labels, output_dir, file_name)
            
            slide.close()
            logger.info(f"Completed processing {len(coordinates)} patches")
            
            logger.info(f"Extracted {len(all_cell_labels)} individual cells")
            
        except Exception as e:
            logger.error(f"Error processing WSI: {e}")
            raise Exception(f"Failed to process WSI: {e}")
    
    def _setup_output_directories(self, output_dir: str, save_masks: bool) -> None:
        """Create necessary output directories."""
        directories = ["images"]
        
        if save_masks:
            directories.extend(["overlay", "masks/img", "masks/data"])
        
        directories.append("cells")
        
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
    
    def _process_patch_with_cells(
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
        save_masks: bool,
        cell_crop_padding: int,
        use_mask_shape: bool,
        background_value: int,
        min_cell_area: float,
        patch_idx: int
    ) -> List[Tuple[str, str]]:
        """Process a single patch and extract individual cells."""
        out_filename = os.path.join(output_dir, f"images/{file_name}_{x}_{y}.png")
        
        # Skip if already exists
        if os.path.exists(out_filename):
            logger.debug(f"File exists, skipping: {out_filename}")
            return []
        
        # Extract patch
        img = slide.read_region((max(0, x), max(0, y)), 0, (patch_size, patch_size))
        img_array = np.array(img)
        
        # Skip if mostly white/empty
        if np.mean(img_array) > 240:
            logger.debug(f"Skipping patch {x}_{y} (mostly white)")
            return []
        
        # Generate or extract masks
        if generate_mask and model is not None:
            mask, mask_np = self._generate_mask_with_model(
                img, model, bbox_threshold, f"{file_name}_{x}_{y}"
            )
            if mask_np is None:
                logger.debug(f"No mask generated for patch {x}_{y}")
                return []
        else:
            mask, mask_np = self.annotation_processor.extract_annotations(
                qp_img, x, y, patch_size, patch_size
            )
        
        # Skip if no annotations/masks
        if not save_masks and np.mean(mask_np) == 0:
            logger.debug(f"Skipping patch {x}_{y} (no annotations)")
            return []
        
        # Extract individual cells if requested
        cell_labels = []
        if mask_np is not None and np.max(mask_np) > 0:
            cell_labels = self._extract_cells_from_mask(
                slide, img, mask_np, x, y, output_dir, file_name,
                patch_idx, cell_crop_padding, use_mask_shape,
                background_value, min_cell_area
            )
        
        # Save patch
        self._save_patch(
            img, mask, mask_np, out_filename, output_dir, 
            file_name, x, y, save_masks
        )
        
        return cell_labels
    
    def _extract_cells_from_mask(
        self,
        slide: openslide.OpenSlide,
        patch_img: Image.Image,
        mask_np: np.ndarray,
        patch_x: int,
        patch_y: int,
        output_dir: str,
        file_name: str,
        patch_idx: int,
        cell_crop_padding: int,
        use_mask_shape: bool,
        background_value: int,
        min_cell_area: float
    ) -> List[Tuple[str, str]]:
        """Extract individual cells from a mask."""
        cell_labels = []
        
        # Handle different mask formats
        if mask_np.ndim == 3:
            # Multi-label mask (each channel is a different label)
            height, width, n_labels = mask_np.shape
            combined_mask = np.zeros((height, width), dtype=np.int32)
            
            instance_id = 1
            for label_idx in range(n_labels):
                label_mask = mask_np[:, :, label_idx]
                if np.max(label_mask) > 0:
                    combined_mask[label_mask > 0] = instance_id
                    instance_id += 1
            
            mask_to_process = combined_mask
        else:
            # Single mask or already instance mask
            mask_to_process = mask_np.astype(np.int32)
        
        # Get unique instance IDs
        unique_instances = np.unique(mask_to_process)
        unique_instances = unique_instances[unique_instances > 0]
        
        for instance_id in unique_instances:
            # Create binary mask for this instance
            instance_mask = (mask_to_process == instance_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour_idx, contour in enumerate(contours):
                if len(contour) < 3:
                    continue
                
                # Calculate area
                area = cv2.contourArea(contour)
                if area < min_cell_area:
                    continue
                
                # Create polygon from contour
                contour_points = contour[:, 0, :].tolist()
                if contour_points[0] != contour_points[-1]:
                    contour_points.append(contour_points[0])
                
                try:
                    from shapely.geometry import Polygon
                    
                    # Convert to absolute coordinates
                    abs_contour_points = [
                        [x + patch_x, y + patch_y] for x, y in contour_points
                    ]
                    cell_polygon = Polygon(abs_contour_points)
                    
                    if not cell_polygon.is_valid:
                        continue
                    
                    # Extract cell crop
                    if use_mask_shape:
                        cell_crop = self._extract_masked_cell_from_wsi(
                            slide, cell_polygon, cell_crop_padding, background_value
                        )
                    else:
                        cell_crop = self._extract_rectangular_cell_from_wsi(
                            slide, cell_polygon, cell_crop_padding
                        )
                    
                    if cell_crop is not None:
                        # Generate filename
                        cell_filename = f"{file_name}_patch{patch_idx:03d}_cell{instance_id:03d}_{contour_idx:02d}.png"
                        cell_path = os.path.join(output_dir, "cells", cell_filename)
                        
                        # Save cell image
                        self._save_cell_image(cell_crop, cell_path)
                        
                        # For WSI processing, we don't have ground truth labels
                        # So we mark them as "unknown" or "unlabeled"
                        cell_labels.append((cell_filename, "unlabeled"))
                        
                except Exception as e:
                    logger.debug(f"Error extracting cell: {e}")
                    continue
        
        return cell_labels
    
    def _extract_rectangular_cell_from_wsi(
        self,
        slide: openslide.OpenSlide,
        cell_polygon: Polygon,
        padding: int
    ) -> Optional[np.ndarray]:
        """Extract rectangular cell crop from WSI."""
        minx, miny, maxx, maxy = cell_polygon.bounds
        minx = max(0, int(minx) - padding)
        miny = max(0, int(miny) - padding)
        maxx = int(maxx) + padding
        maxy = int(maxy) + padding
        
        width = maxx - minx
        height = maxy - miny
        
        if width <= 0 or height <= 0:
            return None
        
        try:
            patch = slide.read_region((minx, miny), 0, (width, height)).convert("RGB")
            return np.array(patch)
        except Exception as e:
            logger.debug(f"Error extracting rectangular cell: {e}")
            return None
    
    def _extract_masked_cell_from_wsi(
        self,
        slide: openslide.OpenSlide,
        cell_polygon: Polygon,
        padding: int,
        background_value: int = 255
    ) -> Optional[np.ndarray]:
        """Extract masked cell crop from WSI."""
        cell_minx, cell_miny, cell_maxx, cell_maxy = cell_polygon.bounds
        
        # Calculate crop bounds with padding
        crop_minx = max(0, int(cell_minx) - padding)
        crop_miny = max(0, int(cell_miny) - padding)
        crop_maxx = int(cell_maxx) + padding
        crop_maxy = int(cell_maxy) + padding
        
        crop_width = crop_maxx - crop_minx
        crop_height = crop_maxy - crop_miny
        
        if crop_width <= 0 or crop_height <= 0:
            return None
        
        try:
            # Extract image patch
            patch = slide.read_region((crop_minx, crop_miny), 0, (crop_width, crop_height)).convert("RGB")
            crop_array = np.array(patch)
            
            # Create mask for the cell shape
            cell_geom_local = shapely.affinity.translate(
                cell_polygon, xoff=-crop_minx, yoff=-crop_miny
            )
                
            # Draw mask
            from PIL import ImageDraw
            mask_img = Image.new("L", (crop_width, crop_height), 0)
            draw = ImageDraw.Draw(mask_img)
            
            coords = list(cell_geom_local.exterior.coords)
            draw.polygon(coords, fill=1)
            cell_mask = np.array(mask_img)
            
            # Apply mask
            if background_value is None:
                # Create RGBA image with transparency
                masked_image = np.zeros((crop_height, crop_width, 4), dtype=np.uint8)
                masked_image[:, :, :3] = crop_array  # RGB channels
                masked_image[:, :, 3] = cell_mask * 255  # Alpha channel
            else:
                # Create RGB image with solid background
                masked_image = crop_array.copy()
                background_pixels = cell_mask == 0
                masked_image[background_pixels] = background_value
            
            return masked_image
            
        except Exception as e:
            logger.debug(f"Error extracting masked cell: {e}")
            return None
    
    def _save_cell_image(self, image: np.ndarray, save_path: str) -> None:
        """Save cell image."""
        try:
            # Handle both RGB and RGBA images
            if image.shape[2] == 4:  # RGBA
                Image.fromarray(image, mode='RGBA').save(save_path)
            else:  # RGB
                Image.fromarray(image).save(save_path)
        except Exception as e:
            logger.debug(f"Error saving cell image: {e}")
    
    def _save_all_cell_labels(
        self,
        cell_labels: List[Tuple[str, str]],
        output_dir: str,
        file_name: str
    ) -> None:
        """Save all cell labels to CSV."""
        if not cell_labels:
            return
        
        csv_path = os.path.join(output_dir, f"{file_name}_cell_labels.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "label"])
                writer.writerows(cell_labels)
            
            logger.info(f"Saved {len(cell_labels)} cell labels to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving cell labels: {e}")
    
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
        
        if save_masks and mask is not None:
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
            if mask_np is not None and mask_np.ndim == 3:
                self._save_label_masks(mask_np, output_dir, file_name, x, y)
            
            # Save numpy mask
            if mask_np is not None:
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


def merge_wsi_cell_labels(root_dir: str, output_name: str = "all_wsi_cell_labels.csv") -> None:
    """Merge multiple WSI cell label CSVs into a single file."""
    import glob
    import pandas as pd
    
    csv_files = glob.glob(os.path.join(root_dir, "*_cell_labels.csv"))
    
    if not csv_files:
        logger.warning("No WSI cell label CSV files found to merge")
        return
    
    all_labels = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            all_labels.append(df)
            os.remove(csv_path)  # Clean up individual files
        except Exception as e:
            logger.warning(f"Error processing {csv_path}: {e}")
    
    if all_labels:
        merged_df = pd.concat(all_labels, ignore_index=True)
        output_path = os.path.join(root_dir, output_name)
        merged_df.to_csv(output_path, index=False)
        
        logger.info(f"Merged {len(csv_files)} WSI cell CSVs into {output_path} with {len(merged_df)} entries")