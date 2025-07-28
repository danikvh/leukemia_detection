"""
Annotation Extractor Module

This module handles extraction of annotations from QuPath projects and GeoJSON files,
creating masks and individual cell crops for leukemia detection.
"""

import os
import json
import csv
import logging
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union
import shapely.affinity

from data.config import DataProcessingConfig
from common.utilities import ensure_dir

logger = logging.getLogger(__name__)


class AnnotationExtractor:
    """Extracts annotations from WSI files and creates training data."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the annotation extractor.
        
        Args:
            config: Configuration object containing extraction parameters
        """
        self.config = config or DataProcessingConfig()
        self.label_mapping = self._get_label_mapping()
        
    def _get_label_mapping(self) -> Dict[str, str]:
        """Get the label mapping from Spanish to English."""
        return {
            "positivo": "positive",
            "negativo": "negative"
        }
    
    def extract_annotations(
        self,
        svs_path: str,
        geojson_path: str,
        output_dir: str,
        image_name: str,
        cell_crop_padding: int = 0,
        background_value: int = 255,
        use_mask_shape: bool = True
    ) -> None:
        """
        Extract annotations from a WSI and GeoJSON file.
        
        Args:
            svs_path: Path to the SVS file
            geojson_path: Path to the GeoJSON annotation file
            output_dir: Directory to save extracted data
            image_name: Base name for output files
            cell_crop_padding: Padding around cell boundary for context
            background_value: Value to use for background pixels (255=white, 0=black, None=transparent)
                             Only used when use_mask_shape=True
            use_mask_shape: If True, extract cells with exact mask shape. If False, use rectangular crops
        """
        try:
            self._setup_output_directories(output_dir)
            
            # Load data
            slide = openslide.OpenSlide(svs_path)
            geojson_data = self._load_geojson(geojson_path)
            
            # Separate ROIs and cells
            roi_features, cell_features = self._separate_features(geojson_data["features"])
            logger.info(f"Found {len(roi_features)} ROIs and {len(cell_features)} cell instances")
            
            # Process each ROI
            cell_labels = []
            for i, roi_feature in enumerate(roi_features):
                self._process_roi(
                    slide, roi_feature, cell_features, output_dir, 
                    image_name, i, cell_labels, cell_crop_padding, background_value, use_mask_shape
                )
            
            # Save cell labels
            self._save_cell_labels(cell_labels, output_dir, image_name)
            slide.close()
            
        except Exception as e:
            logger.error(f"Error extracting annotations: {e}")
            raise Exception(f"Failed to extract annotations: {e}")
    
    def _setup_output_directories(self, output_dir: str) -> None:
        """Create necessary output directories."""
        directories = ["imgs", "masks_png", "masks_npy", "overlay", "cells"]
        for dir_name in directories:
            ensure_dir(os.path.join(output_dir, dir_name))
    
    def _load_geojson(self, geojson_path: str) -> Dict:
        """Load GeoJSON data from file."""
        try:
            with open(geojson_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load GeoJSON: {e}")
    
    def _separate_features(self, features: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Separate ROI and cell features."""
        roi_features, cell_features = [], []
        
        for feature in features:
            props = feature.get("properties", {})
            label = props.get("classification") or props.get("label") or props.get("type")
            
            if label is None:
                continue
                
            name = label["name"]
            if isinstance(name, str) and name.startswith("Region"):
                roi_features.append(feature)
            else:
                cell_features.append(feature)
        
        return roi_features, cell_features
    
    def _process_roi(
        self,
        slide: openslide.OpenSlide,
        roi_feature: Dict,
        cell_features: List[Dict],
        output_dir: str,
        image_name: str,
        roi_index: int,
        cell_labels: List[Tuple[str, str]],
        cell_crop_padding: int,
        background_value: int,
        use_mask_shape: bool
    ) -> None:
        """Process a single ROI and extract cell data."""
        roi_geom = shape(roi_feature["geometry"])
        patch_np, canvas_size = self._get_slide_patch(slide, roi_geom.bounds)
        
        minx, miny, _, _ = roi_geom.bounds
        roi_geom_local = self._normalize_geometry(roi_geom, (minx, miny))
        roi_mask = self._draw_mask(canvas_size, roi_geom_local, fill=128)
        
        # Create combined mask
        mask_combined = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.int32)
        instance_id = 1
        
        # Process cells within ROI
        for cell_feature in cell_features:
            cell_geom = shape(cell_feature["geometry"])
            
            if roi_geom.intersects(cell_geom):
                self._process_cell(
                    slide, cell_feature, cell_geom, minx, miny,
                    canvas_size, mask_combined, instance_id,
                    output_dir, image_name, roi_index, cell_labels,
                    cell_crop_padding, background_value, use_mask_shape
                )
                instance_id += 1
        
        # Create overlay and save outputs
        overlay = self._create_overlay(patch_np, roi_mask, mask_combined)
        self._save_roi_outputs(
            patch_np, mask_combined, overlay, output_dir, 
            image_name, roi_index
        )
        
        # Display results (optional)
        if self.config.show_results:
            self._show_image_and_mask(patch_np, mask_combined, overlay)
    
    def _get_slide_patch(
        self, 
        slide: openslide.OpenSlide, 
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract a patch from the slide."""
        minx, miny, maxx, maxy = bounds
        width = int(maxx - minx)
        height = int(maxy - miny)
        canvas_size = (width, height)
        
        patch = slide.read_region((int(minx), int(miny)), 0, canvas_size)
        patch = patch.convert("RGB")
        return np.array(patch), canvas_size
    
    def _normalize_geometry(self, geom, bbox_min: Tuple[float, float]):
        """Translate geometry to local coordinate system."""
        return shapely.affinity.translate(geom, xoff=-bbox_min[0], yoff=-bbox_min[1])
    
    def _draw_mask(self, image_size: Tuple[int, int], polygon, fill: int = 1) -> np.ndarray:
        """Create a binary mask for a polygon."""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        if isinstance(polygon, MultiPolygon):
            for p in polygon.geoms:
                draw.polygon(list(p.exterior.coords), fill=fill)
        elif isinstance(polygon, Polygon):
            draw.polygon(list(polygon.exterior.coords), fill=fill)
        
        return np.array(mask)
    
    def _process_cell(
        self,
        slide: openslide.OpenSlide,
        cell_feature: Dict,
        cell_geom: Polygon,
        minx: float,
        miny: float,
        canvas_size: Tuple[int, int],
        mask_combined: np.ndarray,
        instance_id: int,
        output_dir: str,
        image_name: str,
        roi_index: int,
        cell_labels: List[Tuple[str, str]],
        cell_crop_padding: int,
        background_value: int,
        use_mask_shape: bool
    ) -> None:
        """Process a single cell within an ROI."""
        # Create mask for ROI visualization
        cell_geom_local = self._normalize_geometry(cell_geom, (minx, miny))
        cell_mask = self._draw_mask(canvas_size, cell_geom_local, fill=1)
        mask_combined[cell_mask == 1] = instance_id
        
        # Extract and save cell crop
        label = self._get_cell_label(cell_feature["properties"])
        if label != "unknown":
            if use_mask_shape:
                cell_crop = self._extract_masked_cell(
                    slide, cell_geom, cell_crop_padding, background_value
                )
            else:
                cell_crop = self._extract_rectangular_cell(
                    slide, cell_geom, cell_crop_padding
                )
                
            if cell_crop is not None:
                filename = f"{image_name}_roi{roi_index:03d}_cell{instance_id:05d}.png"
                cell_img_path = os.path.join(output_dir, "cells", filename)
                self._save_cell_image_and_label(cell_crop, cell_img_path, label, cell_labels)
        else:
            logger.info(f"Label with unknow value {self._get_cell_label(cell_feature['properties'])}")
    
    def _get_cell_label(self, properties: Dict) -> str:
        """Extract cell label from properties."""
        label_data = properties.get("classification") or properties.get("label") or properties.get("type")
        if not label_data:
            return "unknown"
            
        name = label_data.get("name", "").lower()
        
        for spanish_term, english_term in self.label_mapping.items():
            if spanish_term in name:
                return english_term
        
        logger.warning(f"Unknown label: {name}")
        return "unknown"
    
    def _extract_rectangular_cell(
        self, 
        slide: openslide.OpenSlide, 
        cell_geom: Polygon, 
        padding: int
    ) -> Optional[np.ndarray]:
        """
        Extract a rectangular cell crop (original behavior).
        
        Args:
            slide: OpenSlide object
            cell_geom: Cell geometry (polygon)
            padding: Padding around cell bounding box
            
        Returns:
            Rectangular cell crop or None if extraction fails
        """
        minx, miny, maxx, maxy = cell_geom.bounds
        minx = max(0, int(minx) - padding)
        miny = max(0, int(miny) - padding)
        maxx = int(maxx) + padding
        maxy = int(maxy) + padding
        
        width = maxx - minx
        height = maxy - miny
        
        if width <= 0 or height <= 0:
            return None
        
        patch = slide.read_region((minx, miny), 0, (width, height)).convert("RGB")
        return np.array(patch)

    def _extract_masked_cell(
        self, 
        slide: openslide.OpenSlide, 
        cell_geom: Polygon, 
        padding: int,
        background_value: Optional[int] = 255
    ) -> Optional[np.ndarray]:
        """
        Extract a cell image masked to the exact cell shape.
        
        Args:
            slide: OpenSlide object
            cell_geom: Cell geometry (polygon)
            padding: Padding around cell boundary
            background_value: Background pixel value (255=white, 0=black, None=transparent)
            
        Returns:
            Masked cell image or None if extraction fails
        """
        # Get cell bounds
        cell_minx, cell_miny, cell_maxx, cell_maxy = cell_geom.bounds
        
        # Calculate crop bounds with padding
        crop_minx = max(0, int(cell_minx) - padding)
        crop_miny = max(0, int(cell_miny) - padding)
        crop_maxx = int(cell_maxx) + padding
        crop_maxy = int(cell_maxy) + padding
        
        crop_width = crop_maxx - crop_minx
        crop_height = crop_maxy - crop_miny
        
        if crop_width <= 0 or crop_height <= 0:
            return None
        
        # Extract image patch
        patch = slide.read_region((crop_minx, crop_miny), 0, (crop_width, crop_height)).convert("RGB")
        crop_array = np.array(patch)
        
        # Create mask for the cell shape
        cell_geom_local = shapely.affinity.translate(
            cell_geom, xoff=-crop_minx, yoff=-crop_miny
        )
        cell_mask = self._draw_mask((crop_width, crop_height), cell_geom_local, fill=1)
        
        # Apply mask to create shaped cell image
        if background_value is None:
            # Create RGBA image with transparency
            masked_image = np.zeros((crop_height, crop_width, 4), dtype=np.uint8)
            masked_image[:, :, :3] = crop_array  # RGB channels
            masked_image[:, :, 3] = cell_mask * 255  # Alpha channel (transparency)
        else:
            # Create RGB image with solid background
            masked_image = crop_array.copy()
            # Set background pixels to specified value
            background_pixels = cell_mask == 0
            masked_image[background_pixels] = background_value
        
        return masked_image

    
    def _create_overlay(
        self, 
        patch_np: np.ndarray, 
        roi_mask: np.ndarray, 
        mask_combined: np.ndarray
    ) -> np.ndarray:
        """Create an overlay visualization."""
        overlay = patch_np.copy()
        
        # ROI overlay
        overlay = self._blend_mask_color(
            overlay, roi_mask == 128, 
            color=(200, 200, 255), alpha=0.3
        )
        
        # Cell overlay
        overlay = self._blend_mask_color(
            overlay, mask_combined > 0, 
            color=(255, 0, 0), alpha=0.5
        )
        
        return overlay
    
    def _blend_mask_color(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int, int, int], 
        alpha: float = 0.5
    ) -> np.ndarray:
        """Blend a colored mask on top of an image."""
        overlay = image.copy()
        idx = mask > 0
        overlay[idx] = (alpha * np.array(color) + (1 - alpha) * overlay[idx]).astype(np.uint8)
        return overlay
    
    def _save_roi_outputs(
        self,
        patch_np: np.ndarray,
        mask_combined: np.ndarray,
        overlay: np.ndarray,
        output_dir: str,
        image_name: str,
        roi_index: int
    ) -> None:
        """Save ROI-level outputs."""
        base_name = f"roi_{roi_index:03d}"
        
        # Save image
        Image.fromarray(patch_np).save(
            os.path.join(output_dir, f"imgs/{image_name}_{base_name}_image.png")
        )
        
        # Save binary mask
        Image.fromarray((mask_combined > 0).astype(np.uint8) * 255).save(
            os.path.join(output_dir, f"masks_png/{image_name}_{base_name}_mask.png")
        )
        
        # Save instance mask
        np.save(
            os.path.join(output_dir, f"masks_npy/{image_name}_{base_name}_mask.npy"),
            mask_combined
        )
        
        # Save overlay
        Image.fromarray(overlay).save(
            os.path.join(output_dir, f"overlay/{image_name}_{base_name}_overlay.png")
        )
    
    def _save_cell_image_and_label(
        self, 
        image: np.ndarray, 
        save_path: str, 
        label: str, 
        record_list: List[Tuple[str, str]]
    ) -> None:
        """Save cell image and add to label list."""
        # Handle both RGB and RGBA images
        if image.shape[2] == 4:  # RGBA
            Image.fromarray(image, mode='RGBA').save(save_path)
        else:  # RGB
            Image.fromarray(image).save(save_path)
        record_list.append((os.path.basename(save_path), label))
    
    def _save_cell_labels(
        self, 
        records: List[Tuple[str, str]], 
        output_dir: str, 
        image_name: str
    ) -> None:
        """Save cell labels to CSV."""
        csv_path = os.path.join(output_dir, f"{image_name}_cell_labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(records)
    
    def _show_image_and_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        overlay: Optional[np.ndarray] = None
    ) -> None:
        """Display image and mask (for debugging)."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axs = plt.subplots(1, 3 if overlay is not None else 2, figsize=(12, 4))
            axs[0].imshow(image)
            axs[0].set_title("Patch")
            axs[1].imshow(mask)
            axs[1].set_title("Mask")
            
            if overlay is not None:
                axs[2].imshow(overlay)
                axs[2].set_title("Overlay")
            
            for ax in axs:
                ax.axis('off')
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not available for visualization")


def merge_cell_label_csvs(root_dir: str, output_name: str = "all_cell_labels.csv") -> None:
    """Merge multiple cell label CSVs into a single file."""
    import glob
    import pandas as pd
    
    csv_files = glob.glob(os.path.join(root_dir, "*_cell_labels.csv"))
    
    if not csv_files:
        logger.warning("No CSV files found to merge")
        return
    
    all_labels = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_labels.append(df)
        os.remove(csv_path)  # Clean up individual files
    
    merged_df = pd.concat(all_labels, ignore_index=True)
    output_path = os.path.join(root_dir, output_name)
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"Merged {len(csv_files)} CSVs into {output_path} with {len(merged_df)} entries")