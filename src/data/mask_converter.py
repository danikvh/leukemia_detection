"""
Mask to Annotation Converter Module

This module converts binary masks to GeoJSON annotations, handling
overlap removal and border filtering.
"""

import json
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from data.config import DataProcessingConfig

logger = logging.getLogger(__name__)


class MaskToAnnotationConverter:
    """Converts masks to GeoJSON annotations."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the converter.
        
        Args:
            config: Configuration object containing conversion parameters
        """
        self.config = config or DataProcessingConfig()
        self.patch_size = self.config.patch_size
        self.margin = int(self.patch_size * self.config.border_margin_ratio)
        self.overlap_threshold = self.config.overlap_threshold
    
    def convert_masks_to_geojson(
        self,
        output_dir: str,
        filename: str,
        classification_name: str = "CellSAM Mask",
        classification_color: List[int] = [128, 0, 128]
    ) -> None:
        """
        Convert all masks in output directory to GeoJSON format.
        
        Args:
            output_dir: Directory containing masks
            filename: Base filename for output
            classification_name: Name for the classification
            classification_color: RGB color for the classification
        """
        try:
            images_dir = os.path.join(output_dir, "images")
            masks_dir = os.path.join(output_dir, "masks/data")
            
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                raise Exception(f"Required directories not found in {output_dir}")
            
            # Extract all polygons
            all_geometries, poly_to_coords = self._extract_all_polygons(
                images_dir, masks_dir
            )
            
            # Filter overlapping polygons
            filtered_geometries = self._filter_overlapping_polygons(all_geometries)
            
            logger.info(f"Filtered {len(all_geometries)} -> {len(filtered_geometries)} polygons")
            
            # Create GeoJSON
            geojson = self._create_geojson(
                filtered_geometries, poly_to_coords, 
                classification_name, classification_color
            )
            
            # Save GeoJSON
            output_path = os.path.join(output_dir, f"{filename}.geojson")
            self._save_geojson(geojson, output_path)
            
        except Exception as e:
            logger.error(f"Error converting masks to GeoJSON: {e}")
            raise Exception(f"Failed to convert masks: {e}")
    
    def _extract_all_polygons(
        self,
        images_dir: str,
        masks_dir: str
    ) -> Tuple[List[Polygon], Dict[Polygon, List[List[float]]]]:
        """Extract all polygons from masks."""
        all_geometries = []
        poly_to_coords = {}
        
        for image_filename in sorted(os.listdir(images_dir)):
            if not image_filename.endswith(".png"):
                continue
            
            polygons = self._process_single_mask(
                image_filename, images_dir, masks_dir
            )
            
            for poly, coords in polygons:
                all_geometries.append(poly)
                poly_to_coords[poly] = coords
        
        return all_geometries, poly_to_coords
    
    def _process_single_mask(
        self,
        image_filename: str,
        images_dir: str,
        masks_dir: str
    ) -> List[Tuple[Polygon, List[List[float]]]]:
        """Process a single mask file."""
        base_name = image_filename.replace(".png", "")
        image_path = os.path.join(images_dir, image_filename)
        mask_path = os.path.join(masks_dir, f"{base_name}.npy")
        
        if not os.path.exists(mask_path):
            logger.debug(f"Skipping {base_name}: mask not found")
            return []
        
        # Get patch coordinates and bounds
        x_offset, y_offset = self._read_image_metadata(image_path)
        patch_bbox = self._get_patch_bbox(x_offset, y_offset)
        
        # Load and process mask
        mask_np = np.load(mask_path)
        if np.max(mask_np) == 0:
            return []
        
        polygons = []
        unique_labels = np.unique(mask_np)
        unique_labels = unique_labels[unique_labels != 0]
        
        for label in unique_labels:
            single_mask = (mask_np == label).astype(np.uint8)
            label_polygons = self._mask_to_polygons(single_mask)
            
            for poly_coords in label_polygons:
                abs_coords = [
                    [x + x_offset, y + y_offset] for x, y in poly_coords
                ]
                
                try:
                    poly = Polygon(abs_coords)
                    
                    if not self._is_valid_polygon(poly, patch_bbox):
                        continue
                        
                    polygons.append((poly, abs_coords))
                    
                except Exception as e:
                    logger.debug(f"Invalid polygon skipped: {e}")
                    continue
        
        return polygons
    
    def _read_image_metadata(self, image_path: str) -> Tuple[int, int]:
        """Read x, y coordinates from image metadata."""
        img = Image.open(image_path)
        metadata = img.info
        x = int(metadata.get("x", 0))
        y = int(metadata.get("y", 0))
        return x, y
    
    def _get_patch_bbox(self, x_offset: int, y_offset: int) -> Tuple[int, int, int, int]:
        """Get patch bounding box."""
        return (
            x_offset, y_offset,
            x_offset + self.patch_size,
            y_offset + self.patch_size
        )
    
    def _mask_to_polygons(self, mask_np: np.ndarray) -> List[List[List[int]]]:
        """Convert binary mask to polygon coordinates."""
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            
            poly = contour[:, 0, :].tolist()
            if poly[0] != poly[-1]:
                poly.append(poly[0])  # Ensure closed polygon
            
            polygons.append(poly)
        
        return polygons
    
    def _is_valid_polygon(
        self,
        poly: Polygon,
        patch_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if polygon is valid for inclusion."""
        # Check basic validity
        if not poly.is_valid or poly.area < self.config.min_polygon_area:
            return False
        
        # Check if inside safe zone (away from borders)
        if not self._is_inside_safe_zone(poly.bounds, patch_bbox):
            return False
        
        return True
    
    def _is_inside_safe_zone(
        self,
        poly_bbox: Tuple[float, float, float, float],
        patch_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if polygon is inside the safe zone (away from patch borders)."""
        x_min, y_min, x_max, y_max = poly_bbox
        patch_x_min, patch_y_min, patch_x_max, patch_y_max = patch_bbox
        
        # Define safe zone boundaries
        safe_x_min = patch_x_min + self.margin
        safe_y_min = patch_y_min + self.margin
        safe_x_max = patch_x_max - self.margin
        safe_y_max = patch_y_max - self.margin
        
        # Check if polygon is fully inside safe zone
        return (
            x_min >= safe_x_min and y_min >= safe_y_min and
            x_max <= safe_x_max and y_max <= safe_y_max
        )
    
    def _filter_overlapping_polygons(
        self,
        geometries: List[Polygon]
    ) -> List[Polygon]:
        """Filter out overlapping polygons, keeping larger ones."""
        if not geometries:
            return []
        
        # Build spatial index
        tree = STRtree(geometries)
        kept = []
        
        for poly in geometries:
            if self._should_keep_polygon(poly, tree):
                kept.append(poly)
        
        return kept
    
    def _should_keep_polygon(
        self,
        poly: Polygon,
        tree: STRtree
    ) -> bool:
        """Determine if a polygon should be kept based on overlaps."""
        overlaps = tree.query(poly)
        
        for idx in overlaps:
            other = tree.geometries[idx]
            
            if poly.equals(other):
                continue
            
            if poly.intersects(other):
                intersection = poly.intersection(other)
                
                if not intersection.is_empty:
                    intersection_area = intersection.area
                    
                    # Calculate overlap ratios
                    poly_overlap_ratio = intersection_area / poly.area
                    other_overlap_ratio = intersection_area / other.area
                    
                    # If significant overlap, keep the larger polygon
                    if (poly_overlap_ratio > self.overlap_threshold or
                        other_overlap_ratio > self.overlap_threshold):
                        
                        if poly.area < other.area:
                            return False
        
        return True
    
    def _create_geojson(
        self,
        geometries: List[Polygon],
        poly_to_coords: Dict[Polygon, List[List[float]]],
        classification_name: str,
        classification_color: List[int]
    ) -> Dict[str, Any]:
        """Create GeoJSON from filtered polygons."""
        features = []
        
        for poly in geometries:
            coords = poly_to_coords.get(poly)
            if not coords:
                continue
            
            feature = {
                "type": "Feature",
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": classification_name,
                        "color": classification_color
                    }
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _save_geojson(self, geojson: Dict[str, Any], output_path: str) -> None:
        """Save GeoJSON to file."""
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved GeoJSON with {len(geojson['features'])} features to {output_path}")


class AnnotationValidator:
    """Validates annotations for quality and consistency."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
    
    def validate_geojson(self, geojson_path: str) -> Dict[str, Any]:
        """
        Validate a GeoJSON file.
        
        Args:
            geojson_path: Path to GeoJSON file
            
        Returns:
            Validation report dictionary
        """
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
            
            report = {
                'valid': True,
                'feature_count': len(data.get('features', [])),
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Validate structure
            if 'type' not in data or data['type'] != 'FeatureCollection':
                report['errors'].append("Invalid GeoJSON structure")
                report['valid'] = False
            
            # Validate features
            areas = []
            for i, feature in enumerate(data.get('features', [])):
                feature_report = self._validate_feature(feature, i)
                
                if feature_report['errors']:
                    report['errors'].extend(feature_report['errors'])
                    report['valid'] = False
                
                if feature_report['warnings']:
                    report['warnings'].extend(feature_report['warnings'])
                
                if feature_report['area']:
                    areas.append(feature_report['area'])
            
            # Calculate statistics
            if areas:
                report['statistics'] = {
                    'mean_area': np.mean(areas),
                    'median_area': np.median(areas),
                    'min_area': np.min(areas),
                    'max_area': np.max(areas),
                    'std_area': np.std(areas)
                }
            
            return report
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Failed to validate GeoJSON: {e}"],
                'warnings': [],
                'statistics': {}
            }
    
    def _validate_feature(self, feature: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate a single feature."""
        report = {
            'errors': [],
            'warnings': [],
            'area': None
        }
        
        # Check basic structure
        if 'geometry' not in feature:
            report['errors'].append(f"Feature {index}: Missing geometry")
            return report
        
        if 'properties' not in feature:
            report['warnings'].append(f"Feature {index}: Missing properties")
        
        # Validate geometry
        try:
            geom = feature['geometry']
            if geom['type'] != 'Polygon':
                report['warnings'].append(f"Feature {index}: Non-polygon geometry")
            
            # Create shapely polygon for validation
            poly = Polygon(geom['coordinates'][0])
            
            if not poly.is_valid:
                report['errors'].append(f"Feature {index}: Invalid polygon geometry")
            else:
                report['area'] = poly.area
                
                if poly.area < self.config.min_polygon_area:
                    report['warnings'].append(f"Feature {index}: Very small polygon")
                
        except Exception as e:
            report['errors'].append(f"Feature {index}: Geometry validation error: {e}")
        
        return report