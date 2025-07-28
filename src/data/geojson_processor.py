"""
GeoJSON Processor Module

This module handles processing of GeoJSON annotation files,
extracting instances and converting between formats.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import shape, box, mapping
from shapely.affinity import affine_transform

from data.config import DataProcessingConfig
from common.utilities import ensure_dir

logger = logging.getLogger(__name__)


class GeoJSONProcessor:
    """Processes GeoJSON files for annotation handling."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        """
        Initialize the GeoJSON processor.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config or DataProcessingConfig()
    
    def load_geojson(self, geojson_path: str) -> List[Dict[str, Any]]:
        """
        Load features from a GeoJSON file.
        
        Args:
            geojson_path: Path to the GeoJSON file
            
        Returns:
            List of feature dictionaries
            
        Raises:
            GeoJSONProcessingError: If the file cannot be loaded
        """
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'features' not in data:
                raise Exception("GeoJSON file does not contain 'features' key")
            
            logger.info(f"Loaded {len(data['features'])} features from {geojson_path}")
            return data['features']
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
        except FileNotFoundError:
            raise Exception(f"GeoJSON file not found: {geojson_path}")
        except Exception as e:
            raise Exception(f"Error loading GeoJSON: {e}")
    
    def extract_instances(
        self,
        features: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None,
        output_dir: str = "instances",
        save_crop: bool = True,
        save_mask: bool = True,
        min_area: float = 10.0
    ) -> pd.DataFrame:
        """
        Extract individual instances from GeoJSON features.
        
        Args:
            features: List of GeoJSON features
            image: Original image array (optional)
            output_dir: Directory to save extracted instances
            save_crop: Whether to save cropped images
            save_mask: Whether to save binary masks
            min_area: Minimum area threshold for instances
            
        Returns:
            DataFrame with instance information
        """
        try:
            self._setup_instance_directories(output_dir, save_crop, save_mask)
            
            records = []
            valid_features = self._filter_valid_features(features, min_area)
            
            logger.info(f"Processing {len(valid_features)} valid features")
            
            for idx, feature in enumerate(valid_features):
                try:
                    record = self._extract_single_instance(
                        feature, idx, image, output_dir, save_crop, save_mask
                    )
                    
                    if record:
                        records.append(record)
                        
                except Exception as e:
                    logger.warning(f"Error processing feature {idx}: {e}")
                    continue
            
            # Create and save DataFrame
            df = pd.DataFrame(records)
            csv_path = os.path.join(output_dir, "instances.csv")
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Extracted {len(df)} instances to {output_dir}")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting instances: {e}")
            raise Exception(f"Failed to extract instances: {e}")
    
    def _setup_instance_directories(
        self,
        output_dir: str,
        save_crop: bool,
        save_mask: bool
    ) -> None:
        """Create necessary directories for instance extraction."""
        ensure_dir(output_dir)
        
        if save_mask:
            ensure_dir(os.path.join(output_dir, "masks"))
        
        if save_crop:
            ensure_dir(os.path.join(output_dir, "crops"))
    
    def _filter_valid_features(
        self,
        features: List[Dict[str, Any]],
        min_area: float
    ) -> List[Dict[str, Any]]:
        """Filter features based on validity criteria."""
        valid_features = []
        
        for feature in features:
            try:
                geom = shape(feature["geometry"])
                
                # Check basic validity
                if not geom.is_valid:
                    continue
                
                # Check area threshold
                if geom.area < min_area:
                    continue
                
                # Check if geometry has valid bounds
                bounds = geom.bounds
                if len(bounds) != 4 or any(np.isnan(bounds)):
                    continue
                
                valid_features.append(feature)
                
            except Exception as e:
                logger.debug(f"Invalid feature skipped: {e}")
                continue
        
        return valid_features
    
    def _extract_single_instance(
        self,
        feature: Dict[str, Any],
        idx: int,
        image: Optional[np.ndarray],
        output_dir: str,
        save_crop: bool,
        save_mask: bool
    ) -> Optional[Dict[str, Any]]:
        """Extract a single instance from a feature."""
        try:
            geom = shape(feature["geometry"])
            props = feature.get("properties", {})
            
            # Get label
            label = self._extract_label(props)
            
            # Get bounding box
            minx, miny, maxx, maxy = geom.bounds
            bbox = [int(minx), int(miny), int(maxx), int(maxy)]
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if width <= 0 or height <= 0:
                return None
            
            # Create instance ID
            instance_id = f"instance_{idx:05d}_{label}"
            
            # Create binary mask
            mask = None
            mask_path = None
            if save_mask:
                mask = self._create_polygon_mask(geom, bbox, (width, height))
                mask_path = os.path.join(output_dir, "masks", f"{instance_id}.png")
                Image.fromarray(mask * 255).save(mask_path)
            
            # Extract crop
            crop_path = None
            if image is not None and save_crop:
                crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                crop_path = os.path.join(output_dir, "crops", f"{instance_id}.png")
                Image.fromarray(crop).save(crop_path)
            
            return {
                "id": instance_id,
                "label": label,
                "bbox": bbox,
                "area": geom.area,
                "centroid_x": geom.centroid.x,
                "centroid_y": geom.centroid.y,
                "mask_path": mask_path,
                "crop_path": crop_path
            }
            
        except Exception as e:
            logger.warning(f"Error extracting instance {idx}: {e}")
            return None
    
    def _create_polygon_mask(
        self,
        polygon,
        bbox: List[int],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """Create a binary mask from a shapely polygon."""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Translate coordinates to local space
        translated_coords = [
            (x - bbox[0], y - bbox[1]) 
            for x, y in polygon.exterior.coords
        ]
        
        draw.polygon(translated_coords, outline=1, fill=1)
        return np.array(mask)
    
    def _extract_label(self, properties: Dict[str, Any]) -> str:
        """Extract label from feature properties."""
        # Try different property keys
        label_keys = ["classification", "label", "type", "class", "name"]
        
        for key in label_keys:
            if key in properties:
                label_data = properties[key]
                
                # Handle nested structure
                if isinstance(label_data, dict) and "name" in label_data:
                    return str(label_data["name"])
                elif isinstance(label_data, str):
                    return label_data
        
        return "unknown"
    
    def filter_features_by_class(
        self,
        features: List[Dict[str, Any]],
        class_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter features by class names.
        
        Args:
            features: List of GeoJSON features
            class_names: List of class names to include
            
        Returns:
            Filtered list of features
        """
        filtered = []
        
        for feature in features:
            props = feature.get("properties", {})
            label = self._extract_label(props)
            
            if label.lower() in [name.lower() for name in class_names]:
                filtered.append(feature)
        
        logger.info(f"Filtered {len(features)} -> {len(filtered)} features")
        return filtered
    
    def convert_coordinates(
        self,
        features: List[Dict[str, Any]],
        transform_matrix: Optional[np.ndarray] = None,
        scale_factor: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert coordinates of features using transformation parameters.
        
        Args:
            features: List of GeoJSON features
            transform_matrix: 2x3 affine transformation matrix
            scale_factor: Scale factor for coordinates
            offset: (x, y) offset to apply
            
        Returns:
            Features with transformed coordinates
        """
        transformed_features = []
        
        for feature in features:
            try:
                geom = shape(feature["geometry"])
                
                # Apply transformations
                if transform_matrix is not None:
                    geom = affine_transform(geom, transform_matrix.flatten())
                
                if scale_factor is not None:
                    geom = affine_transform(geom, [scale_factor, 0, 0, scale_factor, 0, 0])
                
                if offset is not None:
                    geom = affine_transform(geom, [1, 0, 0, 1, offset[0], offset[1]])
                
                # Create new feature
                new_feature = feature.copy()
                new_feature["geometry"] = mapping(geom)
                transformed_features.append(new_feature)
                
            except Exception as e:
                logger.warning(f"Error transforming feature: {e}")
                continue
        
        return transformed_features
    
    def merge_geojson_files(
        self,
        geojson_paths: List[str],
        output_path: str,
        remove_duplicates: bool = True
    ) -> None:
        """
        Merge multiple GeoJSON files into one.
        
        Args:
            geojson_paths: List of paths to GeoJSON files
            output_path: Path for output merged file
            remove_duplicates: Whether to remove duplicate features
        """
        try:
            all_features = []
            
            for path in geojson_paths:
                features = self.load_geojson(path)
                all_features.extend(features)
            
            if remove_duplicates:
                all_features = self._remove_duplicate_features(all_features)
            
            # Create merged GeoJSON
            merged_geojson = {
                "type": "FeatureCollection",
                "features": all_features
            }
            
            # Save merged file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_geojson, f, indent=2)
            
            logger.info(f"Merged {len(geojson_paths)} files -> {len(all_features)} features")
            
        except Exception as e:
            logger.error(f"Error merging GeoJSON files: {e}")
            raise Exception(f"Failed to merge files: {e}")
    
    def _remove_duplicate_features(
        self,
        features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate features based on geometry."""
        from shapely.strtree import STRtree
        
        unique_features = []
        geometries = []
        
        # Build spatial index for duplicate detection
        for feature in features:
            try:
                geom = shape(feature["geometry"])
                
                # Check if similar geometry already exists
                if geometries:
                    tree = STRtree(geometries)
                    candidates = tree.query(geom)
                    
                    is_duplicate = False
                    for idx in candidates:
                        if geometries[idx].equals(geom):
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                
                geometries.append(geom)
                unique_features.append(feature)
                
            except Exception as e:
                logger.debug(f"Error processing feature for deduplication: {e}")
                continue
        
        logger.info(f"Removed {len(features) - len(unique_features)} duplicate features")
        return unique_features
    
    def get_feature_statistics(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about features in a GeoJSON.
        
        Args:
            features: List of GeoJSON features
            
        Returns:
            Dictionary containing statistics
        """
        try:
            stats = {
                'total_features': len(features),
                'labels': {},
                'areas': [],
                'centroids': []
            }
            
            for feature in features:
                # Extract label
                props = feature.get("properties", {})
                label = self._extract_label(props)
                
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += 1
                
                # Extract geometry info
                try:
                    geom = shape(feature["geometry"])
                    stats['areas'].append(geom.area)
                    stats['centroids'].append((geom.centroid.x, geom.centroid.y))
                except Exception:
                    continue
            
            # Calculate area statistics
            if stats['areas']:
                areas = np.array(stats['areas'])
                stats['area_stats'] = {
                    'mean': float(np.mean(areas)),
                    'median': float(np.median(areas)),
                    'std': float(np.std(areas)),
                    'min': float(np.min(areas)),
                    'max': float(np.max(areas))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {'error': str(e)}


class GeoJSONValidator:
    """Validates GeoJSON files for correctness and consistency."""
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
    
    def validate_file(self, geojson_path: str) -> Dict[str, Any]:
        """
        Validate a GeoJSON file.
        
        Args:
            geojson_path: Path to GeoJSON file
            
        Returns:
            Validation report
        """
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.validate_geojson(data)
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Failed to load file: {e}"],
                'warnings': [],
                'feature_count': 0
            }
    
    def validate_geojson(self, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GeoJSON data structure.
        
        Args:
            geojson_data: GeoJSON data dictionary
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'feature_count': 0
        }
        
        # Check basic structure
        if geojson_data.get('type') != 'FeatureCollection':
            report['errors'].append("Root type must be 'FeatureCollection'")
            report['valid'] = False
        
        features = geojson_data.get('features', [])
        report['feature_count'] = len(features)
        
        if not isinstance(features, list):
            report['errors'].append("Features must be a list")
            report['valid'] = False
            return report
        
        # Validate each feature
        for i, feature in enumerate(features):
            feature_errors = self._validate_feature(feature, i)
            
            if feature_errors['errors']:
                report['errors'].extend(feature_errors['errors'])
                report['valid'] = False
            
            if feature_errors['warnings']:
                report['warnings'].extend(feature_errors['warnings'])
        
        return report
    
    def _validate_feature(self, feature: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate a single feature."""
        errors = []
        warnings = []
        
        # Check feature structure
        if not isinstance(feature, dict):
            errors.append(f"Feature {index}: Must be an object")
            return {'errors': errors, 'warnings': warnings}
        
        if feature.get('type') != 'Feature':
            errors.append(f"Feature {index}: Type must be 'Feature'")
        
        # Check geometry
        geometry = feature.get('geometry')
        if not geometry:
            errors.append(f"Feature {index}: Missing geometry")
        else:
            geom_errors = self._validate_geometry(geometry, index)
            errors.extend(geom_errors)
        
        # Check properties
        properties = feature.get('properties')
        if properties is None:
            warnings.append(f"Feature {index}: Missing properties")
        elif not isinstance(properties, dict):
            warnings.append(f"Feature {index}: Properties should be an object")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_geometry(self, geometry: Dict[str, Any], feature_index: int) -> List[str]:
        """Validate geometry object."""
        errors = []
        
        if not isinstance(geometry, dict):
            errors.append(f"Feature {feature_index}: Geometry must be an object")
            return errors
        
        geom_type = geometry.get('type')
        if not geom_type:
            errors.append(f"Feature {feature_index}: Geometry missing type")
            return errors
        
        coordinates = geometry.get('coordinates')
        if not coordinates:
            errors.append(f"Feature {feature_index}: Geometry missing coordinates")
            return errors
        
        # Validate specific geometry types
        try:
            if geom_type == 'Polygon':
                self._validate_polygon_coordinates(coordinates, feature_index, errors)
            elif geom_type == 'Point':
                self._validate_point_coordinates(coordinates, feature_index, errors)
            # Add more geometry types as needed
            
        except Exception as e:
            errors.append(f"Feature {feature_index}: Geometry validation error: {e}")
        
        return errors
    
    def _validate_polygon_coordinates(
        self,
        coordinates: List,
        feature_index: int,
        errors: List[str]
    ) -> None:
        """Validate polygon coordinates."""
        if not isinstance(coordinates, list) or not coordinates:
            errors.append(f"Feature {feature_index}: Invalid polygon coordinates")
            return
        
        # Check exterior ring
        exterior_ring = coordinates[0]
        if not isinstance(exterior_ring, list) or len(exterior_ring) < 4:
            errors.append(f"Feature {feature_index}: Polygon exterior ring needs at least 4 points")
            return
        
        # Check if ring is closed
        if exterior_ring[0] != exterior_ring[-1]:
            errors.append(f"Feature {feature_index}: Polygon ring is not closed")
        
        # Validate each coordinate pair
        for i, coord in enumerate(exterior_ring):
            if not isinstance(coord, list) or len(coord) != 2:
                errors.append(f"Feature {feature_index}: Invalid coordinate at position {i}")
                break
            
            if not all(isinstance(c, (int, float)) for c in coord):
                errors.append(f"Feature {feature_index}: Non-numeric coordinate at position {i}")
                break
    
    def _validate_point_coordinates(
        self,
        coordinates: List,
        feature_index: int,
        errors: List[str]
    ) -> None:
        """Validate point coordinates."""
        if (not isinstance(coordinates, list) or 
            len(coordinates) != 2 or 
            not all(isinstance(c, (int, float)) for c in coordinates)):
            errors.append(f"Feature {feature_index}: Invalid point coordinates")