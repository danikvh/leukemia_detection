#!/usr/bin/env python3
"""
Data Processing CLI Script

This script provides command-line interface for processing WSI files,
extracting annotations, and converting between formats.

Examples:
    - python .\data_processing.py extract-patches --output-dir ../data/lafe/processed/processed_cellsam --svs-dir ../data/lafe/raw/Image --qupath-project ../data/lafe/raw/annotations/project.qpproj --generate-mask

    - python .\data_processing.py --output-dir ../data/lafe/processed/processed_annotations extract-annotations --svs-dir ../data/lafe/raw/Image --geojson-dir ../data/lafe/annotations/iteration2

    - python .\data_processing.py --output-dir ../data/lafe/processed/processed_annotations convert-masks --svs-dir ../data/lafe/raw/Image --geojson-dir ../data/lafe/annotations/iteration2
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from paquo.projects import QuPathProject

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.annotation_extractor import AnnotationExtractor, merge_cell_label_csvs
from data.wsi_processor import WSIProcessor
from data.mask_converter import MaskToAnnotationConverter
from data.config import DataProcessingConfig
from segmentation.utils.model_utils import load_cellsam_model
from common.utilities import setup_logging

logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Leukemia Detection Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract patches from WSI with model-generated masks
    python data_processing.py extract-patches --svs-dir /data/svs --output-dir /data/processed --generate-mask
    
    # Extract annotations from QuPath project and GeoJSON
    python data_processing.py extract-annotations --geojson-dir /data/annotations --output-dir /data/processed
    
    # Convert masks to GeoJSON format
    python data_processing.py convert-masks --input-dir /data/processed --output-dir /data/geojson
    
    # Process GeoJSON to extract instances
    python data_processing.py process-geojson --geojson /data/annotations.geojson --output-dir /data/instances
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory for processed data"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract patches command
    extract_parser = subparsers.add_parser(
        "extract-patches", help="Extract patches from WSI files"
    )
    extract_parser.add_argument(
        "--svs-dir", type=str, help="Directory containing SVS files"
    )
    extract_parser.add_argument(
        "--qupath-project", type=str, help="Path to QuPath project file"
    )
    extract_parser.add_argument(
        "--patch-size", type=int, default=512,
        help="Size of extracted patches"
    )
    extract_parser.add_argument(
        "--stride", type=int, default=None,
        help="Stride between patches (default: same as patch-size)"
    )
    extract_parser.add_argument(
        "--generate-mask", action="store_true",
        help="Generate masks using trained model"
    )
    extract_parser.add_argument(
        "--model-path", type=str, 
        help="Path to trained model for mask generation"
    )
    extract_parser.add_argument(
        "--bbox-threshold", type=float,
        help="Bounding box threshold for model inference"
    )
    extract_parser.add_argument(
        "--use-rois", action="store_true",
        help="Extract patches centered on ROIs instead of grid"
    )
    extract_parser.add_argument(
        "--no-masks", action="store_true",
        help="Skip mask generation/extraction"
    )
    
    # Extract annotations command
    annotations_parser = subparsers.add_parser(
        "extract-annotations", help="Extract annotations from GeoJSON files"
    )
    annotations_parser.add_argument(
        "--svs-dir", type=str,
        help="Directory containing SVS files"
    )
    annotations_parser.add_argument(
        "--geojson-dir", type=str,
        help="Directory containing GeoJSON annotation files"
    )
    annotations_parser.add_argument(
        "--mapping-file", type=str, default=None,
        help="JSON file mapping SVS files to GeoJSON files"
    )
    annotations_parser.add_argument(
        "--cell-padding", type=int, default=0,
        help="Padding around individual cell crops"
    )
    
    return parser


def extract_patches_command(args, config: DataProcessingConfig) -> None:
    """Execute patch extraction command."""
    logger.info("Starting patch extraction")
    
    # Load QuPath project
    qp = QuPathProject(args.qupath_project, mode='r')
    qp_img_list = {img.image_name: i for i, img in enumerate(qp.images)}
    
    # Load model if needed
    model = None
    if args.generate_mask:
        if not args.model_path or not os.path.exists(args.model_path):
            logger.info("Model path not given, using default model")
        
        model, device = load_cellsam_model(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    
    # Process SVS files
    processor = WSIProcessor(config)
    processed_count = 0
    
    for svs_file in Path(args.svs_dir).rglob("*.svs"):
        svs_name = svs_file.name
        
        if svs_name not in qp_img_list:
            logger.warning(f"SVS file not in QuPath project: {svs_name}")
            continue
        
        logger.info(f"Processing {svs_name}")
        qp_img = qp.images[qp_img_list[svs_name]]
        
        output_subdir = os.path.join(
            args.output_dir, 
            f"image_{qp_img_list[svs_name]}_{svs_file.stem}"
        )
        
        try:
            processor.extract_patches(
                str(svs_file), output_subdir, qp_img,
                patch_size=args.patch_size,
                stride=args.stride or args.patch_size,
                model=model,
                bbox_threshold=args.bbox_threshold,
                generate_mask=args.generate_mask,
                use_rois=args.use_rois,
                save_masks=not args.no_masks
            )
            
            # Convert masks to GeoJSON if generated
            if args.generate_mask and not args.no_masks:
                converter = MaskToAnnotationConverter(config)
                converter.convert_masks_to_geojson(
                    output_subdir, svs_file.stem
                )
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {svs_name}: {e}")
            continue
    
    logger.info(f"Successfully processed {processed_count} SVS files")


def extract_annotations_command(args, config: DataProcessingConfig) -> None:
    """Execute annotation extraction command."""
    logger.info("Starting annotation extraction")
    
    # Load file mappings
    file_mappings = load_file_mappings(args.svs_dir, args.geojson_dir, args.mapping_file)
    
    extractor = AnnotationExtractor(config)
    processed_count = 0
    
    for svs_path, geojson_path, image_name in file_mappings:
        if not os.path.exists(svs_path) or not os.path.exists(geojson_path):
            logger.warning(f"Missing files for {image_name}")
            continue
        
        logger.info(f"Processing {image_name}")
        
        try:
            extractor.extract_annotations(
                svs_path, geojson_path, args.output_dir,
                image_name, args.cell_padding
            )
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            continue
    
    # Merge cell labels
    if processed_count > 0:
        merge_cell_label_csvs(args.output_dir)
    
    logger.info(f"Successfully processed {processed_count} annotation sets")

def load_file_mappings(
    svs_dir: str, 
    geojson_dir: str, 
    mapping_file: Optional[str]
) -> List[tuple]:
    """Load file mappings between SVS and GeoJSON files."""
    mappings = []
    
    if mapping_file and os.path.exists(mapping_file):
        # Load from mapping file
        import json
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        for item in mapping_data:
            svs_path = os.path.join(svs_dir, item['svs_file'])
            geojson_path = os.path.join(geojson_dir, item['geojson_file'])
            mappings.append((svs_path, geojson_path, item['image_name']))
    
    else:
        # Auto-discover mappings based on filenames
        svs_files = list(Path(svs_dir).rglob("*.svs"))
        geojson_files = list(Path(geojson_dir).rglob("*.geojson"))
        
        for svs_file in svs_files:
            # Try to find matching GeoJSON file
            svs_stem = svs_file.stem.lower().replace(" ", "").replace("_", "")
            
            matching_geojson = None
            for geojson_file in geojson_files:
                geojson_stem = geojson_file.stem.lower().replace(" ", "").replace("_", "")
                if svs_stem in geojson_stem or geojson_stem in svs_stem:
                    matching_geojson = geojson_file
                    break
            
            if matching_geojson:
                image_name = f"image_{svs_file.stem}"
                mappings.append((str(svs_file), str(matching_geojson), image_name))
            else:
                logger.warning(f"No matching GeoJSON found for {svs_file.name}")
    
    return mappings


def main():
    """Main entry point."""
    partial_parser = argparse.ArgumentParser(add_help=False)
    partial_parser.add_argument("--config", type=str, default=None)
    partial_args, _ = partial_parser.parse_known_args()

    config = DataProcessingConfig()
    if partial_args.config:
        config.load_from_file(partial_args.config)
        config_dict = config.to_dict()
    else:
        config_dict = {}

    parser = setup_argparser()
    args = parser.parse_args()

    # Convert args Namespace → dict
    args_dict = vars(args)

    # Config completely overrides unless CLI explicitly passed something
    for k, v in config_dict.items():
        cli_value = args_dict.get(k)
        # If CLI explicitly passed, keep it. Otherwise use config
        # `None`, `False` for store_true means not explicitly set
        if cli_value in [None, False] and v is not None:
            args_dict[k] = v

    args = argparse.Namespace(**args_dict)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute command
    try:
        if args.command == "extract-patches":
            extract_patches_command(args, config)
        elif args.command == "extract-annotations":
            extract_annotations_command(args, config)
        else:
            parser.print_help()
            sys.exit(1)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()