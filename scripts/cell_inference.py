#!/usr/bin/env python3
"""
Example script for using the enhanced cell classifier for inference.
Demonstrates both binary and ternary classification modes.

Usage:
    python inference_example.py --model-path models/binary_model.pth --config configs/binary_config.yaml --image path/to/image.png
    python inference_example.py --model-path models/ternary_model.pth --config configs/ternary_config.yaml --image-dir path/to/images/
"""

import argparse
import logging
from pathlib import Path
import json
from typing import List, Union

from classification.config import ClassificationConfig
from classification.inference import CellClassifier
from classification.utils import visualize_cell

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def classify_single_image(classifier: CellClassifier, image_path: Path, visualize: bool = True) -> dict:
    """
    Classify a single image and optionally visualize the result.
    
    Args:
        classifier (CellClassifier): The classifier instance.
        image_path (Path): Path to the image.
        visualize (bool): Whether to visualize the result.
        
    Returns:
        dict: Classification result.
    """
    logger.info(f"Classifying image: {image_path}")
    
    try:
        result = classifier.classify_single_image(image_path)
        
        # Log the result
        if classifier.config.classification_mode == "binary":
            logger.info(f"Prediction: {result['prediction']}")
            logger.info(f"Probability: {result['probability']:.4f}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
        else:  # ternary
            logger.info(f"Prediction: {result['prediction']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info("Class probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                logger.info(f"  {class_name}: {prob:.4f}")
        
        # Visualize if requested
        if visualize:
            predicted_label = result['prediction']
            if classifier.config.classification_mode == "binary":
                probability = result['probability']
            else:
                probability = result['probabilities']
            
            visualize_cell(
                image_input=image_path,
                true_label=0,  # We don't know the true label
                filename_stem=image_path.stem,
                predicted_label=predicted_label,
                probability=probability,
                classification_mode=classifier.config.classification_mode
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying image {image_path}: {e}")
        return {"error": str(e)}

def classify_batch_images(classifier: CellClassifier, image_paths: List[Path], 
                         save_results: bool = True, output_path: Path = None) -> List[dict]:
    """
    Classify a batch of images.
    
    Args:
        classifier (CellClassifier): The classifier instance.
        image_paths (List[Path]): List of image paths.
        save_results (bool): Whether to save results to JSON.
        output_path (Path): Path to save results.
        
    Returns:
        List[dict]: List of classification results.
    """
    logger.info(f"Classifying batch of {len(image_paths)} images...")
    
    try:
        results = classifier.classify_batch(image_paths)
        
        # Combine results with filenames
        detailed_results = []
        for image_path, result in zip(image_paths, results):
            detailed_result = {
                "filename": image_path.name,
                "filepath": str(image_path),
                **result
            }
            detailed_results.append(detailed_result)
        
        # Log summary
        if classifier.config.classification_mode == "binary":
            predictions = [r['prediction'] for r in results]
            summary = {pred: predictions.count(pred) for pred in set(predictions)}
        else:  # ternary
            predictions = [r['prediction'] for r in results]
            summary = {pred: predictions.count(pred) for pred in set(predictions)}
        
        logger.info("Classification summary:")
        for pred, count in summary.items():
            logger.info(f"  {pred}: {count} images")
        
        # Save results if requested
        if save_results and output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(detailed_results, f, indent=4)
            logger.info(f"Results saved to {output_path}")
        
        return detailed_results
        
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Cell classification inference")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str,
                           help="Path to single image for classification")
    input_group.add_argument("--image-dir", type=str,
                           help="Directory containing images for batch classification")
    
    # Output options
    parser.add_argument("--output", type=str, default="inference_results.json",
                       help="Output file for batch results (JSON format)")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualization for single image classification")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"],
                       help="Image file extensions to process")
    
    args = parser.parse_args()
    
    # Load configuration and model
    config = ClassificationConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration: {config.classification_mode} mode")
    
    # Create classifier
    classifier = CellClassifier(
        model_path=args.model_path,
        config=config
    )
    
    # Print model info
    model_info = classifier.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    if args.image:
        # Single image classification
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        
        result = classify_single_image(
            classifier, 
            image_path, 
            visualize=not args.no_visualize
        )
        
        # Print final result
        print("\n" + "="*50)
        print("CLASSIFICATION RESULT")
        print("="*50)
        print(f"Image: {image_path.name}")
        print(f"Prediction: {result.get('prediction', 'ERROR')}")
        
        if 'error' not in result:
            if config.classification_mode == "binary":
                print(f"Probability: {result['probability']:.4f}")
                print(f"Confidence: {result['confidence']:.4f}")
            else:
                print(f"Overall Confidence: {result['confidence']:.4f}")
                print("Class Probabilities:")
                for class_name, prob in result['class_probabilities'].items():
                    print(f"  {class_name}: {prob:.4f}")
        
    else:
        # Batch classification
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return
        
        # Find all images in directory
        image_paths = []
        for ext in args.extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
        
        if not image_paths:
            logger.error(f"No images found in {image_dir} with extensions {args.extensions}")
            return
        
        logger.info(f"Found {len(image_paths)} images to classify")
        
        # Classify batch
        results = classify_batch_images(
            classifier,
            image_paths,
            save_results=True,
            output_path=Path(args.output)
        )
        
        # Print summary
        if results:
            print("\n" + "="*50)
            print("BATCH CLASSIFICATION SUMMARY")
            print("="*50)
            print(f"Total images processed: {len(results)}")
            
            predictions = [r['prediction'] for r in results if 'prediction' in r]
            summary = {pred: predictions.count(pred) for pred in set(predictions)}
            
            for pred, count in summary.items():
                percentage = (count / len(predictions)) * 100 if predictions else 0
                print(f"{pred}: {count} images ({percentage:.1f}%)")
            
            print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()