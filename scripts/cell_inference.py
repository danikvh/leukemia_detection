"""
Multi-folder cell inference script for processing multiple cell image directories.
Processes each folder separately and creates both individual and combined results.

Usage:
    python run_multi_folder_inference.py --config configs/binary_config.yaml --img-dir ../results/patch_extraction/bbox0.43/
    python run_multi_folder_inference.py --config configs/ternary_config.yaml --img-dir ../results/patch_extraction/bbox0.43/ --batch-size 8
"""

import argparse
import logging
from pathlib import Path
import torch
import json
from typing import Dict, List, Any
from collections import defaultdict

from classification.config import ClassificationConfig
from classification.inference import CellClassifier
from classification.analysis import CellAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def find_cell_folders(base_dir: Path) -> List[Path]:
    """Find all cell folders in the base directory."""
    cell_folders = []
    
    # Look for folders with pattern: image_*/cells
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('image_'):
            cells_dir = item / 'cells'
            if cells_dir.exists() and cells_dir.is_dir():
                cell_folders.append(cells_dir)
    
    return sorted(cell_folders)


def setup_gpu_environment(args):
    """Setup GPU environment and clear cache if requested."""
    if args.clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory before analysis: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        try:
            logger.info(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            logger.info("Could not get GPU total memory info")
    else:
        logger.info("GPU not available, using CPU")


def load_classifier(config: ClassificationConfig) -> CellClassifier:
    """Load and initialize the classifier."""
    try:
        classifier = CellClassifier(
            model_path=config.model_path,
            config=config
        )
        
        # Print model info
        model_info = classifier.get_model_info()
        logger.info(f"Model loaded: {model_info['model_name']} ({model_info['classification_mode']} mode)")
        logger.info(f"Device: {model_info['device']}")
        logger.info(f"Number of classes: {model_info['num_classes']}")
        logger.info(f"Image size: {model_info['image_size']}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"Failed to create classifier: {e}")
        raise


def process_single_folder(classifier: CellClassifier, 
                         cell_folder: Path, 
                         config: ClassificationConfig, 
                         args) -> Dict[str, Any]:
    """Process a single cell folder."""
    folder_name = cell_folder.parent.name  # e.g., 'image_0_44G5D'
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing folder: {folder_name}")
    logger.info(f"Cell images directory: {cell_folder}")
    logger.info(f"{'='*60}")
    
    analyzer = CellAnalyzer(classifier)
    
    # Run analysis on this specific folder
    results = analyzer.analyze_image_folder(
        image_dir=cell_folder,
        extensions=args.extensions,
        batch_size=args.batch_size,
        save_individual_results=args.save_individual_results
    )
    
    if not results or 'error' in results:
        logger.error(f"Analysis failed for {folder_name}: {results.get('error', 'Unknown error')}")
        return {
            'folder_name': folder_name,
            'folder_path': str(cell_folder),
            'status': 'failed',
            'error': results.get('error', 'Unknown error'),
            'results': None
        }
    
    # Add folder metadata to results
    results['folder_metadata'] = {
        'folder_name': folder_name,
        'folder_path': str(cell_folder),
        'parent_directory': str(cell_folder.parent)
    }
    
    # Create output directory for this folder
    if config.output_dir:
        folder_output_dir = Path(config.output_dir) / folder_name
    else:
        folder_output_dir = cell_folder.parent.parent / "analysis_results" / folder_name
    
    folder_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual folder results
    individual_results_path = folder_output_dir / f"{folder_name}_analysis_results.json"
    analyzer.save_results(results, individual_results_path)
    
    logger.info(f"✓ Completed analysis for {folder_name}")
    logger.info(f"  Results saved to: {folder_output_dir}")
    
    return {
        'folder_name': folder_name,
        'folder_path': str(cell_folder),
        'output_path': str(folder_output_dir),
        'status': 'success',
        'results': results
    }


def create_combined_summary(all_results: List[Dict], output_dir: Path, config: ClassificationConfig):
    """Create a combined summary of all folder analyses."""
    logger.info("\n" + "="*80)
    logger.info("CREATING COMBINED SUMMARY")
    logger.info("="*80)
    
    # Separate successful and failed results
    successful_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] == 'failed']
    
    # Initialize combined statistics
    combined_stats = {
        'total_folders_processed': len(all_results),
        'successful_folders': len(successful_results),
        'failed_folders': len(failed_results),
        'success_rate': (len(successful_results) / len(all_results) * 100) if all_results else 0,
        'folder_summaries': {},
        'aggregate_statistics': {
            'total_images_across_all_folders': 0,
            'total_decided_predictions': 0,
            'total_uncertain_predictions': 0,
            'combined_prediction_counts': defaultdict(int),
            'combined_prediction_percentages': {},
            'combined_relative_percentages': defaultdict(int),
            'folder_performance_summary': []
        }
    }
    
    # Process successful results
    total_images_all_folders = 0
    total_decided_all_folders = 0
    combined_prediction_counts = defaultdict(int)
    
    for result in successful_results:
        folder_name = result['folder_name']
        analysis = result['results']
        basic_stats = analysis.get('basic_statistics', {})
        
        # Extract key metrics for this folder
        folder_images = basic_stats.get('total_images_analyzed', 0)
        folder_decided = basic_stats.get('decided_predictions_count', 0)
        folder_uncertain = basic_stats.get('prediction_counts', {}).get('uncertain', 0)
        folder_predictions = basic_stats.get('prediction_counts', {})
        
        # Add to totals
        total_images_all_folders += folder_images
        total_decided_all_folders += folder_decided
        
        # Combine prediction counts
        for pred, count in folder_predictions.items():
            combined_prediction_counts[pred] += count
        
        # Store folder summary
        insights = analysis.get('insights_and_recommendations', {}).get('summary', {})
        combined_stats['folder_summaries'][folder_name] = {
            'total_images': folder_images,
            'decided_predictions': folder_decided,
            'uncertain_predictions': folder_uncertain,
            'decided_percentage': basic_stats.get('decided_predictions_percentage', 0),
            'primary_finding': insights.get('primary_finding', 'N/A'),
            'confidence_assessment': insights.get('confidence_assessment', 'N/A'),
            'prediction_counts': dict(folder_predictions),
            'relative_percentages': basic_stats.get('relative_percentages_excluding_uncertain', {})
        }
        
        # Add to performance summary
        combined_stats['aggregate_statistics']['folder_performance_summary'].append({
            'folder_name': folder_name,
            'images_processed': folder_images,
            'success_rate': basic_stats.get('success_rate', 0),
            'decided_percentage': basic_stats.get('decided_predictions_percentage', 0),
            'primary_finding_short': insights.get('primary_finding', 'N/A')[:50] + '...' if len(insights.get('primary_finding', '')) > 50 else insights.get('primary_finding', 'N/A')
        })
    
    # Calculate combined statistics
    combined_stats['aggregate_statistics']['total_images_across_all_folders'] = total_images_all_folders
    combined_stats['aggregate_statistics']['total_decided_predictions'] = total_decided_all_folders
    combined_stats['aggregate_statistics']['total_uncertain_predictions'] = combined_prediction_counts.get('uncertain', 0)
    combined_stats['aggregate_statistics']['combined_prediction_counts'] = dict(combined_prediction_counts)
    
    # Calculate combined percentages
    if total_images_all_folders > 0:
        combined_stats['aggregate_statistics']['combined_prediction_percentages'] = {
            pred: count / total_images_all_folders * 100 
            for pred, count in combined_prediction_counts.items()
        }
    
    # Calculate relative percentages (excluding uncertain)
    if total_decided_all_folders > 0:
        decided_prediction_counts = {k: v for k, v in combined_prediction_counts.items() if k != 'uncertain'}
        combined_stats['aggregate_statistics']['combined_relative_percentages'] = {
            pred: count / total_decided_all_folders * 100 
            for pred, count in decided_prediction_counts.items()
        }
    
    # Add failed folders info
    if failed_results:
        combined_stats['failed_folders_details'] = [
            {
                'folder_name': r['folder_name'],
                'error': r.get('error', 'Unknown error')
            }
            for r in failed_results
        ]
    
    # Save combined results
    combined_results_path = output_dir / "combined_analysis_results.json"
    with open(combined_results_path, 'w') as f:
        json.dump(combined_stats, f, indent=4, default=str)
    
    # Create comprehensive text summary
    create_text_summary(combined_stats, output_dir, config)
    
    logger.info(f"Combined analysis results saved to: {combined_results_path}")


def create_text_summary(combined_stats: Dict, output_dir: Path, config: ClassificationConfig):
    """Create a comprehensive text summary."""
    summary_path = output_dir / "COMBINED_SUMMARY_REPORT.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MULTI-FOLDER CELL CLASSIFICATION ANALYSIS - COMBINED SUMMARY REPORT\n")
        f.write("="*100 + "\n\n")
        
        # Overall processing summary
        f.write("PROCESSING OVERVIEW:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Classification Mode: {config.classification_mode.upper()}\n")
        f.write(f"Total Folders Found: {combined_stats['total_folders_processed']}\n")
        f.write(f"Successfully Processed: {combined_stats['successful_folders']}\n")
        f.write(f"Failed to Process: {combined_stats['failed_folders']}\n")
        f.write(f"Overall Success Rate: {combined_stats['success_rate']:.1f}%\n\n")
        
        # Aggregate statistics
        agg_stats = combined_stats['aggregate_statistics']
        f.write("AGGREGATE STATISTICS ACROSS ALL FOLDERS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Images Processed: {agg_stats['total_images_across_all_folders']}\n")
        f.write(f"Total Decided Predictions: {agg_stats['total_decided_predictions']}\n")
        f.write(f"Total Uncertain Predictions: {agg_stats['total_uncertain_predictions']}\n")
        f.write(f"Overall Decided Rate: {(agg_stats['total_decided_predictions'] / agg_stats['total_images_across_all_folders'] * 100) if agg_stats['total_images_across_all_folders'] > 0 else 0:.1f}%\n\n")
        
        # Combined prediction distribution
        f.write("COMBINED PREDICTION DISTRIBUTION (ALL FOLDERS):\n")
        f.write("-" * 50 + "\n")
        for pred, count in agg_stats['combined_prediction_counts'].items():
            percentage = agg_stats['combined_prediction_percentages'].get(pred, 0)
            f.write(f"{pred.capitalize()}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Relative distribution excluding uncertain
        if agg_stats['combined_relative_percentages']:
            f.write("RELATIVE DISTRIBUTION (EXCLUDING UNCERTAIN - ALL FOLDERS):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Decided Across All Folders: {agg_stats['total_decided_predictions']} "
                   f"({(agg_stats['total_decided_predictions'] / agg_stats['total_images_across_all_folders'] * 100) if agg_stats['total_images_across_all_folders'] > 0 else 0:.1f}% of all images)\n")
            
            for pred, percentage in agg_stats['combined_relative_percentages'].items():
                count = agg_stats['combined_prediction_counts'].get(pred, 0)
                f.write(f"{pred.capitalize()}: {count} ({percentage:.1f}% of decided)\n")
            f.write("\n")
        
        # Individual folder performance
        f.write("INDIVIDUAL FOLDER PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Folder Name':<20} {'Images':<8} {'Decided%':<9} {'Success%':<9} {'Primary Finding':<40}\n")
        f.write("-" * 90 + "\n")
        
        for folder_perf in agg_stats['folder_performance_summary']:
            f.write(f"{folder_perf['folder_name']:<20} "
                   f"{folder_perf['images_processed']:<8} "
                   f"{folder_perf['decided_percentage']:<9.1f} "
                   f"{folder_perf['success_rate']:<9.1f} "
                   f"{folder_perf['primary_finding_short']:<40}\n")
        f.write("\n")
        
        # Detailed folder summaries
        f.write("DETAILED FOLDER ANALYSIS:\n")
        f.write("=" * 50 + "\n\n")
        
        for folder_name, summary in combined_stats['folder_summaries'].items():
            f.write(f"FOLDER: {folder_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images: {summary['total_images']}\n")
            f.write(f"Decided Predictions: {summary['decided_predictions']} ({summary['decided_percentage']:.1f}%)\n")
            f.write(f"Uncertain Predictions: {summary['uncertain_predictions']}\n")
            
            f.write(f"\nPrediction Distribution:\n")
            for pred, count in summary['prediction_counts'].items():
                pct = (count / summary['total_images'] * 100) if summary['total_images'] > 0 else 0
                f.write(f"  {pred.capitalize()}: {count} ({pct:.1f}%)\n")
            
            if summary['relative_percentages']:
                f.write(f"\nRelative Distribution (Decided Only):\n")
                for pred, pct in summary['relative_percentages'].items():
                    count = summary['prediction_counts'].get(pred, 0)
                    f.write(f"  {pred.capitalize()}: {count} ({pct:.1f}% of decided)\n")
            
            f.write(f"\nPrimary Finding: {summary['primary_finding']}\n")
            f.write(f"Confidence Assessment: {summary['confidence_assessment']}\n")
            f.write("\n" + "="*50 + "\n\n")
        
        # Failed folders
        if combined_stats.get('failed_folders_details'):
            f.write("FAILED FOLDERS:\n")
            f.write("-" * 20 + "\n")
            for failed in combined_stats['failed_folders_details']:
                f.write(f"Folder: {failed['folder_name']}\n")
                f.write(f"Error: {failed['error']}\n\n")
        
        # Key insights and recommendations
        f.write("KEY INSIGHTS AND RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Generate insights based on combined data
        total_images = agg_stats['total_images_across_all_folders']
        decided_rate = (agg_stats['total_decided_predictions'] / total_images * 100) if total_images > 0 else 0
        
        if config.classification_mode == "binary":
            canc_count = agg_stats['combined_prediction_counts'].get('cancerous', 0)
            canc_pct = agg_stats['combined_prediction_percentages'].get('cancerous', 0)
            rel_canc_pct = agg_stats['combined_relative_percentages'].get('cancerous', 0)
            
            f.write(f"• Overall Analysis: {canc_count} cancerous cells detected ({canc_pct:.1f}% of all images)\n")
            f.write(f"• Among decided predictions: {rel_canc_pct:.1f}% are classified as cancerous\n")
            f.write(f"• Decision confidence: {decided_rate:.1f}% of images resulted in confident predictions\n")
            
            if canc_pct > 25:
                f.write(f"• HIGH PRIORITY: Significant cancer prevalence detected across folders\n")
            elif decided_rate < 70:
                f.write(f"• ATTENTION: Low decision confidence suggests need for manual review\n")
            else:
                f.write(f"• Overall results show manageable cancer prevalence with good confidence\n")
        else:
            canc_count = agg_stats['combined_prediction_counts'].get('cancerous', 0)
            fp_count = agg_stats['combined_prediction_counts'].get('false-positive', 0)
            canc_pct = agg_stats['combined_prediction_percentages'].get('cancerous', 0)
            fp_pct = agg_stats['combined_prediction_percentages'].get('false-positive', 0)
            
            f.write(f"• True cancerous cells: {canc_count} ({canc_pct:.1f}% of all images)\n")
            f.write(f"• False-positive detections: {fp_count} ({fp_pct:.1f}% of all images)\n")
            f.write(f"• Decision confidence: {decided_rate:.1f}% of images resulted in confident predictions\n")
            
            if canc_pct > 20:
                f.write(f"• HIGH PRIORITY: Significant true cancer prevalence detected\n")
            if fp_pct > 15:
                f.write(f"• ATTENTION: High false-positive rate may indicate segmentation issues\n")
        
        f.write(f"\n• Processing completed successfully for {combined_stats['success_rate']:.1f}% of folders\n")
        f.write(f"• Total processing covered {total_images} cell images across {combined_stats['successful_folders']} folders\n")
    
    logger.info(f"Comprehensive text summary saved to: {summary_path}")


def print_final_summary(all_results: List[Dict], config: ClassificationConfig):
    """Print final summary to console."""
    successful_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] == 'failed']
    
    print("\n" + "="*100)
    print("MULTI-FOLDER CELL CLASSIFICATION ANALYSIS - FINAL SUMMARY")
    print("="*100)
    
    print(f"Classification Mode: {config.classification_mode.upper()}")
    print(f"Total Folders Processed: {len(all_results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Success Rate: {len(successful_results) / len(all_results) * 100:.1f}%")
    
    if successful_results:
        total_images = sum(r['results']['basic_statistics']['total_images_analyzed'] 
                          for r in successful_results)
        total_decided = sum(r['results']['basic_statistics']['decided_predictions_count'] 
                           for r in successful_results)
        
        print(f"\nAggregate Results:")
        print(f"Total Images Analyzed: {total_images}")
        print(f"Total Decided Predictions: {total_decided} ({total_decided/total_images*100:.1f}%)")
        
        print(f"\nSuccessful Folders:")
        for result in successful_results:
            folder_stats = result['results']['basic_statistics']
            print(f"  {result['folder_name']}: {folder_stats['total_images_analyzed']} images, "
                  f"{folder_stats['decided_predictions_percentage']:.1f}% decided")
    
    if failed_results:
        print(f"\nFailed Folders:")
        for result in failed_results:
            print(f"  {result['folder_name']}: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*100)


def print_memory_status():
    """Print final memory status."""
    if torch.cuda.is_available():
        print(f"\nFINAL MEMORY STATUS:")
        print("-" * 50)
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        try:
            print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")
        except Exception:
            print("Could not get peak memory info")


def main():
    """Main function for multi-folder processing."""
    parser = argparse.ArgumentParser(
        description="Multi-folder memory-efficient analysis of cell images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--img-dir", type=str, 
                       help="Base directory containing image_* folders with cells subdirectories")
    
    # Optional arguments
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save analysis results")
    parser.add_argument("--batch-size", type=int, default=2048,
                       help="Batch size for processing (reduce if getting OOM errors)")
    parser.add_argument("--save-individual-results", action="store_true",
                       help="Save detailed results for each individual image")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"],
                       help="Image file extensions to process")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear GPU cache before starting")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Setup GPU environment
        setup_gpu_environment(args)
        
        # Load configuration
        config = ClassificationConfig.from_yaml(args.config)
        
        # Set output directory
        if args.img_dir:
            config.img_dir = args.img_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        
        # Create main output directory
        main_output_dir = Path(config.output_dir)
        main_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loaded configuration: {config.classification_mode} mode")
        logger.info(f"Base directory: {config.img_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        
        # Find all cell folders
        base_dir = Path(config.img_dir)
        cell_folders = find_cell_folders(base_dir)
        
        if not cell_folders:
            logger.error(f"No cell folders found in {base_dir}")
            logger.info("Looking for folders with pattern: image_*/cells")
            return
        
        logger.info(f"Found {len(cell_folders)} cell folders to process:")
        for folder in cell_folders:
            logger.info(f"  - {folder.parent.name}/cells ({folder})")
        
        # Load classifier
        classifier = load_classifier(config)
        
        # Process each folder
        all_results = []
        
        for i, cell_folder in enumerate(cell_folders, 1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"PROCESSING FOLDER {i}/{len(cell_folders)}")
            logger.info(f"{'#'*80}")
            
            try:
                result = process_single_folder(classifier, cell_folder, config, args)
                all_results.append(result)
                
                # Clear GPU memory between folders
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to process folder {cell_folder.parent.name}: {e}")
                all_results.append({
                    'folder_name': cell_folder.parent.name,
                    'folder_path': str(cell_folder),
                    'status': 'failed',
                    'error': str(e),
                    'results': None
                })
        
        # Create combined summary
        create_combined_summary(all_results, main_output_dir, config)
        
        # Print final summary
        print_final_summary(all_results, config)
        
        # Print memory status
        print_memory_status()
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-FOLDER ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Individual results saved in: {main_output_dir}")
        logger.info(f"Combined summary: {main_output_dir}/COMBINED_SUMMARY_REPORT.txt")
        logger.info(f"Combined data: {main_output_dir}/combined_analysis_results.json")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        raise


if __name__ == "__main__":
    main()