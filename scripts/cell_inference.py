"""
Launch script for memory-efficient cell inference on unlabeled image folders.
Processes images in small batches to avoid CUDA out of memory errors.

Usage:
    python run_cell_inference.py --config configs/binary_config.yaml --image-dir path/to/images/
    python run_cell_inference.py --config configs/ternary_config.yaml --image-dir path/to/images/ --batch-size 8
"""

import argparse
import logging
from pathlib import Path
import torch

from classification.config import ClassificationConfig
from classification.memory_efficient import MemoryEfficientCellClassifier
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


def validate_arguments(args):
    """Validate command line arguments."""
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    
    if args.batch_size > 32:
        logger.warning(f"Large batch size ({args.batch_size}) may cause memory issues. Consider using smaller batches.")


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


def load_classifier(config: ClassificationConfig) -> MemoryEfficientCellClassifier:
    """Load and initialize the classifier."""
    try:
        classifier = MemoryEfficientCellClassifier(
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


def run_analysis(classifier: MemoryEfficientCellClassifier, config: ClassificationConfig, args) -> dict:
    """Run the main analysis."""
    analyzer = CellAnalyzer(classifier)
    
    logger.info(f"Starting memory-efficient analysis of images in {config.img_dir}")
    logger.info(f"Using batch size: {args.batch_size}")
    
    results = analyzer.analyze_image_folder(
        image_dir=Path(config.img_dir),
        extensions=args.extensions,
        batch_size=args.batch_size,
        save_individual_results=args.save_individual_results
    )
    
    if not results or 'error' in results:
        raise RuntimeError(f"Analysis failed: {results.get('error', 'Unknown error')}")
    
    # Save results
    # Determine output directory
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = Path(config.img_dir).parent / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results_path = output_dir / "enhanced_analysis_results.json"
    analyzer.save_results(
        results, 
        main_results_path
    )

    return results, output_dir


def print_summary(results: dict, config: ClassificationConfig, output_dir: Path):
    """Print enhanced summary to console."""
    print("\n" + "="*90)
    print("ENHANCED CELL CLASSIFICATION ANALYSIS SUMMARY")
    print("="*90)
    
    # Processing summary
    if 'processing_metadata' in results:
        proc_meta = results['processing_metadata']
        print(f"Batch Size Used: {proc_meta.get('batch_size_used', 'N/A')}")
        print(f"Images Found: {proc_meta.get('total_images_found', 'N/A')}")
        print(f"Successfully Processed: {proc_meta.get('successfully_processed', 'N/A')}")
        print(f"Processing Errors: {proc_meta.get('processing_errors', 'N/A')}")
        print(f"Success Rate: {100 - proc_meta.get('error_rate', 0):.1f}%")
    
    basic_stats = results.get('basic_statistics', {})
    insights = results.get('insights_and_recommendations', {})
    
    print(f"\nClassification Mode: {config.classification_mode}")
    
    if basic_stats.get('prediction_counts'):
        print("\nOVERALL PREDICTION DISTRIBUTION:")
        print("-" * 50)
        for pred, count in basic_stats['prediction_counts'].items():
            percentage = basic_stats.get('prediction_percentages', {}).get(pred, 0)
            print(f"{pred.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Show relative distribution excluding uncertain
        print(f"\nRELATIVE DISTRIBUTION (EXCLUDING UNCERTAIN):")
        print("-" * 50)
        decided_count = basic_stats.get('decided_predictions_count', 0)
        decided_pct = basic_stats.get('decided_predictions_percentage', 0)
        print(f"Total Decided Predictions: {decided_count} ({decided_pct:.1f}% of all)")
        
        for pred, percentage in basic_stats.get('relative_percentages_excluding_uncertain', {}).items():
            count = basic_stats.get('decided_prediction_counts', {}).get(pred, 0)
            print(f"{pred.capitalize()}: {count} ({percentage:.1f}% of decided)")
    
    if insights.get('summary'):
        summary = insights['summary']
        print(f"\nPRIMARY FINDING:")
        print(f"{summary.get('primary_finding', 'N/A')}")
        
        print(f"\nCONFIDENCE ASSESSMENT:")
        print(f"{summary.get('confidence_assessment', 'N/A')}")
        
        print(f"\nUNCERTAINTY LEVEL: {summary.get('uncertainty_level', 0):.1f}%")
        print(f"DECIDED PREDICTIONS: {summary.get('decided_percentage', 0):.1f}%")
        
        # Show relative findings among decided predictions
        if 'relative_findings_among_decided' in summary:
            print(f"\nRELATIVE FINDINGS AMONG DECIDED PREDICTIONS:")
            print("-" * 50)
            rel_findings = summary['relative_findings_among_decided']
            for key, value in rel_findings.items():
                print(f"{key.replace('_', ' ').title()}: {value:.1f}%")
    
    if insights.get('recommendations'):
        print(f"\nRECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print(f"\nOUTPUT FILES:")
    print("-" * 50)
    print(f"Detailed results: {output_dir / 'enhanced_analysis_results.json'}")
    print(f"Summary report: {output_dir / 'analysis_summary_report.txt'}")
    
    # Print key insights
    print_key_insights(basic_stats, config)


def print_key_insights(basic_stats: dict, config: ClassificationConfig):
    """Print key insights summary."""
    print(f"\nKEY INSIGHTS:")
    print("-" * 50)
    
    if basic_stats.get('decided_predictions_count', 0) > 0:
        decided_pct = basic_stats.get('decided_predictions_percentage', 0)
        print(f"• {decided_pct:.1f}% of images resulted in confident predictions")
        
        if config.classification_mode == "binary":
            rel_canc = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0)
            if rel_canc > 50:
                print(f"• Among confident predictions, {rel_canc:.1f}% are classified as cancerous")
                print("• This suggests a high-risk sample population requiring clinical attention")
            else:
                print(f"• Among confident predictions, {100-rel_canc:.1f}% are classified as non-cancerous")
                print("• This suggests a relatively low-risk sample population")
        else:
            rel_canc = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0)
            rel_fp = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('false-positive', 0)
            print(f"• Among confident predictions:")
            print(f"  - {rel_canc:.1f}% are true cancerous cells")
            print(f"  - {rel_fp:.1f}% are false-positive detections")
            
            if rel_canc > 30:
                print("• High prevalence of true cancerous cells detected")
            elif rel_fp > 20:
                print("• Significant false-positive rate may indicate segmentation issues")
    
    uncertain_pct = basic_stats.get('prediction_percentages', {}).get('uncertain', 0)
    if uncertain_pct > 25:
        print(f"• High uncertainty ({uncertain_pct:.1f}%) suggests challenging samples or threshold optimization needed")


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
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient analysis of unlabeled cell images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--image-dir", type=str,
                       help="Directory containing images to analyze")
    
    # Optional arguments
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save analysis results (default: {image-dir}/../analysis_results)")
    parser.add_argument("--batch-size", type=int, default=2048,
                       help="Batch size for processing (reduce if getting OOM errors)")
    parser.add_argument("--save-individual-results", action="store_true",
                       help="Save detailed results for each individual image")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip creating visualization plots")
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
        # Validate arguments
        validate_arguments(args)
        
        # Setup GPU environment
        setup_gpu_environment(args)
        
        # Load configuration
        config = ClassificationConfig.from_yaml(args.config)
        
        # Override config with command line arguments
        if args.image_dir:
            config.img_dir = args.image_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        
        logger.info(f"Loaded configuration: {config.classification_mode} mode")
        
        # Load classifier
        classifier = load_classifier(config)
        
        # Run analysis
        results, output_dir = run_analysis(classifier, config, args)
        
        # Print summary
        print_summary(results, config, output_dir)
        
        # Print memory status
        print_memory_status()
        
        logger.info("Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        raise


if __name__ == "__main__":
    main()