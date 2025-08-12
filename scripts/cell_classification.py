#!/usr/bin/env python3
"""
Integrated training pipeline that includes threshold optimization after training.

Usage:
    python cell_classification.py --config ../configs/classification/trainval_binary_config.yaml --optimize-thresholds
    python cell_classification.py --config ../configs/classification/train_ternary_config.yaml --optimize-thresholds
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from classification.config import ClassificationConfig
from classification.datasets import DataLoaderManager
from classification.evaluation import ModelEvaluator
from classification.models import get_classification_model
from classification.trainer import ClassificationTrainer
from classification.inference import CellClassifier
from classification.threshold_optimizer import ThresholdOptimizer
from classification.utils import load_classification_labels, get_image_paths_and_labels

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_thresholds_and_update_config(model_path: Path, original_config: ClassificationConfig,
                                         test_loader: DataLoader, test_data: list,
                                         output_dir: Path) -> ClassificationConfig:
    """
    Optimize thresholds using the test set and create updated configuration files.
    
    Args:
        model_path: Path to the trained model
        original_config: Original configuration used for training
        test_loader: DataLoader for test set
        test_data: List of test data tuples
        output_dir: Directory to save results
        
    Returns:
        Updated configuration with optimized thresholds
    """
    logger.info("Starting threshold optimization...")
    
    # Create classifier with trained model
    classifier = CellClassifier(
        model_path=str(model_path),
        config=original_config
    )
    
    # Create threshold optimizer
    optimizer = ThresholdOptimizer(classifier)
    
    # Run optimization
    optimization_results = optimizer.optimize_thresholds(
        test_loader=test_loader,
        test_data=test_data,
        output_dir=output_dir / "threshold_optimization"
    )
    
    # Get recommended thresholds (use balanced approach as default)
    optimal_thresholds = optimization_results['optimal_thresholds']
    
    if 'balanced' in optimal_thresholds:
        recommended = optimal_thresholds['balanced']
    elif optimal_thresholds:
        recommended = list(optimal_thresholds.values())[0]  # Use first available
    else:
        logger.warning("No optimal thresholds found, keeping original values")
        return original_config
    
    # Create updated configuration with optimized thresholds
    original_config.update({
        'confidence_threshold_high': recommended.get('high_threshold', 
                                                                recommended.get('confidence_threshold_high', original_config.config.confidence_threshold_high)),
        'confidence_threshold_low': recommended.get('low_threshold', 
                                                                recommended.get('confidence_threshold_low', original_config.config.confidence_threshold_low)),
        'uncertainty_threshold': recommended.get('uncertainty_threshold', original_config.config.uncertainty_threshold)
    })
    
    # Save updated configuration
    optimized_config_path = output_dir / f"optimized_{original_config.classification_mode}_config.yaml"
    original_config.to_yaml(optimized_config_path)
    
    logger.info(f"Optimized configuration saved to {optimized_config_path}")
    logger.info(f"Optimized thresholds:")
    if original_config.classification_mode == "binary":
        logger.info(f"  Low threshold: {original_config.confidence_threshold_low:.3f}")
        logger.info(f"  High threshold: {original_config.confidence_threshold_high:.3f}")
    else:
        logger.info(f"  Confidence low: {original_config.confidence_threshold_low:.3f}")
        logger.info(f"  Confidence high: {original_config.confidence_threshold_high:.3f}")
        logger.info(f"  Uncertainty threshold: {original_config.uncertainty_threshold:.3f}")
    
    # Save optimization summary
    optimization_summary = {
        'original_thresholds': {
            'confidence_threshold_high': original_config.confidence_threshold_high,
            'confidence_threshold_low': original_config.confidence_threshold_low,
            'uncertainty_threshold': original_config.uncertainty_threshold
        },
        'optimized_thresholds': {
            'confidence_threshold_high': original_config.confidence_threshold_high,
            'confidence_threshold_low': original_config.confidence_threshold_low,
            'uncertainty_threshold': original_config.uncertainty_threshold
        },
        'optimization_strategy': recommended.get('criterion', 'balanced'),
        'performance_metrics': {
            'f1_score': recommended.get('f1_score', recommended.get('f1_macro', 0)),
            'accuracy': recommended.get('accuracy', 0),
            'uncertainty_percentage': recommended.get('uncertain_percentage', 0),
            'coverage': recommended.get('coverage', 0)
        },
        'recommendations': optimization_results.get('recommendations', {})
    }
    
    with open(output_dir / "threshold_optimization_summary.json", 'w') as f:
        json.dump(optimization_summary, f, indent=4, default=str)
    
    return original_config

def evaluate_with_optimized_thresholds(optimized_config: ClassificationConfig,
                                     model_path: Path, test_loader: DataLoader,
                                     test_data: list, output_dir: Path) -> None:
    """
    Evaluate the model using optimized thresholds and compare with original thresholds.
    """
    logger.info("Evaluating model with optimized thresholds...")
    
    # Create classifier with optimized configuration
    optimized_classifier = CellClassifier(
        model_path=str(model_path),
        config=optimized_config
    )
    
    # Run evaluation with optimized thresholds
    evaluation_results = optimized_classifier.evaluate_test_set(
        test_loader=test_loader,
        test_data=test_data,
        output_dir=output_dir / "optimized_evaluation",
        visualize=True,
        num_visualize_each=5
    )
    
    # Save comparison results
    with open(output_dir / "optimized_evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=4, default=str)
    
    logger.info("Evaluation with optimized thresholds completed")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("OPTIMIZED THRESHOLD EVALUATION SUMMARY")
    print("="*60)
    
    if "standard_metrics" in evaluation_results:
        metrics = evaluation_results["standard_metrics"]
        print(f"Performance with optimized thresholds:")
        
        if optimized_config.classification_mode == "binary":
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            
            if "threshold_analysis" in evaluation_results:
                threshold_metrics = evaluation_results["threshold_analysis"]
                print(f"  Uncertainty Rate: {threshold_metrics['undecided_percentage']:.2%}")
        else:  # ternary
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")

    
def create_threshold_comparison_report(original_config: ClassificationConfig, 
                                     optimized_config: ClassificationConfig,
                                     output_dir: Path) -> None:
    """Create a comparison report between original and optimized thresholds."""
    
    report_path = output_dir / "threshold_comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("THRESHOLD OPTIMIZATION COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Classification Mode: {original_config.classification_mode}\n")
        f.write(f"Model: {original_config.model_name}\n")
        f.write(f"Optimization Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("THRESHOLD CHANGES:\n")
        f.write("-" * 30 + "\n")
        
        if original_config.classification_mode == "binary":
            f.write(f"Low Threshold:\n")
            f.write(f"  Original: {original_config.confidence_threshold_low:.3f}\n")
            f.write(f"  Optimized: {optimized_config.confidence_threshold_low:.3f}\n")
            f.write(f"  Change: {optimized_config.confidence_threshold_low - original_config.confidence_threshold_low:+.3f}\n\n")
            
            f.write(f"High Threshold:\n")
            f.write(f"  Original: {original_config.confidence_threshold_high:.3f}\n")
            f.write(f"  Optimized: {optimized_config.confidence_threshold_high:.3f}\n")
            f.write(f"  Change: {optimized_config.confidence_threshold_high - original_config.confidence_threshold_high:+.3f}\n\n")
        else:
            f.write(f"Confidence Low Threshold:\n")
            f.write(f"  Original: {original_config.confidence_threshold_low:.3f}\n")
            f.write(f"  Optimized: {optimized_config.confidence_threshold_low:.3f}\n")
            f.write(f"  Change: {optimized_config.confidence_threshold_low - original_config.confidence_threshold_low:+.3f}\n\n")
            
            f.write(f"Confidence High Threshold:\n")
            f.write(f"  Original: {original_config.confidence_threshold_high:.3f}\n")
            f.write(f"  Optimized: {optimized_config.confidence_threshold_high:.3f}\n")
            f.write(f"  Change: {optimized_config.confidence_threshold_high - original_config.confidence_threshold_high:+.3f}\n\n")
            
            f.write(f"Uncertainty Threshold:\n")
            f.write(f"  Original: {original_config.uncertainty_threshold:.3f}\n")
            f.write(f"  Optimized: {optimized_config.uncertainty_threshold:.3f}\n")
            f.write(f"  Change: {optimized_config.uncertainty_threshold - original_config.uncertainty_threshold:+.3f}\n\n")
        
        f.write("USAGE RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Use the optimized configuration for new inference tasks\n")
        f.write("2. The optimized thresholds are designed to balance performance and uncertainty\n")
        f.write("3. Review the detailed optimization results for alternative threshold strategies\n")
        f.write("4. Consider your specific use case when choosing between optimization strategies\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"• Optimized config: optimized_{original_config.classification_mode}_config.yaml\n")
        f.write(f"• Optimization details: threshold_optimization/\n")
        f.write(f"• Evaluation results: optimized_evaluation_results.json\n")
        f.write(f"• This report: threshold_comparison_report.txt\n")
    
    logger.info(f"Threshold comparison report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Integrated training pipeline with threshold optimization")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    parser.add_argument("--output-dir", type=str, default="outputs/integrated_training",
                       help="Output directory for model and results")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate on test set, don't train")
    parser.add_argument("--optimize-thresholds", action="store_true",
                       help="Run threshold optimization after training")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only run threshold optimization (requires --resume)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ClassificationConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Classification mode: {config.classification_mode}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Loss function: {config.loss_function}")
    
    # Set up output directory
    if config.output_dir: args.output_dir = config.output_dir
    output_dir = Path(args.output_dir) / config.classification_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original configuration to output directory
    config.to_yaml(output_dir / f"original_{config.classification_mode}_config.yaml")
    
    # Load labels and get image paths
    label_map = load_classification_labels(
        Path(config.labels_csv), 
        classification_mode=config.classification_mode
    )
    
    data_samples = get_image_paths_and_labels(
        Path(config.data_dir), 
        label_map
    )
    
    # Set up data loaders using the new DataLoaderManager
    data_manager = DataLoaderManager(config)
    train_loader, val_loader, test_loader = data_manager.setup_data_loaders(data_samples)
    train_data, val_data, test_data = data_manager.get_data_splits()
    pos_weight, class_weights = data_manager.get_class_weights()

    # Create model
    model = get_classification_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        input_channels=3,
        input_image_size=config.image_size,
        classification_mode=config.classification_mode
    )
    
    logger.info(f"Created {config.model_name} model with {config.num_classes} output classes")
    
    # Model path for saving/loading
    best_model_path = output_dir / f"best_{config.classification_mode}_classification_model.pth"
    
    # Training phase
    if not args.skip_training and not args.eval_only:
        # Create trainer
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            pos_weight=pos_weight,
            class_weights=class_weights
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint_path = Path(args.resume)
            if checkpoint_path.exists():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path))
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        # Train the model
        logger.info("Starting training...")
        trainer.train(output_dir)
        
        # Plot training metrics
        metrics_plot_path = output_dir / f"{config.classification_mode}_training_metrics.png"
        trainer.plot_metrics_history(save_path=metrics_plot_path)
        
        # Load best model for evaluation
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        
    elif args.skip_training:
        # Load model from checkpoint
        if not args.resume:
            logger.error("--resume is required when using --skip-training")
            return
        
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        
        # Copy the model to expected location for consistency
        import shutil
        shutil.copy2(checkpoint_path, best_model_path)
    
    # Evaluation phase (with original thresholds)
    if test_loader is not None:
        logger.info("Evaluating on test set with original thresholds...")
        
        evaluator = ModelEvaluator(classifier=model)
        test_metrics = evaluator.evaluate_test_set(test_loader=test_loader, test_data=test_data, output_dir=output_dir)

        # Save original test metrics
        original_test_metrics_path = output_dir / f"original_{config.classification_mode}_test_metrics.json"
        with open(original_test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Saved original test metrics to {original_test_metrics_path}")
        
        # Plot confusion matrix with original thresholds
        confusion_matrix_path = output_dir / f"original_{config.classification_mode}_confusion_matrix.png"
        evaluator.plot_confusion_matrix_standalone(confusion_matrix_path)
        
        # Print original evaluation results
        print("\n" + "="*60)
        print("ORIGINAL THRESHOLD EVALUATION SUMMARY")
        print("="*60)
        print(f"Classification Mode: {config.classification_mode}")
        
        if config.classification_mode == "binary":
            print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"Precision: {test_metrics.get('precision', 0):.4f}")
            print(f"Recall: {test_metrics.get('recall', 0):.4f}")
            print(f"F1 Score: {test_metrics.get('f1_score', 0):.4f}")
            print(f"AUC: {test_metrics.get('auc', 0):.4f}")
        else:  # ternary
            print(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"F1 Score (Macro): {test_metrics.get('f1_macro', 0):.4f}")
            print(f"F1 Score (Weighted): {test_metrics.get('f1_weighted', 0):.4f}")
    
    # Threshold optimization phase
    if args.optimize_thresholds and test_loader is not None:
        logger.info("\n" + "="*60)
        logger.info("STARTING THRESHOLD OPTIMIZATION")
        logger.info("="*60)
        
        # Run threshold optimization
        optimized_config = optimize_thresholds_and_update_config(
            model_path=best_model_path,
            original_config=config,
            test_loader=test_loader,
            test_data=test_data,
            output_dir=output_dir
        )
        
        # Evaluate with optimized thresholds
        evaluate_with_optimized_thresholds(
            optimized_config=optimized_config,
            model_path=best_model_path,
            test_loader=test_loader,
            test_data=test_data,
            output_dir=output_dir
        )
        
        # Create comparison report
        create_threshold_comparison_report(
            original_config=config,
            optimized_config=optimized_config,
            output_dir=output_dir
        )
        
    elif args.optimize_thresholds and test_loader is None:
        logger.warning("Threshold optimization requested but no test set available")
    
    # Final summary
    print("\n" + "="*80)
    print("INTEGRATED TRAINING PIPELINE COMPLETE")
    print("="*80)
    
    if not args.skip_training:
        print("✓ Model training completed")
    print("✓ Evaluation with original thresholds completed")
    
    if args.optimize_thresholds and test_loader is not None:
        print("✓ Threshold optimization completed")
        print("✓ Evaluation with optimized thresholds completed")
        print(f"\nOptimized configuration available at:")
        print(f"  {output_dir / f'optimized_{config.classification_mode}_config.yaml'}")
        print(f"\nThreshold optimization results:")
        print(f"  {output_dir / 'threshold_optimization'}")
    
    print(f"\nAll results saved to: {output_dir}")
    
    logger.info("Integrated training pipeline completed successfully!")

if __name__ == "__main__":
    main()