#!/usr/bin/env python3
"""
Integrated training pipeline that includes threshold optimization after training.
This extends the original cell_classification.py to automatically optimize thresholds
using the test set after training is complete.

Usage:
    python integrated_training_pipeline.py --config configs/classification/binary_config.yaml --optimize-thresholds
    python integrated_training_pipeline.py --config configs/classification/ternary_config.yaml --optimize-thresholds
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd

from classification.config import ClassificationConfig
from classification.utils import (
    load_classification_labels, get_image_paths_and_labels, split_data,
    calculate_pos_weight, calculate_class_weights, visualize_class_distribution
)
from classification.datasets import CellClassificationDataset
from classification.models import get_classification_model
from classification.trainer import ClassificationTrainer
from classification.inference import CellClassifier
from classification.threshold_optimizer import ThresholdOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_loaders(config: ClassificationConfig) -> tuple:
    """
    Set up data loaders for training, validation, and testing.
    (Same as original implementation)
    """
    # Load labels
    label_map = load_classification_labels(
        Path(config.labels_csv), 
        classification_mode=config.classification_mode
    )
    
    # Get image paths and labels
    data_samples = get_image_paths_and_labels(
        Path(config.data_dir), 
        label_map
    )
    
    # Visualize class distribution
    visualize_class_distribution(
        data_samples, 
        classification_mode=config.classification_mode,
        title=f"Overall Class Distribution ({config.classification_mode.title()} Mode)"
    )
    
    # Split data
    train_data, val_data, test_data = split_data(
        data_samples,
        config.train_split,
        config.val_split,
        config.test_split,
        config.random_seed
    )
    
    logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Calculate class weights
    pos_weight = None
    class_weights = None
    
    if config.use_class_weights:
        if config.classification_mode == "binary":
            pos_weight = calculate_pos_weight(train_data)
        else:  # ternary
            class_weights = calculate_class_weights(train_data, config.num_classes)
    
    # Create datasets
    train_dataset = CellClassificationDataset(
        train_data, config.image_size, config.normalize_mean, config.normalize_std,
        classification_mode=config.classification_mode, is_train=True
    )
    
    val_dataset = CellClassificationDataset(
        val_data, config.image_size, config.normalize_mean, config.normalize_std,
        classification_mode=config.classification_mode, is_train=False
    )
    
    test_dataset = CellClassificationDataset(
        test_data, config.image_size, config.normalize_mean, config.normalize_std,
        classification_mode=config.classification_mode, is_train=False
    ) if test_data else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data, pos_weight, class_weights

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
    updated_config = ClassificationConfig(
        # Copy all original parameters
        classification_mode=original_config.classification_mode,
        data_dir=original_config.data_dir,
        labels_csv=original_config.labels_csv,
        image_size=original_config.image_size,
        normalize_mean=original_config.normalize_mean,
        normalize_std=original_config.normalize_std,
        batch_size=original_config.batch_size,
        epochs=original_config.epochs,
        learning_rate=original_config.learning_rate,
        optimizer=original_config.optimizer,
        loss_function=original_config.loss_function,
        weight_decay=original_config.weight_decay,
        use_class_weights=original_config.use_class_weights,
        num_workers=original_config.num_workers,
        pin_memory=original_config.pin_memory,
        model_name=original_config.model_name,
        pretrained=original_config.pretrained,
        patience=original_config.patience,
        early_stopping_metric=original_config.early_stopping_metric,
        train_split=original_config.train_split,
        val_split=original_config.val_split,
        test_split=original_config.test_split,
        random_seed=original_config.random_seed,
        
        # Update with optimized thresholds
        confidence_threshold_high=recommended.get('high_threshold', 
                                                 recommended.get('confidence_threshold_high', 
                                                               original_config.confidence_threshold_high)),
        confidence_threshold_low=recommended.get('low_threshold', 
                                                recommended.get('confidence_threshold_low', 
                                                              original_config.confidence_threshold_low)),
        uncertainty_threshold=recommended.get('uncertainty_threshold', 
                                            original_config.uncertainty_threshold)
    )
    
    # Save updated configuration
    optimized_config_path = output_dir / f"optimized_{original_config.classification_mode}_config.yaml"
    updated_config.to_yaml(optimized_config_path)
    
    logger.info(f"Optimized configuration saved to {optimized_config_path}")
    logger.info(f"Optimized thresholds:")
    if original_config.classification_mode == "binary":
        logger.info(f"  Low threshold: {updated_config.confidence_threshold_low:.3f}")
        logger.info(f"  High threshold: {updated_config.confidence_threshold_high:.3f}")
    else:
        logger.info(f"  Confidence low: {updated_config.confidence_threshold_low:.3f}")
        logger.info(f"  Confidence high: {updated_config.confidence_threshold_high:.3f}")
        logger.info(f"  Uncertainty threshold: {updated_config.uncertainty_threshold:.3f}")
    
    # Save optimization summary
    optimization_summary = {
        'original_thresholds': {
            'confidence_threshold_high': original_config.confidence_threshold_high,
            'confidence_threshold_low': original_config.confidence_threshold_low,
            'uncertainty_threshold': original_config.uncertainty_threshold
        },
        'optimized_thresholds': {
            'confidence_threshold_high': updated_config.confidence_threshold_high,
            'confidence_threshold_low': updated_config.confidence_threshold_low,
            'uncertainty_threshold': updated_config.uncertainty_threshold
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
    
    return updated_config

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
    output_dir = Path(args.output_dir) / config.classification_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original configuration to output directory
    config.to_yaml(output_dir / f"original_{config.classification_mode}_config.yaml")
    
    # Set up data loaders
    train_loader, val_loader, test_loader, train_data, val_data, test_data, pos_weight, class_weights = setup_data_loaders(config)
    
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
        
        # Create trainer for evaluation (even if we skipped training)
        trainer = ClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            pos_weight=pos_weight,
            class_weights=class_weights
        )
        
        # Evaluate with original thresholds
        test_metrics = trainer.evaluate_test_set(test_loader)
        
        # Save original test metrics
        original_test_metrics_path = output_dir / f"original_{config.classification_mode}_test_metrics.json"
        with open(original_test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Saved original test metrics to {original_test_metrics_path}")
        
        # Plot confusion matrix with original thresholds
        confusion_matrix_path = output_dir / f"original_{config.classification_mode}_confusion_matrix.png"
        trainer.plot_confusion_matrix(test_loader, save_path=confusion_matrix_path)
        
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


if __name__ == "__main__":
    main()