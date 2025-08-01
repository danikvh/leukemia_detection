#!/usr/bin/env python3
"""
Main training script for cell classification.
Supports both binary (cancerous/non-cancerous) and ternary (cancerous/non-cancerous/false-positive) classification.

Usage:
    python train_classifier.py --config configs/classification/binary_config.yaml
    python train_classifier.py --config configs/classification/ternary_config.yaml
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from classification.config import ClassificationConfig
from classification.utils import (
    load_classification_labels, get_image_paths_and_labels, split_data,
    calculate_pos_weight, calculate_class_weights, visualize_class_distribution
)
from classification.datasets import CellClassificationDataset
from classification.models import get_classification_model
from classification.trainer import ClassificationTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_loaders(config: ClassificationConfig) -> tuple:
    """
    Set up data loaders for training, validation, and testing.
    
    Args:
        config (ClassificationConfig): Configuration object.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_data, pos_weight, class_weights)
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
    
    return train_loader, val_loader, test_loader, train_data, pos_weight, class_weights

def main():
    parser = argparse.ArgumentParser(description="Train cell classification model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    parser.add_argument("--output-dir", type=str, default="outputs/classification",
                       help="Output directory for model and results")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate on test set, don't train")
    
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
    
    # Save configuration to output directory
    config.to_yaml(output_dir / f"{config.classification_mode}_config.yaml")
    
    # Set up data loaders
    train_loader, val_loader, test_loader, train_data, pos_weight, class_weights = setup_data_loaders(config)
    
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
    
    if not args.eval_only:
        # Train the model
        logger.info("Starting training...")
        trainer.train(output_dir)
        
        # Plot training metrics
        metrics_plot_path = output_dir / f"{config.classification_mode}_training_metrics.png"
        trainer.plot_metrics_history(save_path=metrics_plot_path)
        
        # Load best model for evaluation
        best_model_path = output_dir / f"best_{config.classification_mode}_classification_model.pth"
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set if available
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate_test_set(test_loader)
        
        # Save test metrics
        import json
        test_metrics_path = output_dir / f"{config.classification_mode}_test_metrics.json"
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Saved test metrics to {test_metrics_path}")
        
        # Plot confusion matrix
        confusion_matrix_path = output_dir / f"{config.classification_mode}_confusion_matrix.png"
        trainer.plot_confusion_matrix(test_loader, save_path=confusion_matrix_path)
    else:
        logger.info("No test set available for evaluation")
    
    logger.info(f"Training and evaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()