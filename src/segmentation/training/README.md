# CellSAM Training Module

This module provides a comprehensive, well-structured system for training the CellSAM model with support for multi-stage training, various training strategies, and hyperparameter optimization.

## 🏗️ Architecture Overview

The training system is organized into several key components:

```
src/segmentation/training/
├── config/           # Configuration classes
├── core/            # Core training implementations  
├── losses/          # Loss function implementations
├── strategies/      # Training strategies (k-fold, train/val split, etc.)
├── utils/           # Utility functions (monitoring, visualization, etc.)
└── runners/         # Experiment orchestration
```

## ✨ Key Features

* **Multi-stage Training** : Separate configuration and training for Stage 1 (DETR) and Stage 2 (SAM)
* **Multiple Training Strategies** : K-fold cross-validation, train/val split, or full dataset training
* **Hyperparameter Optimization** : Grid search and random search support
* **Comprehensive Monitoring** : GPU memory tracking, metrics logging, and training visualization
* **Flexible Configuration** : Easily configurable loss weights, learning rates, and training parameters
* **Backward Compatibility** : Maintains compatibility with existing `finetune()` function

## 🚀 Quick Start

### Basic Usage (Backward Compatible)

```python
from segmentation.training import finetune

# Simple training with default settings
results = finetune(
    img_path="path/to/images",
    mask_path="path/to/masks",
    output_name="my_experiment",
    total_epochs_s1=100,
    total_epochs_s2=100,
    k_fold_training=True
)
```

### Advanced Usage with Custom Configuration

```python
from segmentation.training import (
    ExperimentRunner, ExperimentConfig, 
    Stage1Config, Stage2Config
)

# Create custom configurations
stage1_config = Stage1Config(
    stage1_type="all_backbones",  # Train both CellFinder and image encoder
    total_epochs=200,
    learning_rate=1e-4,
    ce_loss_weight=2,
    bbox_loss_weight=10,
    giou_loss_weight=5
)

stage2_config = Stage2Config(
    total_epochs=300,
    learning_rate=5e-5,
    focal_loss_weight=2.0,
    dice_loss_weight=1.5,
    boundary_loss_weight=0.5,
    online_hard_negative_mining=True  # Focus on hard examples
)

# Create experiment configuration
config = ExperimentConfig(
    img_path="path/to/images",
    mask_path="path/to/masks",
    output_name="advanced_experiment",
    strategy="k_fold",
    k_folds=5,
    stage1_config=stage1_config,
    stage2_config=stage2_config,
    batch_size=2,
    debug=True,
    augmentation=True,
    stain_transform=True,
    gamma=2.4
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()
```

## 📊 Training Strategies

### 1. K-Fold Cross Validation (Recommended)

```python
config = ExperimentConfig(
    # ... other parameters
    strategy="k_fold",
    k_folds=5
)
```

Best for robust evaluation and model selection. Provides statistical significance testing.

### 2. Train/Validation Split

```python
config = ExperimentConfig(
    # ... other parameters  
    strategy="train_val_split"
)
```

Faster than k-fold, good for hyperparameter tuning and quick experiments.

### 3. Full Dataset Training

```python
config = ExperimentConfig(
    # ... other parameters
    strategy="full_dataset" 
)
```

Uses entire dataset for training (no validation). Best for final model training.

## 🔧 Configuration Options

### Stage 1 Configuration (DETR Training)

```python
Stage1Config(
    stage1_type="cellfinder",        # "cellfinder", "image_encoder", "all_backbones"
    total_epochs=500,
    learning_rate=1e-4,
    patience=15,                     # Early stopping patience
    ce_loss_weight=1,                # Classification loss weight
    bbox_loss_weight=5,              # Bounding box loss weight  
    giou_loss_weight=2               # GIoU loss weight
)
```

### Stage 2 Configuration (SAM Training)

```python
Stage2Config(
    total_epochs=500,
    learning_rate=1e-4,
    patience=15,
    focal_loss_weight=1.0,           # Focal loss weight
    focal_gamma=2.0,                 # Focal loss gamma parameter
    dice_loss_weight=1.0,            # Dice loss weight
    boundary_loss_weight=1.0,        # Boundary loss weight
    online_hard_negative_mining=False,           # Enable OHEM
    online_hard_negative_mining_weighted=False   # Enable weighted OHEM
)
```

## 🔍 Hyperparameter Optimization

### Random Search

```python
from segmentation.training import HyperparameterOptimizer, HyperparameterSpace

# Define search space
param_space = HyperparameterSpace(
    stage1_learning_rates=[1e-4, 1e-5],
    stage2_learning_rates=[1e-4, 5e-5],
    focal_gammas=[2.0, 3.0],
    gammas=[2.1, 2.4],
    augmentation_flags=[False, True]
)

# Run optimization
optimizer = HyperparameterOptimizer(
    base_config=base_config,
    param_space=param_space,
    output_dir="output/hp_search",
    max_parallel=2
)

results = optimizer.run_random_search(n_trials=20)
```

### Grid Search

```python
results = optimizer.run_grid_search()  # Tests all combinations
```

## 📈 Monitoring and Visualization

### GPU Memory Monitoring

```python
from segmentation.training.utils import GPUMonitor

monitor = GPUMonitor()
monitor.print_memory_summary("Training Start")
```

### Training Visualization

The system automatically generates:

* Stage 1: Bounding box predictions vs ground truth
* Stage 2: Instance segmentation masks
* Training loss curves
* Loss component breakdowns

### Metrics Tracking

```python
from segmentation.training.utils import MetricsTracker

tracker = MetricsTracker(save_dir="output/metrics")
tracker.update({"loss": 0.5, "f1": 0.8}, epoch=1)
tracker.save_metrics()
```

## 📁 Output Structure

```
output/
└── experiment_name/
    ├── experiment_config.json       # Experiment configuration
    ├── logs/
    │   └── experiment.log          # Training logs
    ├── models/
    │   └── fold_X/
    │       ├── best_model_s1.pth   # Best Stage 1 model
    │       └── best_model_s2.pth   # Best Stage 2 model
    ├── results/
    │   ├── individual_results.json # Results per fold
    │   ├── aggregated_results.json # K-fold aggregated results
    │   └── fold_X/
    │       └── visualizations/     # Training visualizations
    └── metrics/
        ├── stage1_metrics.json     # Stage 1 metrics
        └── stage2_metrics.json     # Stage 2 metrics
```

## 🎯 Best Practices

### 1. Start with Quick Test

```python
# Quick test with minimal epochs
results = finetune(
    img_path="path/to/images",
    mask_path="path/to/masks", 
    output_name="quick_test",
    strategy="train_val_split",
    total_epochs_s1=5,
    total_epochs_s2=5,
    debug=True
)
```

### 2. Use Pre-evaluation

```python
config = ExperimentConfig(
    # ... other parameters
    preeval=True  # Evaluate model before training
)
```

### 3. Enable Debug Mode During Development

```python
config = ExperimentConfig(
    # ... other parameters
    debug=True  # Enables detailed logging and visualization
)
```

### 4. Hyperparameter Search Strategy

1. Start with focused search space
2. Use train/val split for faster iterations
3. Use random search initially, then grid search around promising areas
4. Run final evaluation with k-fold cross-validation

### 5. Loss Weight Tuning

* **High boundary loss** : Better cell boundary detection
* **High focal gamma** : Focus on hard examples
* **Balanced weights** : Start with equal weights, then adjust based on validation performance

## 🔄 Migration from Legacy Code

The new system maintains backward compatibility. To migrate:

### Option 1: Keep using finetune()

```python
# Your existing code works unchanged
results = finetune(img_path, mask_path, output_name, **kwargs)
```

### Option 2: Gradual migration to new system

```python
# Create equivalent configuration
config = ExperimentConfig(
    img_path=img_path,
    mask_path=mask_path,
    output_name=output_name,
    stage1_config=Stage1Config(total_epochs=total_epochs_s1),
    stage2_config=Stage2Config(total_epochs=total_epochs_s2),
    # ... map other parameters
)

runner = ExperimentRunner(config)
results = runner.run()
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory** : Reduce batch_size, enable debug mode to monitor GPU usage
2. **Training Slow** : Use train/val split instead of k-fold for development
3. **Poor Performance** : Try different loss weight combinations, enable OHEM
4. **Convergence Issues** : Adjust learning rates, increase patience for early stopping

### Debug Mode

Enable debug mode for detailed information:

```python
config.debug = True
```

This provides:

* GPU memory usage tracking
* Detailed loss component logging
* Training visualizations
* Step-by-step progress information

## 📚 API Reference

### Core Classes

* `ExperimentRunner`: Main experiment orchestrator
* `ExperimentConfig`: Complete experiment configuration
* `Stage1Config`: Stage 1 specific configuration
* `Stage2Config`: Stage 2 specific configuration
* `HyperparameterOptimizer`: Hyperparameter search orchestrator

### Training Strategies

* `KFoldStrategy`: K-fold cross-validation
* `TrainValSplitStrategy`: Train/validation split
* `FullDatasetStrategy`: Full dataset training

### Utilities

* `GPUMonitor`: GPU memory monitoring
* `MetricsTracker`: Training metrics tracking
* `TrainingVisualizer`: Training visualization
* `CheckpointManager`: Model checkpoint management
* `EarlyStopping`: Early stopping implementation

## 🤝 Contributing

When adding new features:

1. Follow the modular structure
2. Add appropriate configuration options
3. Include visualization and monitoring
4. Maintain backward compatibility
5. Add examples and documentation

## 📄 License

This module is part of the CellSAM project. See the main project LICENSE file for details.
