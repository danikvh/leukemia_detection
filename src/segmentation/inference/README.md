# Cellular Segmentation Inference Module

A comprehensive inference framework for cellular segmentation using Cell-SAM models, featuring threshold optimization, cross-validation, and extensive evaluation metrics.

## 🚀 Features

* **Robust Inference Engine** : Process single images or entire datasets with automatic error handling
* **Adaptive Threshold Optimization** : Find optimal thresholds using bidirectional search algorithms
* **Cross-Validation Support** : Validate results across multiple data splits
* **Comprehensive Metrics** : COCO metrics, Dice score, Jaccard index, F1-score, and more
* **Flexible Configuration** : Easy-to-use configuration classes for all parameters
* **Production-Ready** : Logging, caching, error handling, and result persistence

## 📁 Project Structure

```
src/inference/
├── __init__.py              # Module exports and version info
├── inference_engine.py      # Core inference functionality
├── threshold_optimizer.py   # Threshold optimization algorithms
├── inference_pipeline.py    # High-level pipeline orchestration
└── README.md               # This file
```

## 🛠️ Installation

```bash
# Install dependencies
pip install torch torchvision numpy scikit-image scipy scikit-learn pandas

# Install your project in development mode
pip install -e .
```

## 📖 Quick Start

### Basic Inference

```python
from src.inference import create_inference_engine, InferenceConfig

# Load your model and dataset
model = load_your_cellsam_model()
dataset = load_your_dataset()

# Create inference engine
config = InferenceConfig(
    bbox_threshold=0.4,
    iou_threshold=0.6,
    save_outputs=True
)
engine = create_inference_engine(model, **config.__dict__)

# Run inference
results = engine.process_dataset(
    dataset=dataset,
    output_path="./results",
    test_mode=False  # Set True to process only 5 images for testing
)

print(f"F1-Score: {results['f1']:.4f}")
print(f"Dice Score: {results['dice']:.4f}")
print(f"AP50: {results['ap50']:.4f}")
```

### Threshold Optimization

```python
from src.inference import optimize_thresholds

# Optimize thresholds for multiple metrics
optimization_results = optimize_thresholds(
    model=model,
    dataset=dataset,
    metrics=['ap50', 'dice', 'f1'],
    is_deep_model=False,  # Set True if trained for many epochs
    iou_threshold=0.6,
    normalize_inference=True
)

# Access results
best_ap50 = optimization_results['aggregated_results']['ap50']
print(f"Optimal AP50 threshold: {best_ap50['mean_threshold']:.6f}")
print(f"AP50 Score: {best_ap50['mean_score']:.4f}")
```

### Complete Pipeline

```python
from src.inference import run_inference_pipeline

# Run everything with one function call
results = run_inference_pipeline(
    model_path="path/to/your/model.pth",
    dataset_path="path/to/your/dataset",
    output_path="./complete_results",
    optimize_thresholds=True,
    bbox_threshold=0.4,
    test_mode=False
)
```

## 🔧 Configuration Options

### InferenceConfig

| Parameter               | Type         | Default | Description                          |
| ----------------------- | ------------ | ------- | ------------------------------------ |
| `bbox_threshold`      | float        | 0.4     | Bounding box detection threshold     |
| `iou_threshold`       | float        | 0.6     | IoU threshold for object matching    |
| `cf2_threshold`       | float        | 0.1     | Secondary threshold for metrics      |
| `normalize_inference` | bool         | False   | Apply normalization during inference |
| `postprocess`         | bool         | False   | Apply post-processing to masks       |
| `save_outputs`        | bool         | False   | Save masks and scores to disk        |
| `device`              | torch.device | auto    | Device for computation               |

### ThresholdSearchConfig

| Parameter                  | Type        | Default            | Description                        |
| -------------------------- | ----------- | ------------------ | ---------------------------------- |
| `initial_threshold`      | float       | 0.2                | Starting threshold for search      |
| `initial_threshold_deep` | float       | 0.05               | Starting threshold for deep models |
| `step_sizes`             | List[float] | [0.1, 0.01, 0.001] | Step sizes for search              |
| `min_threshold`          | float       | 0.0001             | Minimum allowed threshold          |
| `max_threshold`          | float       | 1.0                | Maximum allowed threshold          |

### BatchInferenceConfig

| Parameter                  | Type      | Default                | Description                   |
| -------------------------- | --------- | ---------------------- | ----------------------------- |
| `model_path`             | str       | None                   | Path to model checkpoint      |
| `dataset_path`           | str       | ""                     | Path to dataset               |
| `output_base_path`       | str       | "./inference_results"  | Output directory              |
| `optimize_thresholds`    | bool      | False                  | Enable threshold optimization |
| `optimization_metrics`   | List[str] | ['ap50', 'dice', 'f1'] | Metrics to optimize           |
| `cross_validation_folds` | int       | 5                      | Number of CV folds            |
| `test_mode`              | bool      | False                  | Process subset for testing    |

## 📊 Available Metrics

The framework computes comprehensive evaluation metrics:

* **Object Detection** : Precision, Recall, F1-Score
* **Segmentation Quality** : Dice Score, Jaccard Index
* **COCO Metrics** : AP, AP50, AP75
* **Detailed Analysis** : Per-image statistics, confusion matrices

## 🎯 Advanced Usage

### Custom Metric Optimization

```python
from src.inference import ThresholdOptimizer, InferenceConfig, ThresholdSearchConfig

# Custom optimization setup
inference_config = InferenceConfig(bbox_threshold=0.4)
search_config = ThresholdSearchConfig(
    initial_threshold=0.3,
    step_sizes=[0.05, 0.01, 0.001]
)

optimizer = ThresholdOptimizer(model, inference_config, search_config)

# Run with custom settings
results = optimizer.optimize_with_cross_validation(
    dataset=dataset,
    metrics=['dice', 'jaccard'],  # Focus on segmentation quality
    n_folds=3,
    is_deep_model=True
)
```

### Single Image Processing

```python
from src.inference import CellSAMInferenceEngine, InferenceConfig

config = InferenceConfig(bbox_threshold=0.4)
engine = CellSAMInferenceEngine(model, config)

# Process single image
image_tensor = your_image  # Shape: (C, H, W)
result = engine.segment_single_image(image_tensor, "sample.jpg")

if result.success:
    mask = result.mask
    scores = result.scores
    print(f"Segmentation successful with threshold {result.bbox_threshold}")
else:
    print(f"Segmentation failed: {result.error_message}")
```

### Batch Processing with Custom Logic

```python
from src.inference import InferencePipeline, BatchInferenceConfig

# Custom pipeline configuration
config = BatchInferenceConfig(
    output_base_path="./custom_results",
    bbox_threshold=0.35,
    save_visualizations=True,
    optimize_thresholds=True,
    optimization_metrics=['ap50', 'recall']
)

pipeline = InferencePipeline(config)

# Load your data
dataset = your_dataset_loader()
model = your_model_loader()

# Run complete workflow
results = pipeline.run_complete_pipeline(dataset, model)

# Access different result components
standard_results = results['inference_results']
optimization_results = results['optimization_results']
optimized_results = results['optimized_inference_results']
```

## 🔍 Result Analysis

### Understanding Output Structure

```python
# Standard inference results
{
    'precision': 0.8234,
    'recall': 0.7891,
    'f1': 0.8056,
    'dice': 0.8123,
    'jaccard': 0.6834,
    'ap': 0.7456,
    'ap50': 0.8234,
    'ap75': 0.6789,
    'coco_metrics': {...},  # Detailed COCO results
    'detailed_metrics': pandas.DataFrame,  # Per-image metrics
    'segmentation_results': [...],  # Individual segmentation results
    'config': {...}  # Configuration used
}

# Optimization results
{
    'fold_results': {
        'fold_1': {...},
        'fold_2': {...},
        ...
    },
    'aggregated_results': {
        'ap50': {
            'mean_threshold': 0.3456,
            'std_threshold': 0.0123,
            'mean_score': 0.8456,
            'std_score': 0.0089
        },
        ...
    }
}
```

### Accessing Detailed Results

```python
# Get per-image metrics
detailed_df = results['detailed_metrics']
print(detailed_df.head())

# Plot threshold optimization curves
import matplotlib.pyplot as plt

opt_results = optimization_results['fold_results']['fold_1']['ap50']
search_history = opt_results['search_history']

thresholds = [r.threshold for r in search_history]
scores = [r.score for r in search_history]

plt.plot(thresholds, scores, 'bo-')
plt.xlabel('Threshold')
plt.ylabel('AP50 Score')
plt.title('Threshold Optimization Progress')
plt.show()
```

## 🚀 Performance Tips

1. **Use GPU** : Set `device='cuda'` for faster processing
2. **Enable Caching** : Set `save_outputs=True` to cache results
3. **Batch Processing** : Process multiple images together when possible
4. **Test Mode** : Use `test_mode=True` during development
5. **Memory Management** : Monitor memory usage for large datasets

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory** : Reduce batch size or use CPU
2. **Empty Masks** : Try lower bbox_threshold values
3. **Slow Processing** : Enable GPU acceleration and caching
4. **Import Errors** : Ensure all dependencies are installed

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use test mode for development
config.test_mode = True

# Check intermediate results
config.save_outputs = True
config.save_visualizations = True
```

## 📈 Future Enhancements

* [ ] Multi-GPU support for large-scale processing
* [ ] Real-time inference optimization
* [ ] Advanced visualization tools
* [ ] Model ensemble support
* [ ] Cloud deployment utilities

## 🤝 Contributing

1. Follow the existing code structure and documentation style
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure backward compatibility

## 📄 License

This module is part of the leukemia detection project. Please refer to the main project license.

---

*For more detailed examples and advanced usage patterns, check the `examples/` directory in the main project.*
