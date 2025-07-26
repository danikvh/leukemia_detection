"""Training metrics tracking utilities."""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
from pathlib import Path
import logging


class MetricsTracker:
    """Track training metrics over time."""
    
    def __init__(self, save_dir: Optional[str] = None, window_size: int = 10):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics.
            window_size: Window size for moving averages.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.window_size = window_size
        
        # Store all metrics
        self.metrics = defaultdict(list)
        
        # Store recent values for moving averages
        self.recent_values = defaultdict(lambda: deque(maxlen=window_size))
        
        # Store best values
        self.best_values = {}
        self.best_epochs = {}
        
        # Current epoch
        self.current_epoch = 0
        
        self.logger = logging.getLogger(__name__)
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def update(self, metrics_dict: Dict[str, float], epoch: Optional[int] = None) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric names to values.
            epoch: Current epoch number.
        """
        if epoch is not None:
            self.current_epoch = epoch
            
        for name, value in metrics_dict.items():
            # Convert tensor to float if needed
            if hasattr(value, 'item'):
                value = value.item()
            
            # Store the value
            self.metrics[name].append(value)
            self.recent_values[name].append(value)
            
            # Update best values (assuming lower is better for losses)
            if name not in self.best_values:
                self.best_values[name] = value
                self.best_epochs[name] = self.current_epoch
            elif value < self.best_values[name]:
                self.best_values[name] = value
                self.best_epochs[name] = self.current_epoch
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_moving_average(self, metric_name: str) -> Optional[float]:
        """Get the moving average of a metric."""
        if metric_name in self.recent_values and self.recent_values[metric_name]:
            return np.mean(list(self.recent_values[metric_name]))
        return None
    
    def get_best(self, metric_name: str) -> Optional[tuple]:
        """
        Get the best value and epoch for a metric.
        
        Returns:
            Tuple of (best_value, best_epoch) or None.
        """
        if metric_name in self.best_values:
            return self.best_values[metric_name], self.best_epochs[metric_name]
        return None
    
    def get_all_metrics(self, metric_name: str) -> List[float]:
        """Get all recorded values for a metric."""
        return self.metrics.get(metric_name, [])
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with summary statistics for each metric.
        """
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'latest': values[-1],
                    'best': self.best_values.get(name),
                    'best_epoch': self.best_epochs.get(name),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'moving_avg': self.get_moving_average(name),
                    'count': len(values)
                }
        
        return summary
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save all metrics to a JSON file."""
        if not self.save_dir:
            self.logger.warning("No save directory specified, cannot save metrics")
            return
            
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = {
            'metrics': dict(self.metrics),
            'best_values': self.best_values,
            'best_epochs': self.best_epochs,
            'summary': self.get_summary()
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Metrics saved to {save_path}")
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore metrics
        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.best_values = data.get('best_values', {})
        self.best_epochs = data.get('best_epochs', {})
        
        # Restore recent values (only keep last window_size values)
        for name, values in self.metrics.items():
            self.recent_values[name] = deque(values[-self.window_size:], 
                                           maxlen=self.window_size)
        
        self.logger.info(f"Metrics loaded from {filepath}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.recent_values.clear()
        self.best_values.clear()
        self.best_epochs.clear()
        self.current_epoch = 0
    
    def has_improved(self, metric_name: str, patience: int = 1) -> bool:
        """
        Check if a metric has improved in the last N epochs.
        
        Args:
            metric_name: Name of the metric to check.
            patience: Number of epochs to look back.
            
        Returns:
            True if the metric improved in the last `patience` epochs.
        """
        if metric_name not in self.best_epochs:
            return False
            
        best_epoch = self.best_epochs[metric_name]
        return self.current_epoch - best_epoch <= patience


class MultiStageMetricsTracker:
    """Track metrics for multi-stage training."""
    
    def __init__(self, save_dir: Optional[str] = None, stages: List[str] = None):
        """
        Initialize multi-stage metrics tracker.
        
        Args:
            save_dir: Directory to save metrics.
            stages: List of stage names (e.g., ['stage1', 'stage2']).
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.stages = stages or ['stage1', 'stage2']
        
        # Create tracker for each stage
        self.stage_trackers = {}
        for stage in self.stages:
            stage_dir = self.save_dir / stage if self.save_dir else None
            self.stage_trackers[stage] = MetricsTracker(stage_dir)
        
        # Global tracker for overall metrics
        self.global_tracker = MetricsTracker(self.save_dir)
    
    def update_stage(self, stage: str, metrics_dict: Dict[str, float], 
                    epoch: Optional[int] = None) -> None:
        """Update metrics for a specific stage."""
        if stage in self.stage_trackers:
            self.stage_trackers[stage].update(metrics_dict, epoch)
    
    def update_global(self, metrics_dict: Dict[str, float], 
                     epoch: Optional[int] = None) -> None:
        """Update global metrics."""
        self.global_tracker.update(metrics_dict, epoch)
    
    def get_stage_tracker(self, stage: str) -> Optional[MetricsTracker]:
        """Get tracker for a specific stage."""
        return self.stage_trackers.get(stage)
    
    def get_global_tracker(self) -> MetricsTracker:
        """Get global tracker."""
        return self.global_tracker
    
    def save_all_metrics(self) -> None:
        """Save metrics for all stages."""
        # Save stage-specific metrics
        for stage, tracker in self.stage_trackers.items():
            tracker.save_metrics(f"{stage}_metrics.json")
        
        # Save global metrics
        self.global_tracker.save_metrics("global_metrics.json")
    
    def get_combined_summary(self) -> Dict[str, Dict]:
        """Get combined summary of all stages."""
        summary = {}
        
        for stage, tracker in self.stage_trackers.items():
            summary[stage] = tracker.get_summary()
        
        summary['global'] = self.global_tracker.get_summary()
        return summary


class FoldMetricsAggregator:
    """Aggregate metrics across multiple folds for k-fold cross-validation."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize fold metrics aggregator.
        
        Args:
            save_dir: Directory to save aggregated metrics.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.fold_results = []
        self.logger = logging.getLogger(__name__)
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def add_fold_results(self, fold_idx: int, results: Dict) -> None:
        """
        Add results from a single fold.
        
        Args:
            fold_idx: Fold index.
            results: Dictionary containing fold results.
        """
        fold_data = {
            'fold': fold_idx,
            'results': results
        }
        self.fold_results.append(fold_data)
    
    def aggregate_metrics(self) -> Dict[str, Dict]:
        """
        Aggregate metrics across all folds.
        
        Returns:
            Dictionary containing aggregated statistics.
        """
        if not self.fold_results:
            return {}
        
        # Extract metric keys from first fold
        first_fold = self.fold_results[0]['results']
        
        # Handle nested structure (e.g., best_f1, best_dice, best_ap50)
        aggregated = {}
        
        for result_type in first_fold.keys():
            if isinstance(first_fold[result_type], dict):
                aggregated[result_type] = {}
                
                for metric_name in first_fold[result_type].keys():
                    # Collect values across folds
                    values = []
                    for fold_data in self.fold_results:
                        fold_result = fold_data['results'][result_type][metric_name]
                        if isinstance(fold_result, list):
                            values.append(fold_result)
                        else:
                            values.append([fold_result])
                    
                    # Convert to numpy array for easier computation
                    values_array = np.array(values)
                    
                    # Compute statistics
                    aggregated[result_type][metric_name] = {
                        'mean': np.mean(values_array, axis=0).tolist(),
                        'std': np.std(values_array, axis=0).tolist(),
                        'min': np.min(values_array, axis=0).tolist(),
                        'max': np.max(values_array, axis=0).tolist(),
                        'median': np.median(values_array, axis=0).tolist(),
                        'values_per_fold': [v.tolist() if isinstance(v, np.ndarray) 
                                          else v for v in values]
                    }
        
        return aggregated
    
    def print_summary(self) -> None:
        """Print summary of aggregated metrics."""
        aggregated = self.aggregate_metrics()
        
        print("=" * 60)
        print("K-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        
        for result_type, metrics in aggregated.items():
            print(f"\n{result_type.upper()}:")
            print("-" * 40)
            
            for metric_name, stats in metrics.items():
                if isinstance(stats['mean'], list):
                    mean_val = np.mean(stats['mean'])
                    std_val = np.mean(stats['std'])
                else:
                    mean_val = stats['mean']
                    std_val = stats['std']
                
                print(f"  {metric_name:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    def save_aggregated_results(self, filename: str = "aggregated_results.json") -> None:
        """Save aggregated results to file."""
        if not self.save_dir:
            self.logger.warning("No save directory specified")
            return
        
        aggregated = self.aggregate_metrics()
        
        # Add metadata
        results_with_metadata = {
            'num_folds': len(self.fold_results),
            'fold_indices': [fold['fold'] for fold in self.fold_results],
            'aggregated_metrics': aggregated,
            'individual_fold_results': self.fold_results
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        self.logger.info(f"Aggregated results saved to {save_path}")


class LossComponentTracker:
    """Track individual loss components during training."""
    
    def __init__(self):
        """Initialize loss component tracker."""
        self.components = defaultdict(list)
        self.weights = {}
    
    def update_components(self, loss_dict: Dict[str, Union[float, torch.Tensor]], 
                         weights: Optional[Dict[str, float]] = None) -> None:
        """
        Update loss components.
        
        Args:
            loss_dict: Dictionary of loss component names to values.
            weights: Optional dictionary of loss weights.
        """
        for name, value in loss_dict.items():
            if hasattr(value, 'item'):
                value = value.item()
            self.components[name].append(value)
        
        if weights:
            self.weights.update(weights)
    
    def get_weighted_total(self) -> Optional[float]:
        """Calculate weighted total loss from latest components."""
        if not self.components:
            return None
        
        total = 0.0
        for name, values in self.components.items():
            if values:  # Check if list is not empty
                latest_value = values[-1]
                weight = self.weights.get(name, 1.0)
                total += latest_value * weight
        
        return total
    
    def get_component_history(self, component: str) -> List[float]:
        """Get history of a specific component."""
        return self.components.get(component, [])
    
    def get_all_components(self) -> Dict[str, List[float]]:
        """Get all component histories."""
        return dict(self.components)


# Utility functions for metrics aggregation
def aggregate_metrics_across_folds(results_per_fold: List[Dict]) -> Dict:
    """
    Aggregate metrics across folds.
    
    Args:
        results_per_fold: List of result dictionaries from each fold.
        
    Returns:
        Dictionary with aggregated statistics.
    """
    aggregator = FoldMetricsAggregator()
    
    for i, results in enumerate(results_per_fold):
        aggregator.add_fold_results(i, results)
    
    return aggregator.aggregate_metrics()


def save_metrics_summary(metrics_dict: Dict, save_path: str) -> None:
    """
    Save metrics summary to JSON file.
    
    Args:
        metrics_dict: Dictionary containing metrics.
        save_path: Path to save the summary.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)


def print_metrics_summary(metrics_dict: Dict, title: str = "Metrics Summary") -> None:
    """
    Print formatted metrics summary.
    
    Args:
        metrics_dict: Dictionary containing metrics.
        title: Title for the summary.
    """
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    
    for category, metrics in metrics_dict.items():
        print(f"\n{category.upper()}:")
        print("-" * 30)
        
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, dict) and 'mean' in value:
                    if isinstance(value['mean'], list):
                        mean_val = np.mean(value['mean'])
                        std_val = np.mean(value.get('std', [0]))
                    else:
                        mean_val = value['mean']
                        std_val = value.get('std', 0)
                    print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"  {metric:15s}: {value}")
        else:
            print(f"  Value: {metrics}")