"""Enhanced training metrics tracking utilities with final model evaluation."""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, deque
from pathlib import Path
import logging
import os

# Import existing components
from segmentation.inference.inference_engine import InferenceEngine, InferenceConfig
from segmentation.inference.threshold_optimizer import ThresholdOptimizer, ThresholdSearchConfig


class MetricsTracker:
    """Metrics tracker with final model evaluation capabilities."""
    
    def __init__(self, save_dir: Optional[str] = None, window_size: int = 10):
        """
        Initialize enhanced metrics tracker.
        
        Args:
            save_dir: Directory to save metrics.
            window_size: Window size for moving averages.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.window_size = window_size
        
        # Store metrics by phase (train/val)
        self.history = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        
        # Store recent values for moving averages
        self.recent_values = {
            'train': defaultdict(lambda: deque(maxlen=window_size)),
            'val': defaultdict(lambda: deque(maxlen=window_size))
        }
        
        # Store best values (assuming lower is better for losses)
        self.best_values = {
            'train': {},
            'val': {}
        }
        self.best_epochs = {
            'train': {},
            'val': {}
        }
        
        # Final evaluation results
        self.final_evaluation_results = {}
        self.optimal_thresholds = {}
        
        self.logger = logging.getLogger(__name__)
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def log_epoch_metrics(self, phase: str, epoch: int, metrics_dict: Dict[str, Union[float, torch.Tensor]]) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            phase: Training phase ('train' or 'val').
            epoch: Current epoch number.
            metrics_dict: Dictionary of metric names to values.
        """
        if phase not in self.history:
            self.history[phase] = defaultdict(list)
            self.recent_values[phase] = defaultdict(lambda: deque(maxlen=self.window_size))
            self.best_values[phase] = {}
            self.best_epochs[phase] = {}
        
        for name, value in metrics_dict.items():
            # Convert tensor to float if needed
            if hasattr(value, 'item'):
                value = value.item()
            elif isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().numpy())
            
            # Store the value
            self.history[phase][name].append(value)
            self.recent_values[phase][name].append(value)
            
            # Update best values (assuming lower is better for losses)
            if name not in self.best_values[phase]:
                self.best_values[phase][name] = value
                self.best_epochs[phase][name] = epoch
            elif value < self.best_values[phase][name]:
                self.best_values[phase][name] = value
                self.best_epochs[phase][name] = epoch
    
    def perform_final_evaluation(
        self,
        model: torch.nn.Module,
        test_dataset,
        metrics_to_optimize: List[str] = None,
        is_deep_model: bool = False,
        evaluation_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform final model evaluation with optimal threshold finding.
        
        Args:
            model: Trained model to evaluate
            test_dataset: Test dataset for evaluation
            metrics_to_optimize: List of metrics to find optimal thresholds for
            is_deep_model: Whether the model was trained for many epochs
            evaluation_config: Additional configuration parameters
            
        Returns:
            Dictionary containing all evaluation results
        """
        if metrics_to_optimize is None:
            metrics_to_optimize = ['AP50']
        
        if evaluation_config is None:
            evaluation_config = {}
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING FINAL MODEL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Optimizing thresholds for: {metrics_to_optimize}")
        
        # Create evaluation directory
        eval_dir = self.save_dir / "final_evaluation" if self.save_dir else Path("./final_evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Find optimal thresholds
        optimal_thresholds = self._find_optimal_thresholds(
            model, test_dataset, metrics_to_optimize, is_deep_model, evaluation_config
        )
        
        # Step 2: Evaluate with optimal thresholds
        evaluation_results = self._evaluate_with_optimal_thresholds(
            model, test_dataset, optimal_thresholds, evaluation_config
        )
        
        # Step 3: Combine and store results
        final_results = {
            'optimal_thresholds': optimal_thresholds,
            'evaluation_results': evaluation_results,
            'training_summary': self.get_training_summary(),
            'best_training_metrics': self._get_best_training_metrics()
        }
        
        # Store results
        self.final_evaluation_results = final_results
        self.optimal_thresholds = optimal_thresholds
        
        # Save results
        self._save_final_evaluation_results(final_results, eval_dir)
        
        # Print summary
        self._print_final_evaluation_summary(final_results)
        
        self.logger.info("Final model evaluation completed successfully")
        
        return final_results
    
    def _find_optimal_thresholds(
        self,
        model: torch.nn.Module,
        dataset,
        metrics_to_optimize: List[str],
        is_deep_model: bool,
        evaluation_config: Dict
    ) -> Dict[str, Dict]:
        """Find optimal thresholds for specified metrics."""
        self.logger.info("🔍 Finding optimal thresholds...")
        
        # Create base inference config
        base_config = InferenceConfig(
            save_outputs=False,  # keep explicit control
            evaluation_methods=evaluation_config.get('evaluation_methods', ['deepcell', 'coco']),
            **{k: v for k, v in evaluation_config.items() 
            if k not in ['evaluation_methods', 'save_outputs']}
        )
        
        # Create threshold search config
        search_config = ThresholdSearchConfig()
        
        # Initialize threshold optimizer
        optimizer = ThresholdOptimizer(model, base_config, search_config)
        
        # Run optimization
        threshold_results = optimizer.optimize(
            dataset=dataset,
            metrics=metrics_to_optimize,
            is_deep_model=is_deep_model
        )
        
        # Extract optimal thresholds
        optimal_thresholds = {}
        for metric, result_data in threshold_results.items():
            best_result = result_data['best_result']
            optimal_thresholds[metric] = {
                'threshold': best_result.threshold,
                'score': best_result.score,
                'search_history': len(result_data['search_history'])
            }
            
            self.logger.info(
                f"✅ {metric.upper()}: optimal threshold = {best_result.threshold:.6f}, "
                f"score = {best_result.score:.6f}"
            )
        
        return optimal_thresholds
    
    def _evaluate_with_optimal_thresholds(
        self,
        model: torch.nn.Module,
        dataset,
        optimal_thresholds: Dict[str, Dict],
        evaluation_config: Dict
    ) -> Dict[str, Dict]:
        """Evaluate model with each optimal threshold."""
        self.logger.info("📊 Evaluating with optimal thresholds...")
        
        evaluation_results = {}
        
        for metric, threshold_info in optimal_thresholds.items():
            threshold = threshold_info['threshold']
            
            self.logger.info(f"Evaluating with {metric} optimal threshold: {threshold:.6f}")
            
            # Create inference config with optimal threshold
            config = InferenceConfig(
                bbox_threshold=threshold,
                save_outputs=True,
                evaluation_methods=evaluation_config.get('evaluation_methods', ['deepcell', 'coco']),
                **{k: v for k, v in evaluation_config.items() if k not in ['evaluation_methods', 'save_outputs']}
            )
            
            # Create inference engine
            inference_engine = InferenceEngine(model, config)
            
            # Run evaluation
            results = inference_engine.process_dataset(
                dataset,
                output_path=str(self.save_dir / "final_evaluation" / f"{metric}_threshold_results"),
                test_mode=False,
                visualize=False
            )
            
            # Store results
            evaluation_results[metric] = {
                'threshold_used': threshold,
                'metrics': results,
                'summary': self._extract_key_metrics(results)
            }
        
        return evaluation_results
    
    def _extract_key_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract key metrics from evaluation results."""
        batch_metrics = results.get('batch_metrics', {})
        
        key_metrics = {}
        
        # Standard metrics
        for metric in ['precision', 'recall', 'f1', 'dice', 'jaccard', 'AP', 'AP50', 'AP75']:
            if metric in batch_metrics:
                key_metrics[metric] = batch_metrics[metric]
        
        return key_metrics
    
    def _get_best_training_metrics(self) -> Dict[str, Dict]:
        """Get best metrics achieved during training."""
        best_metrics = {}
        
        for phase in ['train', 'val']:
            if phase in self.best_values:
                best_metrics[phase] = {}
                for metric, value in self.best_values[phase].items():
                    epoch = self.best_epochs[phase].get(metric, 0)
                    best_metrics[phase][metric] = {
                        'value': value,
                        'epoch': epoch
                    }
        
        return best_metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {}
        
        for phase in ['train', 'val']:
            if phase in self.history:
                phase_summary = {}
                for metric, values in self.history[phase].items():
                    if values:
                        phase_summary[metric] = {
                            'final_value': values[-1] if values else None,
                            'best_value': self.best_values[phase].get(metric),
                            'best_epoch': self.best_epochs[phase].get(metric),
                            'total_epochs': len(values),
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                summary[phase] = phase_summary
        
        return summary
    
    def _save_final_evaluation_results(self, results: Dict, eval_dir: Path) -> None:
        """Save final evaluation results to files."""
        # Save complete results
        complete_path = eval_dir / "complete_evaluation_results.json"
        with open(complete_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        # Save optimal thresholds summary
        thresholds_path = eval_dir / "optimal_thresholds.json"
        with open(thresholds_path, 'w') as f:
            json.dump(results['optimal_thresholds'], f, indent=2, default=self._json_serializer)
        
        # Save performance summary
        performance_summary = {}
        for metric, eval_result in results['evaluation_results'].items():
            performance_summary[metric] = {
                'threshold': eval_result['threshold_used'],
                'key_metrics': eval_result['summary']
            }
        
        performance_path = eval_dir / "performance_summary.json"
        with open(performance_path, 'w') as f:
            json.dump(performance_summary, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Final evaluation results saved to {eval_dir}")
    
    def _print_final_evaluation_summary(self, results: Dict) -> None:
        """Print formatted final evaluation summary."""
        print("\n" + "=" * 80)
        print("FINAL MODEL EVALUATION SUMMARY")
        print("=" * 80)
        
        # Print optimal thresholds
        print("\n📍 OPTIMAL THRESHOLDS:")
        print("-" * 40)
        for metric, threshold_info in results['optimal_thresholds'].items():
            print(f"  {metric.upper():8s}: {threshold_info['threshold']:.6f} "
                  f"(score: {threshold_info['score']:.6f})")
        
        # Print evaluation results
        print("\n📊 EVALUATION RESULTS:")
        print("-" * 40)
        for metric, eval_result in results['evaluation_results'].items():
            print(f"\n  {metric.upper()} Threshold ({eval_result['threshold_used']:.6f}):")
            summary = eval_result['summary']
            for key, value in summary.items():
                print(f"    {key:12s}: {value:.6f}")
        
        # Print best training metrics
        print("\n🏆 BEST TRAINING METRICS:")
        print("-" * 40)
        best_metrics = results['best_training_metrics']
        for phase, metrics in best_metrics.items():
            if metrics:  # Only print if there are metrics
                print(f"\n  {phase.upper()}:")
                for metric, info in metrics.items():
                    print(f"    {metric:15s}: {info['value']:.6f} (epoch {info['epoch']})")
        
        print("\n" + "=" * 80)
    
    def get_final_evaluation_results(self) -> Dict[str, Any]:
        """Get final evaluation results."""
        return self.final_evaluation_results
    
    def get_optimal_thresholds(self) -> Dict[str, Dict]:
        """Get optimal thresholds."""
        return self.optimal_thresholds
    
    def get_best_threshold_for_metric(self, metric: str) -> Optional[float]:
        """Get the best threshold for a specific metric."""
        if metric in self.optimal_thresholds:
            return self.optimal_thresholds[metric]['threshold']
        return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return str(obj)
    
    # Keep all existing methods from the original MetricsTracker
    def get_latest(self, metric_name: str, phase: str = 'train') -> Optional[float]:
        """Get the latest value of a metric."""
        if phase in self.history and metric_name in self.history[phase] and self.history[phase][metric_name]:
            return self.history[phase][metric_name][-1]
        return None
    
    def get_moving_average(self, metric_name: str, phase: str = 'train') -> Optional[float]:
        """Get the moving average of a metric."""
        if phase in self.recent_values and metric_name in self.recent_values[phase] and self.recent_values[phase][metric_name]:
            return np.mean(list(self.recent_values[phase][metric_name]))
        return None
    
    def get_best(self, metric_name: str, phase: str = 'train') -> Optional[tuple]:
        """Get the best value and epoch for a metric."""
        if phase in self.best_values and metric_name in self.best_values[phase]:
            return self.best_values[phase][metric_name], self.best_epochs[phase][metric_name]
        return None
    
    def get_history(self, phase: str) -> Optional[Dict[str, List[float]]]:
        """Get history of all metrics for a phase."""
        if phase in self.history:
            return dict(self.history[phase])
        return None
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save all metrics to a JSON file."""
        if not self.save_dir:
            self.logger.warning("No save directory specified, cannot save metrics")
            return
            
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = {
            'history': {phase: dict(metrics) for phase, metrics in self.history.items()},
            'best_values': self.best_values,
            'best_epochs': self.best_epochs,
            'summary': self.get_summary(),
            'final_evaluation': self.final_evaluation_results,
            'optimal_thresholds': self.optimal_thresholds
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Metrics saved to {save_path}")
    
    def get_summary(self, phase: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all metrics."""
        summary = {}
        
        phases_to_process = [phase] if phase else list(self.history.keys())
        
        for p in phases_to_process:
            if p not in self.history:
                continue
                
            phase_summary = {}
            for name, values in self.history[p].items():
                if values:
                    phase_summary[name] = {
                        'latest': values[-1],
                        'best': self.best_values[p].get(name),
                        'best_epoch': self.best_epochs[p].get(name),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'moving_avg': self.get_moving_average(name, p),
                        'count': len(values)
                    }
            
            if phase:
                return phase_summary
            else:
                summary[p] = phase_summary
        
        return summary


class MultiStageMetricsTracker:
    """Enhanced multi-stage metrics tracker with final evaluation."""
    
    def __init__(self, save_dir: Optional[str] = None, stages: List[str] = None):
        """Initialize enhanced multi-stage metrics tracker."""
        self.save_dir = Path(save_dir) if save_dir else None
        self.stages = stages or ['stage1', 'stage2']
        
        # Create enhanced tracker for each stage
        self.stage_trackers = {}
        for stage in self.stages:
            stage_dir = self.save_dir / stage if self.save_dir else None
            self.stage_trackers[stage] = MetricsTracker(stage_dir)
        
        # Global enhanced tracker
        self.global_tracker = MetricsTracker(self.save_dir)
    
    def perform_final_evaluation(
        self,
        model: torch.nn.Module,
        test_dataset,
        metrics_to_optimize: List[str] = None,
        is_deep_model: bool = False,
        evaluation_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform final evaluation using the global tracker."""
        return self.global_tracker.perform_final_evaluation(
            model, test_dataset, metrics_to_optimize, is_deep_model, evaluation_config
        )
    
    def get_global_tracker(self) -> MetricsTracker:
        """Get the enhanced global tracker."""
        return self.global_tracker
    
    def log_stage_metrics(self, stage: str, phase: str, epoch: int, 
                         metrics_dict: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Log metrics for a specific stage."""
        if stage in self.stage_trackers:
            self.stage_trackers[stage].log_epoch_metrics(phase, epoch, metrics_dict)
    
    def log_global_metrics(self, phase: str, epoch: int, 
                          metrics_dict: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Log global metrics."""
        self.global_tracker.log_epoch_metrics(phase, epoch, metrics_dict)


# Utility function to integrate with existing training pipeline
def create_metrics_tracker(save_dir: str, multi_stage: bool = True) -> Union[MetricsTracker, MultiStageMetricsTracker]:
    """
    Factory function to create appropriate metrics tracker.
    
    Args:
        save_dir: Directory to save metrics
        multi_stage: Whether to create multi-stage tracker
        
    Returns:
        Enhanced metrics tracker instance
    """
    if multi_stage:
        return MultiStageMetricsTracker(save_dir)
    else:
        return MetricsTracker(save_dir)


# Integration helper for existing training scripts
class TrainingIntegrationHelper:
    """Helper class to integrate metrics tracking with existing training pipeline."""
    
    @staticmethod
    def finalize_training_with_evaluation(
        trainer,  # MultiStageTrainer or similar
        test_dataset,
        metrics_to_optimize: List[str] = None,
        is_deep_model: bool = False,
        evaluation_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Finalize training with comprehensive evaluation.
        
        Args:
            trainer: Trained model instance
            test_dataset: Test dataset for evaluation
            metrics_to_optimize: Metrics to optimize thresholds for
            is_deep_model: Whether model was trained extensively
            evaluation_config: Additional evaluation configuration
            
        Returns:
            Complete training and evaluation results
        """
        # Get the model from trainer
        model = trainer.model if hasattr(trainer, 'model') else trainer
        
        # Create enhanced metrics tracker if trainer doesn't have one
        if hasattr(trainer, 'metrics_tracker') and isinstance(trainer.metrics_tracker, (MetricsTracker, MultiStageMetricsTracker)):
            metrics_tracker = trainer.metrics_tracker
        else:
            output_dir = getattr(trainer, 'output_dir', './results')
            metrics_tracker = create_metrics_tracker(output_dir, multi_stage=True)
        
        # Perform final evaluation
        final_results = metrics_tracker.perform_final_evaluation(
            model, test_dataset, metrics_to_optimize, is_deep_model, evaluation_config
        )
        
        return final_results