"""
Threshold optimization module for finding optimal bbox_threshold values.

This module implements adaptive search algorithms to find the best threshold
values for different metrics (AP50, Dice, F1, Recall).
"""

import logging
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold

from .inference_engine import InferenceEngine, InferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSearchConfig:
    """Configuration for threshold search parameters."""
    initial_threshold: float = 0.2
    initial_threshold_deep: float = 0.05  # For models trained for many epochs
    step_sizes: List[float] = None
    step_sizes_deep: List[float] = None
    min_threshold: float = 0.0001
    max_threshold: float = 1.0
    tolerance: float = 1e-6
    
    def __post_init__(self):
        if self.step_sizes is None:
            self.step_sizes = [0.1, 0.01, 0.001]
        if self.step_sizes_deep is None:
            self.step_sizes_deep = [0.01, 0.001, 0.0001]


@dataclass
class ThresholdResult:
    """Container for threshold optimization results."""
    threshold: float
    score: float
    metrics: Dict[str, float]
    step_size: float
    
    def __repr__(self):
        return (f"ThresholdResult(threshold={self.threshold:.6f}, "
                f"score={self.score:.4f}, step={self.step_size:.6f})")


class MetricEvaluator:
    """Evaluates different metrics for threshold optimization."""
    
    METRIC_FUNCTIONS = {
        'AP50': lambda m: m['AP50'],
        'dice': lambda m: m['dice'],
        'f1': lambda m: m['f1'],
        'jaccard': lambda m: m['jaccard'],
        'precision': lambda m: m['precision'],
        'recall': lambda m: m['recall'],
        'recall_precision': lambda m: (m['recall'], m['precision'])  # Special case
    }
    
    def __init__(self, inference_engine: InferenceEngine, dataset):
        self.inference_engine = inference_engine
        self.dataset = dataset
        self.evaluation_cache = {}  # Cache to avoid re-computation
    
    def evaluate_threshold(
        self, 
        threshold: float, 
        metric_name: str,
        step_size: float = 0.0
    ) -> ThresholdResult:
        """
        Evaluate a specific threshold for a given metric.
        
        Args:
            threshold: The threshold value to evaluate
            metric_name: Name of the metric to optimize
            step_size: Step size used in search (for logging)
            
        Returns:
            ThresholdResult containing the evaluation results
        """
        threshold_rounded = round(threshold, 6)
        
        # Check cache first
        if threshold_rounded in self.evaluation_cache:
            cached_result = self.evaluation_cache[threshold_rounded]
            score = cached_result['metrics'][metric_name]
            return ThresholdResult(
                threshold=threshold_rounded,
                score=score,
                metrics=cached_result['metrics'],
                step_size=step_size
            )
        
        # Update inference engine configuration
        original_threshold = self.inference_engine.config.bbox_threshold
        self.inference_engine.config.bbox_threshold = threshold_rounded
        
        try:
            # Run inference
            results = self.inference_engine.process_dataset(
                self.dataset,
                output_path=".results/temp_threshold_search",
                test_mode=False
            )
            
            # Cache results
            self.evaluation_cache[threshold_rounded] = {
                'metrics': results,
                'threshold': threshold_rounded,
                'step_size': step_size
            }

            score = results[metric_name]
            
            return ThresholdResult(
                threshold=threshold_rounded,
                score=score,
                metrics=results,
                step_size=step_size
            )
            
        except Exception as e:
            logger.error(f"Error evaluating threshold {threshold_rounded}: {e}")
            # Return a result with score 0 to indicate failure
            return ThresholdResult(
                threshold=threshold_rounded,
                score=0.0,
                metrics={},
                step_size=step_size
            )
        finally:
            # Restore original threshold
            self.inference_engine.config.bbox_threshold = original_threshold


class AdaptiveThresholdSearcher:
    """Implements adaptive threshold search algorithms."""
    
    def __init__(
        self, 
        evaluator: MetricEvaluator, 
        config: ThresholdSearchConfig
    ):
        self.evaluator = evaluator
        self.config = config
        self.search_history = []
    
    def bidirectional_search(
        self, 
        metric_name: str, 
        is_deep_model: bool = False
    ) -> Tuple[ThresholdResult, List[ThresholdResult]]:
        """
        Perform bidirectional threshold search to find optimal value.
        
        Args:
            metric_name: Name of metric to optimize
            is_deep_model: Whether the model was trained for many epochs
            
        Returns:
            Tuple of (best_result, all_evaluation_results)
        """
        logger.info(f"Starting bidirectional search for {metric_name.upper()}")
        
        # Select appropriate configuration
        initial_threshold = (self.config.initial_threshold_deep if is_deep_model 
                           else self.config.initial_threshold)
        step_sizes = (self.config.step_sizes_deep if is_deep_model 
                     else self.config.step_sizes)
        
        # Initialize search
        current_best = self.evaluator.evaluate_threshold(
            initial_threshold, metric_name, step_sizes[0]
        )
        self.search_history.append(current_best)
        
        logger.info(f"Initial threshold: {current_best.threshold:.6f}, "
                   f"score: {current_best.score:.6f}")
        
        # Multi-level search with decreasing step sizes
        for step in step_sizes:
            logger.info(f"\n🔍 Searching with step size {step:.6f}")
            logger.info(f"Current best: threshold={current_best.threshold:.6f}, "
                       f"score={current_best.score:.6f}")
            
            level_best_score = current_best.score
            
            # Search left (decrease threshold)
            current_best = self._search_direction(
                current_best, metric_name, step, direction=-1
            )
            
            # Search right (increase threshold)  
            current_best = self._search_direction(
                current_best, metric_name, step, direction=1
            )
            
            # Allow to check all step sizes
            # # Check for convergence
            # if abs(current_best.score - level_best_score) < self.config.tolerance:
            #     logger.info(f"🛑 Converged at step size {step:.6f}")
            #     break
        
        logger.info(f"\n✅ Optimal {metric_name.upper()} threshold: "
                   f"{current_best.threshold:.6f} | "
                   f"Score: {current_best.score:.6f}")
        
        return current_best, self.search_history
    
    def _search_direction(
        self, 
        current_best: ThresholdResult, 
        metric_name: str, 
        step: float, 
        direction: int
    ) -> ThresholdResult:
        """
        Search in a specific direction (left or right).
        
        Args:
            current_best: Current best result
            metric_name: Metric to optimize
            step: Step size for search
            direction: -1 for left (decrease), 1 for right (increase)
            
        Returns:
            Updated best result
        """
        direction_name = "left" if direction == -1 else "right"
        arrow = "⬅️" if direction == -1 else "➡️"
        
        while True:
            new_threshold = current_best.threshold + (direction * step)
            new_threshold = round(new_threshold, 6)
            
            # Check bounds
            if new_threshold < self.config.min_threshold or new_threshold > self.config.max_threshold:
                break
            
            logger.info(f"Evaluating threshold: {new_threshold:.6f}")
            
            result = self.evaluator.evaluate_threshold(new_threshold, metric_name, step)
            self.search_history.append(result)
            
            # Check if this is better
            if self._is_better_result(result, current_best):
                logger.info(f"{arrow} Moving {direction_name}: {new_threshold:.6f} "
                           f"improves {metric_name.upper()} to {result.score:.6f}")
                current_best = result
            else:
                break  # Stop when performance drops
        
        return current_best
    
    def _is_better_result(
        self, 
        new_result: ThresholdResult, 
        current_best: ThresholdResult
    ) -> bool:
        """Determine if new result is better than current best."""
        return new_result.score > current_best.score


class ThresholdOptimizer:
    """Main class for threshold optimization with cross-validation support."""
    
    def __init__(
        self, 
        model, 
        base_config: InferenceConfig,
        search_config: Optional[ThresholdSearchConfig] = None
    ):
        self.model = model
        self.base_config = base_config
        self.search_config = search_config or ThresholdSearchConfig()
    
    def optimize(
        self,
        dataset,
        metrics: List[str] = None,
        is_deep_model: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform threshold optimization on the entire dataset.
        
        Args:
            dataset: Dataset for optimization
            metrics: List of metrics to optimize for
            is_deep_model: Whether model was trained extensively
            
        Returns:
            Dictionary containing optimization results for each metric
        """
        if metrics is None:
            metrics = ['AP50']

        logger.info(f"Starting threshold optimization on full dataset")
        logger.info(f"Optimizing for metrics: {metrics}")

        # Initialize components
        inference_engine = InferenceEngine(self.model, self.base_config)
        evaluator = MetricEvaluator(inference_engine, dataset)
        searcher = AdaptiveThresholdSearcher(evaluator, self.search_config)

        results = {}

        # Optimize for each metric
        for metric in metrics:
            logger.info(f"\n--- Optimizing {metric.upper()} ---")
            best_result, search_history = searcher.bidirectional_search(
                metric, is_deep_model
            )

            results[metric] = {
                'best_result': best_result,
                'search_history': search_history,
                'evaluation_cache': evaluator.evaluation_cache.copy()
            }

        # Log summary
        self._log_summary(results)

        return results

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of threshold optimization."""
        logger.info(f"\n📊 Optimization Summary:")
        for metric, result_data in results.items():
            best = result_data['best_result']
            logger.info(f"{metric.upper()}: threshold={best.threshold:.6f}, "
                        f"score={best.score:.6f}")

def optimize_thresholds(
    model,
    dataset,
    metrics: List[str] = None,
    is_deep_model: bool = False,
    **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function for threshold optimization.
    
    Args:
        model: The trained Cell-SAM model
        dataset: Dataset for optimization
        metrics: List of metrics to optimize
        is_deep_model: Whether model was trained extensively
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing optimization results
    """
    if metrics is None:
        metrics = ['AP50']
    
    # Create configurations
    inference_config = InferenceConfig(**config_kwargs)
    search_config = ThresholdSearchConfig()
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(model, inference_config, search_config)
    
    # Run optimization
    return optimizer.optimize(
        dataset=dataset,
        metrics=metrics,
        is_deep_model=is_deep_model
    )