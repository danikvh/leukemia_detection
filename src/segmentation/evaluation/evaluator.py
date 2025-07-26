"""
Main evaluation interface for instance segmentation.
Provides a unified interface for different evaluation methods.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from .metrics import InstanceMetrics, DeepCellEvaluator, COCOEvaluator


class SegmentationEvaluator:
    """
    Unified evaluator for instance segmentation that supports multiple evaluation methods.
    """
    
    def __init__(self, 
                 methods: List[str] = None,
                 iou_threshold: float = 0.5,
                 max_detections: List[int] = None):
        """
        Initialize evaluator with specified methods.
        
        Args:
            methods: List of evaluation methods ('hungarian', 'deepcell', 'coco')
            iou_threshold: IoU threshold for matching
            max_detections: Maximum detections for COCO evaluation
        """
        self.methods = methods or ['hungarian']
        self.iou_threshold = iou_threshold
        
        self.evaluators = {}
        self._initialize_evaluators(max_detections)
    
    def _initialize_evaluators(self, max_detections: Optional[List[int]]):
        """Initialize requested evaluators."""
        if 'hungarian' in self.methods:
            self.evaluators['hungarian'] = InstanceMetrics(self.iou_threshold)
        
        if 'deepcell' in self.methods:
            try:
                self.evaluators['deepcell'] = DeepCellEvaluator(self.iou_threshold)
            except ImportError:
                print("Warning: DeepCell not available, skipping deepcell evaluation")
        
        if 'coco' in self.methods:
            try:
                self.evaluators['coco'] = COCOEvaluator(max_detections)
            except ImportError:
                print("Warning: pycocotools not available, skipping COCO evaluation")
    
    def evaluate_single(self, 
                       pred_mask: Union[np.ndarray, torch.Tensor],
                       gt_mask: Union[np.ndarray, torch.Tensor],
                       **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single prediction-ground truth pair.
        
        Args:
            pred_mask: Predicted instance mask
            gt_mask: Ground truth instance mask
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary with results from each evaluation method
        """
        results = {}
        
        for method_name, evaluator in self.evaluators.items():
            try:
                results[method_name] = evaluator.evaluate(pred_mask, gt_mask, **kwargs)
            except Exception as e:
                print(f"Error in {method_name} evaluation: {e}")
                results[method_name] = {}
        
        return results
    
    def evaluate_batch(self,
                      pred_masks: List[Union[np.ndarray, torch.Tensor]],
                      gt_masks: List[Union[np.ndarray, torch.Tensor]],
                      aggregate: bool = True,
                      **kwargs) -> Dict[str, Union[Dict[str, float], List[Dict[str, float]]]]:
        """
        Evaluate a batch of predictions.
        
        Args:
            pred_masks: List of predicted instance masks
            gt_masks: List of ground truth instance masks
            aggregate: Whether to aggregate results across the batch
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary with aggregated or individual results from each method
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of predictions and ground truths must match")
        
        batch_results = {method: [] for method in self.evaluators.keys()}
        
        # Evaluate each pair
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            single_results = self.evaluate_single(pred_mask, gt_mask, **kwargs)
            for method, result in single_results.items():
                batch_results[method].append(result)
        
        if not aggregate:
            return batch_results
        
        # Aggregate results
        aggregated = {}
        for method, results_list in batch_results.items():
            if not results_list or not results_list[0]:  # Skip empty results
                continue
                
            # Get all metric names from first non-empty result
            metric_names = results_list[0].keys()
            aggregated[method] = {}
            
            for metric in metric_names:
                values = [r.get(metric, 0.0) for r in results_list if r]
                if values:
                    aggregated[method][metric] = np.mean(values)
                    aggregated[method][f"{metric}_std"] = np.std(values)
        
        return aggregated
    
    def get_summary_metrics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Extract key summary metrics from evaluation results.
        
        Args:
            results: Results from evaluate_single or evaluate_batch
            
        Returns:
            Dictionary with key metrics
        """
        summary = {}
        
        # Hungarian/Instance metrics
        if 'hungarian' in results:
            hungarian = results['hungarian']
            summary.update({
                'precision': hungarian.get('precision', 0.0),
                'recall': hungarian.get('recall', 0.0),
                'f1': hungarian.get('f1', 0.0),
                'jaccard': hungarian.get('jaccard', 0.0),
                'dice': hungarian.get('dice', 0.0)
            })
        
        # COCO metrics
        if 'coco' in results:
            coco = results['coco']
            summary.update({
                'AP': coco.get('AP', 0.0),
                'AP50': coco.get('AP50', 0.0),
                'AP75': coco.get('AP75', 0.0)
            })
        
        # DeepCell metrics (if different from Hungarian)
        if 'deepcell' in results and 'hungarian' not in results:
            deepcell = results['deepcell']
            summary.update({
                'precision': deepcell.get('precision', 0.0),
                'recall': deepcell.get('recall', 0.0),
                'f1': deepcell.get('f1', 0.0),
                'jaccard': deepcell.get('jaccard', 0.0),
                'dice': deepcell.get('dice', 0.0)
            })
        
        return summary
    
    def print_results(self, results: Dict[str, Dict[str, float]], title: str = "Evaluation Results"):
        """
        Print formatted evaluation results.
        
        Args:
            results: Results from evaluation
            title: Title for the results display
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        for method, metrics in results.items():
            print(f"\n{method.upper()} Metrics:")
            print("-" * (len(method) + 9))
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:<12}: {value:.4f}")
                else:
                    print(f"  {metric:<12}: {value}")