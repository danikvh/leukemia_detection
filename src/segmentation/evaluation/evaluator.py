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
                      scores_list: List[np.ndarray] = None,
                      **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of predictions using proper batch processing.
        
        Args:
            pred_masks: List of predicted instance masks
            gt_masks: List of ground truth instance masks
            scores_list: List of prediction scores (for COCO evaluation)
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary with aggregated results from each method
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Convert all masks to numpy arrays
        pred_masks_np = [self._to_numpy(mask) for mask in pred_masks]
        gt_masks_np = [self._to_numpy(mask) for mask in gt_masks]
        
        batch_results = {}
        
        # Hungarian method - proper batch aggregation
        if 'hungarian' in self.evaluators:
            try:
                batch_results['hungarian'] = self.evaluators['hungarian'].evaluate_batch(
                    pred_masks_np, gt_masks_np, **kwargs
                )
            except Exception as e:
                print(f"Error in hungarian batch evaluation: {e}")
                batch_results['hungarian'] = {}
        
        # DeepCell method - uses DeepCell's native batch processing
        if 'deepcell' in self.evaluators:
            try:
                batch_results['deepcell'] = self.evaluators['deepcell'].evaluate_batch(
                    pred_masks_np, gt_masks_np, **kwargs
                )
            except Exception as e:
                print(f"Error in deepcell batch evaluation: {e}")
                batch_results['deepcell'] = {}
        
        # COCO method - accumulate all annotations then evaluate
        if 'coco' in self.evaluators:
            try:
                batch_results['coco'] = self.evaluators['coco'].evaluate_batch(
                    pred_masks_np, gt_masks_np, scores_list, **kwargs
                )
            except Exception as e:
                print(f"Error in coco batch evaluation: {e}")
                batch_results['coco'] = {}
        
        return batch_results
    
    def _to_numpy(self, mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        return mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    
    def get_summary_metrics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Extract key summary metrics from evaluation results.
        Prioritizes metrics from the most comprehensive evaluation method available.
        
        Args:
            results: Results from evaluate_batch
            
        Returns:
            Dictionary with key metrics
        """
        summary = {}
        
        # Priority order: DeepCell -> Hungarian
        primary_method = None
        if 'deepcell' in results and results['deepcell']:
            primary_method = 'deepcell'
        elif 'hungarian' in results and results['hungarian']:
            primary_method = 'hungarian'
        
        # Extract core metrics from primary method
        if primary_method:
            primary = results[primary_method]
            summary.update({
                'precision': primary.get('precision', 0.0),
                'recall': primary.get('recall', 0.0),
                'f1': primary.get('f1', 0.0),
                'jaccard': primary.get('jaccard', 0.0),
                'dice': primary.get('dice', 0.0),
                'true_positives': primary.get('true_positives', 0),
                'false_positives': primary.get('false_positives', 0),
                'false_negatives': primary.get('false_negatives', 0)
            })
        
        # Add COCO metrics if available
        if 'coco' in results and results['coco']:
            coco = results['coco']
            summary.update({
                'AP': coco.get('AP', 0.0),
                'AP50': coco.get('AP50', 0.0),
                'AP75': coco.get('AP75', 0.0),
                'AP_small': coco.get('AP_small', 0.0),
                'AP_medium': coco.get('AP_medium', 0.0),
                'AP_large': coco.get('AP_large', 0.0)
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
            if not metrics:  # Skip empty results
                continue
                
            print(f"\n{method.upper()} Metrics:")
            print("-" * (len(method) + 9))
            
            # Core metrics first
            core_metrics = ['precision', 'recall', 'f1', 'jaccard', 'dice']
            ap_metrics = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
            count_metrics = ['true_positives', 'false_positives', 'false_negatives']
            
            # Print core metrics
            for metric in core_metrics:
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    print(f"  {metric:<15}: {metrics[metric]:.4f}")
            
            # Print AP metrics if available
            for metric in ap_metrics:
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    print(f"  {metric:<15}: {metrics[metric]:.4f}")
            
            # Print count metrics
            for metric in count_metrics:
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    print(f"  {metric:<15}: {int(metrics[metric])}")