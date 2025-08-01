"""
Instance segmentation evaluation metrics.
Provides different evaluation approaches for comparing predicted and ground truth masks.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
from scipy.stats import hmean


# Third-party evaluation libraries
from deepcell.metrics import Metrics
from deepcell_toolbox.metrics import ObjectMetrics
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray, **kwargs) -> Dict[str, float]:
        """Evaluate predicted vs ground truth masks."""
        pass
    
    @abstractmethod
    def evaluate_batch(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], **kwargs) -> Dict[str, float]:
        """Evaluate batch of predicted vs ground truth masks."""
        pass

    def _to_numpy(self, mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        return mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask


class InstanceMetrics(BaseEvaluator):
    """
    Hungarian algorithm-based instance segmentation evaluation.
    Uses optimal bipartite matching to compute precision, recall, F1, and IoU metrics.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def compute_iou_matrix(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute IoU matrix between all predicted and ground truth instances."""
        pred_labels = np.unique(pred_mask)[1:]  # skip background (0)
        gt_labels = np.unique(gt_mask)[1:]
        
        iou_matrix = np.zeros((len(gt_labels), len(pred_labels)))
        
        for i, g in enumerate(gt_labels):
            gt_bin = (gt_mask == g)
            for j, p in enumerate(pred_labels):
                pred_bin = (pred_mask == p)
                intersection = np.logical_and(gt_bin, pred_bin).sum()
                union = np.logical_or(gt_bin, pred_bin).sum()
                iou_matrix[i, j] = intersection / (union + 1e-8)
        
        return iou_matrix, gt_labels, pred_labels
    
    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate instance segmentation using Hungarian algorithm matching.
        
        Args:
            pred_mask: Predicted instance mask (H, W) with integer labels
            gt_mask: Ground truth instance mask (H, W) with integer labels
            
        Returns:
            Dictionary containing precision, recall, f1, jaccard, and dice scores
        """
        pred_mask = self._to_numpy(pred_mask)
        gt_mask = self._to_numpy(gt_mask)
        
        iou_matrix, gt_labels, pred_labels = self.compute_iou_matrix(pred_mask, gt_mask)
        
        if len(gt_labels) == 0 and len(pred_labels) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0, "dice": 1.0}
        
        if len(gt_labels) == 0 or len(pred_labels) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "dice": 0.0}
        
        # Match using Hungarian algorithm (maximize IoU)
        gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)
        
        tp = 0
        matched_ious = []
        
        for g, p in zip(gt_idx, pred_idx):
            iou = iou_matrix[g, p]
            if iou >= self.iou_threshold:
                tp += 1
                matched_ious.append(iou)
        
        fp = len(pred_labels) - tp
        fn = len(gt_labels) - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        jaccard = np.mean(matched_ious) if matched_ious else 0.0
        dice = np.mean([2*iou/(1+iou) for iou in matched_ious]) if matched_ious else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "dice": dice,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }



class DeepCellEvaluator(BaseEvaluator):
    """
    DeepCell toolbox-based evaluation.
    Provides comprehensive object-level metrics for cell segmentation.
    """
    
    def __init__(self, iou_threshold: float = 0.6):
        self.iou_threshold = iou_threshold
    
    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate using DeepCell ObjectMetrics.
        
        Args:
            pred_mask: Predicted instance mask
            gt_mask: Ground truth instance mask
            
        Returns:
            Dictionary containing precision, recall, f1, jaccard, and dice scores
        """
        pred_mask = self._to_numpy(pred_mask)
        gt_mask = self._to_numpy(gt_mask)
        
        try:
            metrics = ObjectMetrics(y_true=gt_mask, y_pred=pred_mask, cutoff1=self.iou_threshold)
            results = metrics.to_dict()
            
            return {
                "precision": results.get("precision", 0.0),
                "recall": results.get("recall", 0.0),
                "f1": results.get("f1", 0.0),
                "jaccard": results.get("jaccard", 0.0),
                "dice": results.get("dice", 0.0)
            }
        except Exception as e:
            print(f"DeepCell evaluation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "dice": 0.0}

    def evaluate_batch(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], **kwargs) -> Dict[str, float]:
        """
        Batch evaluation using DeepCell's proper batch processing.
        This replicates the original inference.py methodology exactly.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of predictions and ground truths must match")
        
        try:
            # Convert to numpy arrays and stack (original methodology)
            gt_batch = np.stack(gt_masks, axis=0)
            pred_batch = np.stack(pred_masks, axis=0)
            
            # Use DeepCell's batch processing (matches original)
            
            print("Calculating object statistics using DeepCell batch processing...")
            metrics = Metrics(model_name="custom", cutoff1=0.6, cutoff2=0.1) # cutoff1: iouthreshold, cutoff2: cf2threshold
            object_metrics_df = metrics.calc_object_stats(gt_batch, pred_batch, progbar=True)
            
            # Extract metrics using original methodology
            avg_dice = object_metrics_df['dice'].mean()
            avg_jaccard = object_metrics_df['jaccard'].mean()
            
            # Aggregate counts (original methodology)
            total_tp = object_metrics_df['correct_detections'].sum()
            total_n_true = object_metrics_df['n_true'].sum()
            total_n_pred = object_metrics_df['n_pred'].sum()
            total_fp = total_n_pred - total_tp
            total_fn = total_n_true - total_tp
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = hmean([precision, recall]) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": avg_jaccard,
                "dice": avg_dice,
                "true_positives": int(total_tp),
                "false_positives": int(total_fp),
                "false_negatives": int(total_fn),
                "object_metrics_df": object_metrics_df  # Return for compatibility
            }
            
        except Exception as e:
            print(f"DeepCell batch evaluation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "dice": 0.0}


class COCOEvaluator(BaseEvaluator):
    """
    COCO-style evaluation for instance segmentation.
    Provides AP metrics at different IoU thresholds and object sizes.
    """
    
    def __init__(self, max_detections: List[int] = None):
        self.max_detections = max_detections or [10, 100, 10000]  # Higher for cellular images
        self._global_ann_id = 0
    
    def _prepare_coco_annotations(self, 
                                gt_mask: np.ndarray, 
                                pred_mask: np.ndarray,
                                image_id: int,
                                scores: Optional[List[float]] = None,
                                category_id: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """Convert masks to COCO format annotations."""
        gt_annotations = []
        pred_annotations = []
        
        # Obtain all instance ids
        gt_instance_ids = np.unique(gt_mask)
        pred_instance_ids = np.unique(pred_mask)
        # Remove background
        gt_instance_ids = gt_instance_ids[gt_instance_ids != 0]
        pred_instance_ids = pred_instance_ids[pred_instance_ids != 0]

        for inst_id in gt_instance_ids:
            # Mask of the particular instance
            binary_mask = (gt_mask == inst_id).astype(np.uint8)
            rle = mask.encode(np.asfortranarray(binary_mask))
            # COCOeval requires string-encoded counts
            rle['counts'] = rle['counts'].decode('utf-8')
            
            area = mask.area(rle)
            if isinstance(area, np.ndarray):
                area = area[0]
            
            bbox = mask.toBbox(rle).tolist()
            
            gt_annotations.append({
                'id': self._global_ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rle,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
            })
            self._global_ann_id += 1
        
        for inst_id in pred_instance_ids:
            binary_mask = (pred_mask == inst_id).astype(np.uint8)
            rle = mask.encode(np.asfortranarray(binary_mask))
            
            area = mask.area(rle)
            if isinstance(area, np.ndarray):
                area = area[0]
            
            bbox = mask.toBbox(rle).tolist()
            if scores is None:
                score = 1.0
            else:
                # Obtain score for each prediction
                score = scores[0]
                scores = scores[1:]

            
            pred_annotations.append({
                'id': self._global_ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rle,
                'area': area,
                'bbox': bbox,
                'score': score
            })
            self._global_ann_id += 1
        
        return gt_annotations, pred_annotations
    
    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Single mask pair evaluation using COCO metrics.
        
        Args:
            pred_mask: Predicted instance mask
            gt_mask: Ground truth instance mask
            **kwargs: Additional arguments (scores, image_id)
            
        Returns:
            Dictionary containing AP metrics
        """
        pred_mask = self._to_numpy(pred_mask)
        gt_mask = self._to_numpy(gt_mask)
        
        image_id = kwargs.get('image_id', 0)
        scores = kwargs.get('scores', None)
        
        gt_anns, pred_anns = self._prepare_coco_annotations(gt_mask, pred_mask, image_id, scores)
        
        # Create minimal COCO dataset
        images_info = [{'id': image_id, 'width': gt_mask.shape[1], 'height': gt_mask.shape[0]}]
        categories = [{'id': 1, 'name': 'cell'}]
        
        return self.evaluate_batch([gt_anns], [pred_anns], images_info, categories)
    
    def evaluate_batch(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                      scores_list: List[np.ndarray] = None, **kwargs) -> Dict[str, float]:
        """
        Batch evaluation using COCO metrics.
        This replicates the original prepare_coco_format + evaluate_coco methodology.
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Reset counter
        self._global_ann_id = 0
        
        # Accumulate all annotations (original methodology)
        all_gt_annotations = []
        all_pred_annotations = []
        images_info = []
        
        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            pred_mask = self._to_numpy(pred_mask)
            gt_mask = self._to_numpy(gt_mask)
            
            # Image info
            images_info.append({
                "id": i,
                "width": gt_mask.shape[1],
                "height": gt_mask.shape[0],
                "file_name": f"image_{i}.png",
            })
            
            # Get scores for this image
            scores = scores_list[i] if scores_list and i < len(scores_list) else None
            
            # Convert to COCO format (original methodology)
            gt_ann, pred_ann = self._prepare_coco_annotations(gt_mask, pred_mask, i, scores)
            all_gt_annotations.extend(gt_ann)
            all_pred_annotations.extend(pred_ann)
        
        # Evaluate using COCO (original methodology)
        categories = [{"id": 1, "name": "cell"}]
        return self._evaluate_coco_batch(all_gt_annotations, all_pred_annotations, images_info, categories)
    
    def _evaluate_coco_batch(self, gt_annotations: List[Dict], pred_annotations: List[Dict],
                           images_info: List[Dict], categories: List[Dict]) -> Dict[str, float]:
        """
        Internal method that replicates the original evaluate_coco function exactly.
        """
        if not gt_annotations and not pred_annotations:
            return self._empty_results()
        
        try:
            # Create COCO ground truth (original methodology)
            coco_gt = COCO()
            coco_gt.dataset['images'] = images_info
            coco_gt.dataset['annotations'] = gt_annotations
            coco_gt.dataset['categories'] = categories
            coco_gt.createIndex()
            
            # Load predictions (original methodology)
            coco_dt = coco_gt.loadRes(pred_annotations)
            # Evaluate
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
            # Object density is much higher in cellular images than in natural images, modified the limit for the maximum number of detections from 100 to 10,000.
            coco_eval.params.maxDets = self.max_detections
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            return {
                "AP": coco_eval.stats[0],
                "AP50": coco_eval.stats[1],
                "AP75": coco_eval.stats[2],
                "AP_small": coco_eval.stats[3],
                "AP_medium": coco_eval.stats[4],
                "AP_large": coco_eval.stats[5],
            }
            
        except Exception as e:
            print(f"COCO evaluation failed: {e}")
            return self._empty_results()
    
    def _empty_results(self) -> Dict[str, float]:
        """Return empty results when evaluation fails."""
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }