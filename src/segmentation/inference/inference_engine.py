"""
Core inference engine for cellular segmentation using Cell-SAM.

This module provides the main inference functionality for processing cellular images
and generating segmentation masks with various evaluation metrics.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave
from cellSAM import segment_cellular_image
from deepcell.metrics import Metrics

from segmentation.utils.visualization import ImageVisualizer
from segmentation.evaluation.evaluator import SegmentationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceConfig:
    """Configuration class for inference parameters."""
    
    def __init__(
        self,
        bbox_threshold: float = 0.4,
        iou_threshold: float = 0.6,
        cf2_threshold: float = 0.1,
        normalize_inference: bool = True,
        postprocess: bool = False,
        save_outputs: bool = False,
        device: Optional[torch.device] = None,
        evaluation_methods: List[str] = None
    ):
        self.bbox_threshold = bbox_threshold
        self.iou_threshold = iou_threshold
        self.cf2_threshold = cf2_threshold
        self.normalize_inference = normalize_inference
        self.postprocess = postprocess
        self.save_outputs = save_outputs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluation_methods = evaluation_methods or ["deepcell", "coco"]


class SegmentationResult:
    """Container for segmentation results and metadata."""
    
    def __init__(
        self,
        mask: np.ndarray,
        scores: np.ndarray,
        filename: str,
        bbox_threshold: float,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        self.mask = mask
        self.scores = scores
        self.filename = filename
        self.bbox_threshold = bbox_threshold
        self.success = success
        self.error_message = error_message


class MetricsCalculator:
    """Handles calculation of various segmentation metrics using the new evaluator system."""
    
    def __init__(self, iou_threshold: float = 0.6, cf2_threshold: float = 0.1, evaluation_methods: List[str] = None):
        self.iou_threshold = iou_threshold
        self.cf2_threshold = cf2_threshold
        self.evaluation_methods = evaluation_methods or ["hungarian", "deepcell", "coco"]
        
        # Initialize the unified evaluator
        self.evaluator = SegmentationEvaluator(
            methods=self.evaluation_methods,
            iou_threshold=iou_threshold
        )
        
        # Keep the old deepcell metrics engine for batch processing compatibility
        self.metrics_engine = Metrics(
            model_name="CellSAM", 
            cutoff1=iou_threshold, 
            cutoff2=cf2_threshold
        )
    
    def calculate_object_metrics(
        self, 
        gt_masks: np.ndarray, 
        pred_masks: np.ndarray
    ) -> Tuple[Dict[str, float], Any]:
        """Calculate object-level metrics for segmentation results using batch evaluation."""
        logger.info("Calculating object-level segmentation metrics...")
        
        # Convert numpy arrays to lists for the evaluator
        gt_mask_list = [gt_masks[i] for i in range(gt_masks.shape[0])]
        pred_mask_list = [pred_masks[i] for i in range(pred_masks.shape[0])]

        # Use the new evaluator for batch processing
        batch_results = self.evaluator.evaluate_batch(
            pred_mask_list, 
            gt_mask_list, 
            aggregate=True
        )
        
        # Extract summary metrics
        metrics_summary = self.evaluator.get_summary_metrics(batch_results)

        # Also calculate using the old deepcell metrics for detailed DataFrame (if needed)
        try:
            object_metrics_df = self.metrics_engine.calc_object_stats(
                gt_masks, pred_masks, progbar=True
            )
        except Exception as e:
            logger.warning(f"DeepCell detailed metrics calculation failed: {e}")
            object_metrics_df = None
        
        return metrics_summary, object_metrics_df


class InferenceEngine:
    """Main inference engine for cellular segmentation using Cell-SAM."""
    
    def __init__(self, model, config: InferenceConfig):
        self.model = model
        self.config = config
        self.metrics_calculator = MetricsCalculator(
            config.iou_threshold, 
            config.cf2_threshold,
            config.evaluation_methods
        )
        
    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array in HWC uint8 format."""
        return image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    def _create_empty_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create an empty mask with the given shape."""
        return np.zeros(shape, dtype=np.uint8)
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Process and normalize the segmentation mask."""
        if mask is None:
            return None
        
        # Remove extra dimensions if present
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        
        return mask.astype(np.uint8)
    
    def segment_single_image(
        self, 
        image: torch.Tensor, 
        filename: str = "unknown"
    ) -> SegmentationResult:
        """
        Segment a single image using Cell-SAM with adaptive thresholding.
        
        Args:
            image: Input image tensor (C, H, W)
            filename: Optional filename for tracking
            
        Returns:
            SegmentationResult containing the segmentation mask and metadata
        """
        image_np = self._prepare_image(image)
        threshold = self.config.bbox_threshold
        
        logger.info(f"Processing image: {filename}")
        
        # Obtain mask
        try:
            mask, _, _, scores = segment_cellular_image(
                image_np,
                self.model,
                bbox_threshold=threshold,
                normalize=self.config.normalize_inference,
                device=self.config.device,
                postprocess=self.config.postprocess
            )
            
            if mask is not None:
                logger.info(f"✓ Segmentation succeeded at threshold={threshold:.3f}")
                processed_mask = self._process_mask(mask)
                return SegmentationResult(
                    mask=processed_mask,
                    scores=scores,
                    filename=filename,
                    bbox_threshold=threshold,
                    success=True
                )
                
        except Exception as e:
            # Attempt failed, return empty mask
            logger.warning(f"✗ Segmentation failed at threshold={threshold:.3f}: {e}")
            height, width = image.shape[1], image.shape[2]
            empty_mask = self._create_empty_mask((height, width))
        
            return SegmentationResult(
                mask=empty_mask,
                scores=np.array([]),
                filename=filename,
                bbox_threshold=self.config.bbox_threshold,
                success=False,
                error_message="Failed to generate segmentation at any threshold"
            )
    
    def _save_results(
        self, 
        result: SegmentationResult, 
        output_path: Path
    ) -> None:
        """Save segmentation results to disk."""
        if not self.config.save_outputs:
            return
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save numpy mask
        mask_path = output_path / f"{result.filename}.npy"
        np.save(mask_path, result.mask)
        
        # Save scores
        scores_path = output_path / f"{result.filename}_scores.npy"
        np.save(scores_path, result.scores)
        
        # Save PNG visualization
        png_path = output_path / f"{result.filename}.png"
        imsave(png_path, result.mask)
    
    def _load_cached_results(
        self, 
        filename: str, 
        output_path: Path
    ) -> Optional[SegmentationResult]:
        """Load cached segmentation results if available."""
        if not self.config.save_outputs:
            return None
            
        mask_path = output_path / f"{filename}.npy"
        scores_path = output_path / f"{filename}_scores.npy"
        
        if mask_path.exists() and scores_path.exists():
            logger.info(f"Loading cached results for {filename}")
            mask = np.load(mask_path)
            scores = np.load(scores_path)
            
            return SegmentationResult(
                mask=mask,
                scores=scores,
                filename=filename,
                bbox_threshold=self.config.bbox_threshold,
                success=True
            )
        
        return None
    
    def process_dataset(
        self, 
        dataset, 
        output_path: str = ".results/inference_output",
        test_mode: bool = False,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Process an entire dataset and return comprehensive metrics.
        
        Args:
            dataset: Dataset containing images and ground truth masks
            output_path: Directory to save results
            test_mode: If True, process only first 5 images
            visualize: If True, display intermediate visualizations
            
        Returns:
            Dictionary containing all calculated metrics
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Containers for batch metrics
        gt_masks = []
        pred_masks = []
        scores_list = []
        results = []
        single_evaluations = []
        
        dataset_size = min(5, len(dataset)) if test_mode else len(dataset)
        logger.info(f"Processing {dataset_size} images...")
        
        for i in range(dataset_size):
            image, gt_mask, filename = dataset[i]
            
            # Handle mask dimensions
            if gt_mask.ndimension() == 3:
                gt_mask = gt_mask.squeeze(0)
            
            # Load cached or compute new results
            cached_result = self._load_cached_results(filename, output_path)
            if cached_result:
                result = cached_result
            else:
                result = self.segment_single_image(image, filename)
                self._save_results(result, output_path)
            
            # Visualization if requested
            if visualize:
                visualizer = ImageVisualizer()
                visualizer.display_image_with_mask(image, gt_mask, torch.from_numpy(result.mask))
            
            # Store for batch processing
            gt_masks.append(gt_mask.cpu().numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask)
            pred_masks.append(result.mask)
            scores_list.append(result.scores)
            results.append(result)
            
            logger.info(f"Processed {i+1}/{dataset_size}: {filename}")
        
        # Calculate batch metrics using the new system
        batch_results = self.metrics_calculator.evaluator.evaluate_batch(
            pred_masks, 
            gt_masks, 
            scores_list=scores_list
        )
        
        # Extract summary metrics (matches original output format)
        summary_metrics = self.metrics_calculator.evaluator.get_summary_metrics(batch_results)
        
        final_results = {
            # Primary metrics (matches original return values)
            'precision': summary_metrics.get('precision', 0.0),
            'recall': summary_metrics.get('recall', 0.0),
            'f1': summary_metrics.get('f1', 0.0),  # This is the aggregated F1, not averaged
            'dice': summary_metrics.get('dice', 0.0),  # Average per-image pixel Dice
            'jaccard': summary_metrics.get('jaccard', 0.0),
            'AP': summary_metrics.get('AP', 0.0),
            'AP50': summary_metrics.get('AP50', 0.0),
            'AP75': summary_metrics.get('AP75', 0.0),
            
            # Additional metrics for compatibility
            'batch_metrics': batch_results,
            'summary_metrics': summary_metrics,
            'segmentation_results': results,
            'config': self.config.__dict__,
            
            # Statistics (matches original)
            'true_positives': summary_metrics.get('true_positives', 0),
            'false_positives': summary_metrics.get('false_positives', 0),
            'false_negatives': summary_metrics.get('false_negatives', 0),
        }
        
        # Add detailed DataFrame if available from DeepCell
        if 'deepcell' in batch_results and 'object_metrics_df' in batch_results['deepcell']:
            final_results['object_metrics_df'] = batch_results['deepcell']['object_metrics_df']
        
        self._log_final_results(final_results)
        return final_results
    
    def _log_final_results(self, results: Dict[str, Any]) -> None:
        """Log final results summary."""
        logger.info("\n" + "="*50)
        logger.info("FINAL SEGMENTATION RESULTS")
        logger.info("="*50)
        logger.info(f"Precision: {results.get('precision', 0.0):.4f}")
        logger.info(f"Recall: {results.get('recall', 0.0):.4f}")
        logger.info(f"F1-Score: {results.get('f1', 0.0):.4f}")
        logger.info(f"Dice Score: {results.get('dice', 0.0):.4f}")
        logger.info(f"Jaccard Score: {results.get('jaccard', 0.0):.4f}")
        logger.info(f"AP: {results.get('AP', 0.0):.4f}")
        logger.info(f"AP50: {results.get('AP50', 0.0):.4f}")
        logger.info(f"AP75: {results.get('AP75', 0.0):.4f}")
        logger.info("="*50)


def create_inference_engine(
    model, 
    bbox_threshold: float = 0.4,
    evaluation_methods: List[str] = None,
    **kwargs
) -> InferenceEngine:
    """Factory function to create an inference engine with default configuration."""
    config = InferenceConfig(
        bbox_threshold=bbox_threshold, 
        evaluation_methods=evaluation_methods,
        **kwargs
    )
    return InferenceEngine(model, config)