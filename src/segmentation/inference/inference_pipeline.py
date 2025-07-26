"""
High-level inference pipeline that combines all inference capabilities.

This module provides a unified interface for running complete inference
workflows including preprocessing, inference, evaluation, and optimization.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import torch

from .inference_engine import CellSAMInferenceEngine, InferenceConfig
from .threshold_optimizer import ThresholdOptimizer, ThresholdSearchConfig

logger = logging.getLogger(__name__)


@dataclass 
class BatchInferenceConfig:
    """Configuration for batch inference operations."""
    
    # Model and data
    model_path: Optional[str] = None
    dataset_path: str = ""
    output_base_path: str = "../results/inference_results"
    
    # Inference settings
    bbox_threshold: float = 0.4
    iou_threshold: float = 0.6
    cf2_threshold: float = 0.1
    normalize_inference: bool = False
    postprocess: bool = False
    
    # Processing options
    save_outputs: bool = True
    save_visualizations: bool = False
    test_mode: bool = False  # Process only subset for testing
    batch_size: int = 1
    
    # Optimization settings
    optimize_thresholds: bool = False
    optimization_metrics: List[str] = None
    cross_validation_folds: int = 5
    is_deep_model: bool = False
    
    # Device settings
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.optimization_metrics is None:
            self.optimization_metrics = ['ap50', 'dice', 'f1']
        
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class InferencePipeline:
    """
    High-level pipeline for comprehensive inference workflows.
    
    This class orchestrates the entire inference process from data loading
    to final result generation and optimization.
    """
    
    def __init__(self, config: BatchInferenceConfig):
        self.config = config
        self.output_path = Path(config.output_base_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize model (if path provided)
        self.model = None
        if config.model_path:
            self.model = self._load_model(config.model_path)
        
        # Setup inference engine
        self.inference_config = InferenceConfig(
            bbox_threshold=config.bbox_threshold,
            iou_threshold=config.iou_threshold,
            cf2_threshold=config.cf2_threshold,
            normalize_inference=config.normalize_inference,
            postprocess=config.postprocess,
            save_outputs=config.save_outputs,
            device=self.device
        )
        
        self.inference_engine = None
        if self.model:
            self.inference_engine = CellSAMInferenceEngine(self.model, self.inference_config)
    
    def _load_model(self, model_path: str):
        """Load the Cell-SAM model from checkpoint."""
        logger.info(f"Loading model from: {model_path}")
        try:
            # This would depend on your specific model loading logic
            # Replace with your actual model loading code
            checkpoint = torch.load(model_path, map_location=self.device)
            model = checkpoint.get('model', checkpoint)  # Handle different checkpoint formats
            model.to(self.device)
            model.eval()
            logger.info("✓ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    def load_dataset(self, dataset_path: str = None):
        """Load dataset for inference."""
        dataset_path = dataset_path or self.config.dataset_path
        
        # This would depend on your dataset loading logic
        # Replace with your actual dataset loading code
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Example placeholder - replace with your dataset loading
        # from ..datasets.dataset_factory import create_dataset
        # dataset = create_dataset(dataset_path, transform=your_transforms)
        
        logger.info("✓ Dataset loaded successfully")
        return None  # Return your actual dataset
    
    def run_inference(
        self, 
        dataset, 
        model=None, 
        save_detailed_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on the provided dataset.
        
        Args:
            dataset: Dataset to process
            model: Optional model (uses pipeline model if None)
            save_detailed_results: Whether to save detailed results to disk
            
        Returns:
            Dictionary containing all inference results
        """
        if model:
            # Create temporary inference engine with provided model
            temp_engine = CellSAMInferenceEngine(model, self.inference_config)
            engine = temp_engine
        else:
            if not self.inference_engine:
                raise ValueError("No model available for inference")
            engine = self.inference_engine
        
        logger.info("Starting inference pipeline...")
        
        # Run inference
        results = engine.process_dataset(
            dataset=dataset,
            output_path=str(self.output_path / "segmentation_masks"),
            test_mode=self.config.test_mode,
            visualize=self.config.save_visualizations
        )
        
        # Save results if requested
        if save_detailed_results:
            self._save_inference_results(results)
        
        logger.info("✓ Inference completed successfully")
        return results
    
    def run_threshold_optimization(
        self, 
        dataset, 
        model=None
    ) -> Dict[str, Any]:
        """
        Run threshold optimization on the dataset.
        
        Args:
            dataset: Dataset for optimization
            model: Optional model (uses pipeline model if None)
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.config.optimize_thresholds:
            logger.info("Threshold optimization disabled")
            return {}
        
        logger.info("Starting threshold optimization...")
        
        # Setup optimizer
        search_config = ThresholdSearchConfig()
        model_to_use = model or self.model
        
        if not model_to_use:
            raise ValueError("No model available for optimization")
        
        optimizer = ThresholdOptimizer(
            model=model_to_use,
            base_config=self.inference_config,
            search_config=search_config
        )
        
        # Run optimization
        optimization_results = optimizer.optimize_with_cross_validation(
            dataset=dataset,
            metrics=self.config.optimization_metrics,
            n_folds=self.config.cross_validation_folds,
            is_deep_model=self.config.is_deep_model
        )
        
        # Save optimization results
        self._save_optimization_results(optimization_results)
        
        logger.info("✓ Threshold optimization completed")
        return optimization_results
    
    def run_complete_pipeline(
        self, 
        dataset=None, 
        model=None
    ) -> Dict[str, Any]:
        """
        Run the complete inference pipeline including optimization.
        
        Args:
            dataset: Dataset to process (loads from config if None)
            model: Model to use (uses pipeline model if None)
            
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE INFERENCE PIPELINE")
        logger.info("="*60)
        
        # Load dataset if not provided
        if dataset is None:
            dataset = self.load_dataset()
        
        pipeline_results = {
            'config': asdict(self.config),
            'inference_results': {},
            'optimization_results': {}
        }
        
        try:
            # Run standard inference
            logger.info("\n--- PHASE 1: Standard Inference ---")
            inference_results = self.run_inference(dataset, model)
            pipeline_results['inference_results'] = inference_results
            
            # Run threshold optimization if enabled
            if self.config.optimize_thresholds:
                logger.info("\n--- PHASE 2: Threshold Optimization ---")
                optimization_results = self.run_threshold_optimization(dataset, model)
                pipeline_results['optimization_results'] = optimization_results
                
                # Run inference with optimized thresholds
                logger.info("\n--- PHASE 3: Optimized Inference ---")
                optimized_results = self._run_optimized_inference(
                    dataset, model, optimization_results
                )
                pipeline_results['optimized_inference_results'] = optimized_results
            
            # Save complete pipeline results
            self._save_pipeline_results(pipeline_results)
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _run_optimized_inference(
        self, 
        dataset, 
        model, 
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference with optimized thresholds for each metric."""
        optimized_results = {}
        
        # Extract best thresholds from optimization results
        aggregated = optimization_results.get('aggregated_results', {})
        
        for metric, stats in aggregated.items():
            logger.info(f"Running optimized inference for {metric.upper()}")
            
            optimal_threshold = stats['mean_threshold']
            
            # Create new inference config with optimal threshold
            optimized_config = InferenceConfig(
                bbox_threshold=optimal_threshold,
                iou_threshold=self.inference_config.iou_threshold,
                cf2_threshold=self.inference_config.cf2_threshold,
                normalize_inference=self.inference_config.normalize_inference,
                postprocess=self.inference_config.postprocess,
                save_outputs=False,  # Don't save intermediate results
                device=self.inference_config.device
            )
            
            # Create temporary engine and run inference
            temp_engine = CellSAMInferenceEngine(model, optimized_config)
            results = temp_engine.process_dataset(
                dataset=dataset,
                output_path=str(self.output_path / f"optimized_{metric}"),
                test_mode=self.config.test_mode
            )
            
            optimized_results[metric] = {
                'optimal_threshold': optimal_threshold,
                'results': results
            }
        
        return optimized_results
    
    def _save_inference_results(self, results: Dict[str, Any]) -> None:
        """Save inference results to disk."""
        results_file = self.output_path / "inference_results.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Inference results saved to: {results_file}")
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to disk."""
        results_file = self.output_path / "optimization_results.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Optimization results saved to: {results_file}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save complete pipeline results to disk."""
        results_file = self.output_path / "complete_pipeline_results.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Complete pipeline results saved to: {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # Convert anything else to string


def run_inference_pipeline(
    model_path: Optional[str] = None,
    dataset_path: str = "",
    output_path: str = "./inference_results",
    optimize_thresholds: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run a complete inference pipeline.
    
    Args:
        model_path: Path to model checkpoint
        dataset_path: Path to dataset
        output_path: Output directory for results
        optimize_thresholds: Whether to run threshold optimization
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing complete pipeline results
    """
    # Create configuration
    config = BatchInferenceConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        output_base_path=output_path,
        optimize_thresholds=optimize_thresholds,
        **kwargs
    )
    
    # Initialize and run pipeline
    pipeline = InferencePipeline(config)
    return pipeline.run_complete_pipeline()