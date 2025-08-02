"""
Memory-efficient cell inference script for unlabeled image folders.
Processes images in small batches to avoid CUDA out of memory errors.

Usage:
    python cell_inference_efficient.py --model-path models/binary_model.pth --config configs/binary_config.yaml --image-dir path/to/images/
    python cell_inference_efficient.py --model-path models/ternary_model.pth --config configs/ternary_config.yaml --image-dir path/to/images/ --batch-size 8
"""

import argparse
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Iterator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
import gc
from tqdm import tqdm

from classification.config import ClassificationConfig
from classification.inference import CellClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryEfficientCellClassifier(CellClassifier):
    """
    Memory-efficient version of CellClassifier that processes images in small batches
    and includes explicit memory management.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Clear any unnecessary memory after initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def classify_batch_efficient(self, img_inputs: List, batch_size: int = 128) -> List[Dict]:
        """
        Memory-efficient batch classification that processes images in smaller sub-batches.
        
        Args:
            img_inputs: List of image inputs (paths, arrays, or PIL images)
            batch_size: Maximum number of images to process simultaneously
            
        Returns:
            List of classification results
        """
        if not img_inputs:
            return []
        
        all_results = []
        total_batches = (len(img_inputs) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(img_inputs)} images in {total_batches} batches of size {batch_size}")
        
        # Process in small batches
        for i in tqdm(range(0, len(img_inputs), batch_size), desc="Processing batches"):
            batch_inputs = img_inputs[i:i + batch_size]
            
            try:
                # Process current batch
                batch_results = self._process_single_batch(batch_inputs)
                all_results.extend(batch_results)
                
                # Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM in batch {i//batch_size + 1}. Falling back to single image processing.")
                # Fallback to processing images one by one for this batch
                for img_input in batch_inputs:
                    try:
                        result = self.classify_single_image(img_input)
                        all_results.append(result)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as single_error:
                        logger.error(f"Error processing single image: {single_error}")
                        # Add a placeholder result for failed images
                        all_results.append({
                            "prediction": "error",
                            "probability": 0.0,
                            "confidence": 0.0,
                            "error": str(single_error)
                        })
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add placeholder results for the entire failed batch
                for _ in batch_inputs:
                    all_results.append({
                        "prediction": "error",
                        "probability": 0.0,
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        return all_results
    
    def _process_single_batch(self, batch_inputs: List) -> List[Dict]:
        """Process a single batch of images efficiently."""
        # Preprocess all images in the batch
        try:
            processed_tensors = []
            valid_indices = []
            
            for idx, img_input in enumerate(batch_inputs):
                try:
                    tensor = self._preprocess_image(img_input).squeeze(0)  # Remove batch dim
                    processed_tensors.append(tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to preprocess image {idx}: {e}")
                    continue
            
            if not processed_tensors:
                return [{"prediction": "error", "probability": 0.0, "confidence": 0.0, 
                        "error": "No valid images in batch"} for _ in batch_inputs]
            
            # Check tensor shapes
            if not all(t.shape == processed_tensors[0].shape for t in processed_tensors):
                logger.warning("Images in batch have different shapes. Processing individually.")
                results = []
                for idx, img_input in enumerate(batch_inputs):
                    try:
                        result = self.classify_single_image(img_input)
                        results.append(result)
                    except Exception as e:
                        results.append({"prediction": "error", "probability": 0.0, 
                                      "confidence": 0.0, "error": str(e)})
                return results
            
            # Stack tensors and process
            batch_tensor = torch.stack(processed_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                if self.config.classification_mode == "binary":
                    batch_results = self._process_binary_batch_output(outputs)
                else:
                    batch_results = self._process_ternary_batch_output(outputs)
            
            # Map results back to original indices
            final_results = []
            result_idx = 0
            for original_idx in range(len(batch_inputs)):
                if original_idx in valid_indices:
                    final_results.append(batch_results[result_idx])
                    result_idx += 1
                else:
                    final_results.append({"prediction": "error", "probability": 0.0, 
                                        "confidence": 0.0, "error": "Preprocessing failed"})
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in _process_single_batch: {e}")
            return [{"prediction": "error", "probability": 0.0, "confidence": 0.0, 
                    "error": str(e)} for _ in batch_inputs]


class MemoryEfficientCellAnalyzer:
    """
    Memory-efficient analyzer that processes images in batches and includes
    progress tracking and memory monitoring.
    """
    
    def __init__(self, classifier: MemoryEfficientCellClassifier):
        self.classifier = classifier
        self.config = classifier.config
    
    def analyze_image_folder(self, image_dir: Path, 
                           extensions: List[str] = None,
                           batch_size: int = 8,
                           save_individual_results: bool = False,
                           save_progress: bool = True) -> Dict[str, Any]:
        """
        Analyze all images in a folder with memory-efficient processing.
        
        Args:
            image_dir: Directory containing images to analyze
            extensions: List of file extensions to process
            batch_size: Number of images to process in each batch
            save_individual_results: Whether to save detailed results for each image
            save_progress: Whether to save intermediate progress
            
        Returns:
            Dict containing analysis results
        """
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"]
        
        logger.info(f"Analyzing images in {image_dir} with batch size {batch_size}...")
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
        
        if not image_paths:
            logger.error(f"No images found in {image_dir} with extensions {extensions}")
            return {}
        
        logger.info(f"Found {len(image_paths)} images to analyze")
        
        # Log memory usage if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Perform efficient batch classification
        logger.info(f"Running memory-efficient inference with batch size {batch_size}...")
        results = self.classifier.classify_batch_efficient(image_paths, batch_size=batch_size)
        
        # Log final memory usage
        if torch.cuda.is_available():
            logger.info(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Filter out error results for analysis (but keep them in detailed results)
        valid_results = [r for r in results if r.get('prediction') != 'error']
        error_count = len(results) - len(valid_results)
        
        if error_count > 0:
            logger.warning(f"{error_count} images failed to process")
        
        if not valid_results:
            logger.error("No images were successfully processed")
            return {
                'error': f'Failed to process any of {len(results)} images',
                'error_count': error_count,
                'total_images': len(results)
            }
        
        # Combine results with image information
        detailed_results = []
        for i, (image_path, result) in enumerate(zip(image_paths, results)):
            try:
                file_size = image_path.stat().st_size if image_path.exists() else 0
            except:
                file_size = 0
                
            detailed_result = {
                "filename": image_path.name,
                "filepath": str(image_path),
                "file_size_bytes": file_size,
                "processing_order": i + 1,
                **result
            }
            detailed_results.append(detailed_result)
        
        # Generate comprehensive analysis using only valid results
        analysis = self._generate_comprehensive_analysis(valid_results, len(results), error_count)
        
        # Add processing metadata
        analysis['processing_metadata'] = {
            'batch_size_used': batch_size,
            'total_images_found': len(image_paths),
            'successfully_processed': len(valid_results),
            'processing_errors': error_count,
            'error_rate': error_count / len(results) * 100 if results else 0
        }
        
        # Add individual results if requested
        if save_individual_results:
            analysis['individual_results'] = detailed_results
        
        return analysis
    
    def _generate_comprehensive_analysis(self, valid_results: List[Dict], 
                                       total_attempted: int, error_count: int) -> Dict[str, Any]:
        """Generate comprehensive analysis from valid classification results."""
        logger.info(f"Generating analysis from {len(valid_results)} valid results...")
        
        if not valid_results:
            return {
                'error': 'No valid results to analyze',
                'total_attempted': total_attempted,
                'error_count': error_count
            }
        
        # Extract predictions and probabilities
        predictions = [r['prediction'] for r in valid_results]
        
        # Count predictions
        prediction_counts = Counter(predictions)
        total_valid = len(valid_results)
        
        # Calculate basic statistics
        basic_stats = {
            'total_images_analyzed': total_valid,
            'total_images_attempted': total_attempted,
            'processing_errors': error_count,
            'success_rate': (total_valid / total_attempted * 100) if total_attempted > 0 else 0,
            'prediction_counts': dict(prediction_counts),
            'prediction_percentages': {
                pred: count / total_valid * 100 
                for pred, count in prediction_counts.items()
            }
        }
        
        # Calculate relative percentages excluding uncertain predictions
        decided_predictions = [p for p in predictions if p != 'uncertain']
        decided_count = len(decided_predictions)
        
        if decided_count > 0:
            decided_prediction_counts = Counter(decided_predictions)
            relative_percentages = {
                pred: count / decided_count * 100 
                for pred, count in decided_prediction_counts.items()
            }
        else:
            decided_prediction_counts = Counter()
            relative_percentages = {}
        
        # Add relative analysis to basic stats
        basic_stats.update({
            'decided_predictions_count': decided_count,
            'decided_predictions_percentage': (decided_count / total_valid * 100) if total_valid > 0 else 0,
            'decided_prediction_counts': dict(decided_prediction_counts),
            'relative_percentages_excluding_uncertain': relative_percentages
        })
        
        # Generate mode-specific analysis
        if self.config.classification_mode == "binary":
            detailed_analysis = self._analyze_binary_results(valid_results)
        else:
            detailed_analysis = self._analyze_ternary_results(valid_results)
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_statistics(valid_results)
        
        # Generate summary insights
        insights = self._generate_insights(prediction_counts, total_valid, detailed_analysis, decided_count)
        
        return {
            'analysis_metadata': {
                'classification_mode': self.config.classification_mode,
                'model_info': self.classifier.get_model_info(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'basic_statistics': basic_stats,
            'detailed_analysis': detailed_analysis,
            'confidence_statistics': confidence_stats,
            'insights_and_recommendations': insights
        }
    
    def _analyze_binary_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for binary classification."""
        # Separate results by prediction
        cancerous = [r for r in results if r['prediction'] == 'cancerous']
        non_cancerous = [r for r in results if r['prediction'] == 'non-cancerous']
        uncertain = [r for r in results if r['prediction'] == 'uncertain']
        
        # Calculate probability statistics
        all_probs = [r['probability'] for r in results]
        
        # Calculate decided-only statistics
        decided_results = cancerous + non_cancerous
        decided_probs = [r['probability'] for r in decided_results]
        
        detailed_analysis = {
            'cancerous_cells': {
                'count': len(cancerous),
                'percentage': len(cancerous) / len(results) * 100,
                'stats': self._calculate_prob_stats([r['probability'] for r in cancerous]),
                'high_confidence_count': len([r for r in cancerous if r.get('confidence', 0) >= 0.8])
            },
            'non_cancerous_cells': {
                'count': len(non_cancerous),
                'percentage': len(non_cancerous) / len(results) * 100,
                'stats': self._calculate_prob_stats([r['probability'] for r in non_cancerous]),
                'high_confidence_count': len([r for r in non_cancerous if r.get('confidence', 0) >= 0.8])
            },
            'uncertain_cells': {
                'count': len(uncertain),
                'percentage': len(uncertain) / len(results) * 100,
                'stats': self._calculate_prob_stats([r['probability'] for r in uncertain])
            },
            'overall_probability_distribution': self._calculate_prob_stats(all_probs),
            'decided_only_analysis': {
                'count': len(decided_results),
                'percentage': len(decided_results) / len(results) * 100 if results else 0,
                'relative_cancerous_percentage': len(cancerous) / len(decided_results) * 100 if decided_results else 0,
                'relative_non_cancerous_percentage': len(non_cancerous) / len(decided_results) * 100 if decided_results else 0,
                'probability_distribution': self._calculate_prob_stats(decided_probs)
            }
        }
        
        return detailed_analysis
    
    def _analyze_ternary_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for ternary classification."""
        # Separate results by prediction
        cancerous = [r for r in results if r['prediction'] == 'cancerous']
        non_cancerous = [r for r in results if r['prediction'] == 'non-cancerous']
        false_positive = [r for r in results if r['prediction'] == 'false-positive']
        uncertain = [r for r in results if r['prediction'] == 'uncertain']
        
        # Calculate decided-only statistics
        decided_results = cancerous + non_cancerous + false_positive
        decided_count = len(decided_results)
        
        def analyze_class(class_results, class_name):
            if not class_results:
                return {
                    'count': 0,
                    'percentage': 0,
                    'confidence_stats': {},
                    'high_confidence_count': 0
                }
            
            confidences = [r.get('confidence', 0) for r in class_results]
            class_probs = []
            
            # Extract class-specific probabilities if available
            for r in class_results:
                if 'class_probabilities' in r and class_name in r['class_probabilities']:
                    class_probs.append(r['class_probabilities'][class_name])
            
            return {
                'count': len(class_results),
                'percentage': len(class_results) / len(results) * 100,
                'relative_percentage': len(class_results) / decided_count * 100 if decided_count > 0 else 0,
                'confidence_stats': self._calculate_prob_stats(confidences),
                'class_probability_stats': self._calculate_prob_stats(class_probs) if class_probs else {},
                'high_confidence_count': len([r for r in class_results if r.get('confidence', 0) >= 0.7])
            }
        
        detailed_analysis = {
            'cancerous_cells': analyze_class(cancerous, 'cancerous'),
            'non_cancerous_cells': analyze_class(non_cancerous, 'non-cancerous'),
            'false_positive_cells': analyze_class(false_positive, 'false-positive'),
            'uncertain_cells': {
                'count': len(uncertain),
                'percentage': len(uncertain) / len(results) * 100 if results else 0,
                'confidence_stats': self._calculate_prob_stats([r.get('confidence', 0) for r in uncertain])
            },
            'overall_confidence_distribution': self._calculate_prob_stats([r.get('confidence', 0) for r in results]),
            'decided_only_analysis': {
                'count': decided_count,
                'percentage': decided_count / len(results) * 100 if results else 0,
                'relative_cancerous_percentage': len(cancerous) / decided_count * 100 if decided_count > 0 else 0,
                'relative_non_cancerous_percentage': len(non_cancerous) / decided_count * 100 if decided_count > 0 else 0,
                'relative_false_positive_percentage': len(false_positive) / decided_count * 100 if decided_count > 0 else 0,
                'confidence_distribution': self._calculate_prob_stats([r.get('confidence', 0) for r in decided_results])
            }
        }
        
        return detailed_analysis
    
    def _calculate_prob_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of probability/confidence values."""
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q25': 0.0,
                'q75': 0.0
            }
        
        values = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _calculate_confidence_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence-related statistics."""
        confidences = [r.get('confidence', 0) for r in results]
        
        # Define confidence levels
        high_confidence = [c for c in confidences if c >= 0.8]
        medium_confidence = [c for c in confidences if 0.5 <= c < 0.8]
        low_confidence = [c for c in confidences if c < 0.5]
        
        return {
            'overall_confidence': self._calculate_prob_stats(confidences),
            'confidence_levels': {
                'high_confidence': {
                    'count': len(high_confidence),
                    'percentage': len(high_confidence) / len(confidences) * 100 if confidences else 0
                },
                'medium_confidence': {
                    'count': len(medium_confidence),
                    'percentage': len(medium_confidence) / len(confidences) * 100 if confidences else 0
                },
                'low_confidence': {
                    'count': len(low_confidence),
                    'percentage': len(low_confidence) / len(confidences) * 100 if confidences else 0
                }
            }
        }
    
    def _generate_insights(self, prediction_counts: Counter, total_images: int, 
                         detailed_analysis: Dict, decided_count: int) -> Dict[str, Any]:
        """Generate insights and recommendations based on results."""
        insights = {
            'summary': {},
            'recommendations': []
        }
        
        if self.config.classification_mode == "binary":
            cancerous_pct = prediction_counts.get('cancerous', 0) / total_images * 100
            uncertain_pct = prediction_counts.get('uncertain', 0) / total_images * 100
            
            # Calculate relative percentages among decided predictions
            if decided_count > 0:
                relative_cancerous_pct = prediction_counts.get('cancerous', 0) / decided_count * 100
                relative_non_cancerous_pct = prediction_counts.get('non-cancerous', 0) / decided_count * 100
            else:
                relative_cancerous_pct = 0
                relative_non_cancerous_pct = 0
            
            insights['summary'] = {
                'primary_finding': self._get_primary_binary_finding(prediction_counts, total_images),
                'confidence_assessment': self._assess_confidence_from_analysis(detailed_analysis),
                'uncertainty_level': uncertain_pct,
                'decided_percentage': decided_count / total_images * 100 if total_images > 0 else 0,
                'relative_findings_among_decided': {
                    'cancerous_percentage': relative_cancerous_pct,
                    'non_cancerous_percentage': relative_non_cancerous_pct
                }
            }
            
            insights['recommendations'] = self._generate_binary_recommendations(
                prediction_counts, total_images, detailed_analysis, decided_count
            )
        else:
            uncertain_pct = prediction_counts.get('uncertain', 0) / total_images * 100
            
            # Calculate relative percentages among decided predictions
            if decided_count > 0:
                relative_cancerous_pct = prediction_counts.get('cancerous', 0) / decided_count * 100
                relative_non_cancerous_pct = prediction_counts.get('non-cancerous', 0) / decided_count * 100
                relative_false_pos_pct = prediction_counts.get('false-positive', 0) / decided_count * 100
            else:
                relative_cancerous_pct = relative_non_cancerous_pct = relative_false_pos_pct = 0
            
            insights['summary'] = {
                'primary_finding': self._get_primary_ternary_finding(prediction_counts, total_images),
                'confidence_assessment': self._assess_confidence_from_analysis(detailed_analysis),
                'uncertainty_level': uncertain_pct,
                'decided_percentage': decided_count / total_images * 100 if total_images > 0 else 0,
                'relative_findings_among_decided': {
                    'cancerous_percentage': relative_cancerous_pct,
                    'non_cancerous_percentage': relative_non_cancerous_pct,
                    'false_positive_percentage': relative_false_pos_pct
                }
            }
            
            insights['recommendations'] = self._generate_ternary_recommendations(
                prediction_counts, total_images, detailed_analysis, decided_count
            )
        
        return insights
    
    def _get_primary_binary_finding(self, counts: Counter, total: int) -> str:
        """Generate primary finding summary for binary classification."""
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        non_cancerous_pct = counts.get('non-cancerous', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        # Calculate relative percentages among decided predictions
        decided_count = counts.get('cancerous', 0) + counts.get('non-cancerous', 0)
        if decided_count > 0:
            relative_cancerous = counts.get('cancerous', 0) / decided_count * 100
        else:
            relative_cancerous = 0
        
        if cancerous_pct > 50:
            return f"High prevalence of cancerous cells detected ({cancerous_pct:.1f}% overall, {relative_cancerous:.1f}% among decided)"
        elif cancerous_pct > 20:
            return f"Moderate prevalence of cancerous cells detected ({cancerous_pct:.1f}% overall, {relative_cancerous:.1f}% among decided)"
        elif uncertain_pct > 30:
            return f"High uncertainty in classification ({uncertain_pct:.1f}% uncertain)"
        else:
            return f"Predominantly non-cancerous cells ({non_cancerous_pct:.1f}% overall, {100-relative_cancerous:.1f}% among decided)"
    
    def _get_primary_ternary_finding(self, counts: Counter, total: int) -> str:
        """Generate primary finding summary for ternary classification."""
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        false_pos_pct = counts.get('false-positive', 0) / total * 100
        non_cancerous_pct = counts.get('non-cancerous', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        # Calculate relative percentages among decided predictions
        decided_count = counts.get('cancerous', 0) + counts.get('non-cancerous', 0) + counts.get('false-positive', 0)
        if decided_count > 0:
            relative_cancerous = counts.get('cancerous', 0) / decided_count * 100
        else:
            relative_cancerous = 0
        
        if cancerous_pct > 30:
            return f"High prevalence of cancerous cells detected ({cancerous_pct:.1f}% overall, {relative_cancerous:.1f}% among decided)"
        elif false_pos_pct > 20:
            return f"Significant false-positive detections ({false_pos_pct:.1f}%)"
        elif uncertain_pct > 25:
            return f"High classification uncertainty ({uncertain_pct:.1f}%)"
        else:
            return f"Mixed population with {non_cancerous_pct:.1f}% non-cancerous cells ({100-relative_cancerous:.1f}% among decided)"
    
    def _assess_confidence_from_analysis(self, analysis: Dict) -> str:
        """Assess overall confidence from detailed analysis."""
        if self.config.classification_mode == "binary":
            # Use probability statistics
            overall_stats = analysis.get('overall_probability_distribution', {})
            mean_prob = overall_stats.get('mean', 0.5)
            std_prob = overall_stats.get('std', 0)
            
            if std_prob < 0.2 and (mean_prob > 0.7 or mean_prob < 0.3):
                return f"High confidence (mean prob: {mean_prob:.3f}, low variance)"
            elif std_prob < 0.3:
                return f"Moderate confidence (mean prob: {mean_prob:.3f})"
            else:
                return f"Variable confidence (high variance: {std_prob:.3f})"
        else:
            # Use confidence statistics
            overall_stats = analysis.get('overall_confidence_distribution', {})
            mean_conf = overall_stats.get('mean', 0)
            
            if mean_conf > 0.7:
                return f"High overall confidence (mean: {mean_conf:.3f})"
            elif mean_conf > 0.5:
                return f"Moderate overall confidence (mean: {mean_conf:.3f})"
            else:
                return f"Low overall confidence (mean: {mean_conf:.3f})"
    
    def _generate_binary_recommendations(self, counts: Counter, total: int, 
                                       analysis: Dict, decided_count: int) -> List[str]:
        """Generate recommendations for binary classification results."""
        recommendations = []
        
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        
        if uncertain_pct > 30:
            recommendations.append(
                "High uncertainty detected. Consider manual review of uncertain cases "
                "or threshold optimization."
            )
        
        if cancerous_pct > 25:
            recommendations.append(
                "Significant number of cancerous cells detected. "
                "Prioritize these samples for clinical review."
            )
        
        # Relative analysis recommendations
        if decided_count > 0:
            relative_cancerous = counts.get('cancerous', 0) / decided_count * 100
            if relative_cancerous > 50:
                recommendations.append(
                    f"Among decided predictions, {relative_cancerous:.1f}% are cancerous. "
                    "Focus clinical attention on this high-risk subset."
                )
        
        cancerous_high_conf = analysis.get('cancerous_cells', {}).get('high_confidence_count', 0)
        if cancerous_high_conf > 0:
            recommendations.append(
                f"Focus on {cancerous_high_conf} high-confidence cancerous predictions "
                "for immediate attention."
            )
        
        return recommendations
    
    def _generate_ternary_recommendations(self, counts: Counter, total: int, 
                                        analysis: Dict, decided_count: int) -> List[str]:
        """Generate recommendations for ternary classification results."""
        recommendations = []
        
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        false_pos_pct = counts.get('false-positive', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        if cancerous_pct > 20:
            high_conf_cancerous = analysis.get('cancerous_cells', {}).get('high_confidence_count', 0)
            recommendations.append(
                f"Prioritize {high_conf_cancerous} high-confidence cancerous cases "
                "for clinical review."
            )
        
        # Relative analysis recommendations
        if decided_count > 0:
            relative_cancerous = counts.get('cancerous', 0) / decided_count * 100
            if relative_cancerous > 40:
                recommendations.append(
                    f"Among decided predictions, {relative_cancerous:.1f}% are cancerous. "
                    "This suggests a high-risk sample population."
                )
        
        if false_pos_pct > 15:
            recommendations.append(
                f"High false-positive rate ({false_pos_pct:.1f}%) detected. "
                "Consider reviewing segmentation quality."
            )
        
        if uncertain_pct > 25:
            recommendations.append(
                "High uncertainty suggests need for manual review or threshold optimization."
            )
        
        return recommendations
    
    def save_results(self, analysis_results: Dict, output_path: Path, 
                    create_visualizations: bool = True) -> None:
        """Save analysis results to files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main results to JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")
        
        # Create summary report
        self._create_summary_report(analysis_results, output_path.parent)
        
        if create_visualizations and 'basic_statistics' in analysis_results:
            try:
                self._create_enhanced_visualizations(analysis_results, output_path.parent)
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")
    
    def _create_summary_report(self, analysis: Dict, output_dir: Path) -> None:
        """Create a human-readable summary report."""
        report_path = output_dir / "analysis_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("MEMORY-EFFICIENT CELL CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Processing metadata
            if 'processing_metadata' in analysis:
                proc_meta = analysis['processing_metadata']
                f.write("PROCESSING SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Batch size used: {proc_meta.get('batch_size_used', 'N/A')}\n")
                f.write(f"Images found: {proc_meta.get('total_images_found', 'N/A')}\n")
                f.write(f"Successfully processed: {proc_meta.get('successfully_processed', 'N/A')}\n")
                f.write(f"Processing errors: {proc_meta.get('processing_errors', 'N/A')}\n")
                f.write(f"Success rate: {100 - proc_meta.get('error_rate', 0):.1f}%\n\n")
            
            # Basic statistics
            if 'basic_statistics' in analysis:
                basic_stats = analysis['basic_statistics']
                f.write(f"Images Successfully Analyzed: {basic_stats.get('total_images_analyzed', 'N/A')}\n")
                f.write(f"Classification Mode: {analysis.get('analysis_metadata', {}).get('classification_mode', 'N/A')}\n")
                f.write(f"Analysis Date: {analysis.get('analysis_metadata', {}).get('analysis_timestamp', 'N/A')}\n\n")
                
                # Prediction counts
                f.write("PREDICTION DISTRIBUTION (ALL PREDICTIONS):\n")
                f.write("-" * 40 + "\n")
                for pred, count in basic_stats.get('prediction_counts', {}).items():
                    percentage = basic_stats.get('prediction_percentages', {}).get(pred, 0)
                    f.write(f"{pred.capitalize()}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Decided-only distribution
                decided_count = basic_stats.get('decided_predictions_count', 0)
                decided_pct = basic_stats.get('decided_predictions_percentage', 0)
                f.write("RELATIVE DISTRIBUTION (EXCLUDING UNCERTAIN):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Decided Predictions: {decided_count} ({decided_pct:.1f}%)\n")
                
                for pred, percentage in basic_stats.get('relative_percentages_excluding_uncertain', {}).items():
                    count = basic_stats.get('decided_prediction_counts', {}).get(pred, 0)
                    f.write(f"{pred.capitalize()}: {count} ({percentage:.1f}% of decided)\n")
                f.write("\n")
            
            # Key findings
            if 'insights_and_recommendations' in analysis:
                insights = analysis['insights_and_recommendations']
                f.write("KEY FINDINGS:\n")
                f.write("-" * 30 + "\n")
                summary = insights.get('summary', {})
                f.write(f"Primary Finding: {summary.get('primary_finding', 'N/A')}\n")
                f.write(f"Confidence Assessment: {summary.get('confidence_assessment', 'N/A')}\n")
                f.write(f"Uncertainty Level: {summary.get('uncertainty_level', 0):.1f}%\n")
                f.write(f"Decided Predictions: {summary.get('decided_percentage', 0):.1f}%\n\n")
                
                # Relative findings
                if 'relative_findings_among_decided' in summary:
                    rel_findings = summary['relative_findings_among_decided']
                    f.write("RELATIVE FINDINGS AMONG DECIDED PREDICTIONS:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in rel_findings.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value:.1f}%\n")
                    f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(insights.get('recommendations', []), 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def _create_enhanced_visualizations(self, analysis: Dict, output_dir: Path) -> None:
        """Create enhanced visualization plots including relative analysis and probability densities."""
        logger.info("Creating enhanced visualization plots...")
        
        basic_stats = analysis.get('basic_statistics', {})
        if not basic_stats.get('prediction_counts'):
            logger.warning("No prediction data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Enhanced Memory-Efficient Cell Classification Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Overall prediction distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        counts = basic_stats['prediction_counts']
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Overall Prediction Distribution', fontweight='bold')
        
        # Plot 2: Relative distribution excluding uncertain (pie chart)
        ax2 = fig.add_subplot(gs[0, 1])
        decided_counts = basic_stats.get('decided_prediction_counts', {})
        if decided_counts:
            rel_labels = list(decided_counts.keys())
            rel_sizes = list(decided_counts.values())
            rel_colors = plt.cm.Dark2(np.linspace(0, 1, len(rel_labels)))
            
            wedges, texts, autotexts = ax2.pie(rel_sizes, labels=rel_labels, autopct='%1.1f%%', 
                                              colors=rel_colors, startangle=90)
            ax2.set_title('Relative Distribution\n(Excluding Uncertain)', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No decided\npredictions', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Relative Distribution\n(Excluding Uncertain)', fontweight='bold')
        
        # Plot 3: Comparison bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        all_predictions = list(counts.keys())
        overall_pcts = [basic_stats.get('prediction_percentages', {}).get(pred, 0) for pred in all_predictions]
        relative_pcts = [basic_stats.get('relative_percentages_excluding_uncertain', {}).get(pred, 0) for pred in all_predictions]
        
        x = np.arange(len(all_predictions))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, overall_pcts, width, label='Overall %', alpha=0.8)
        bars2 = ax3.bar(x + width/2, relative_pcts, width, label='Relative % (decided only)', alpha=0.8)
        
        ax3.set_xlabel('Prediction Type')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Overall vs Relative Percentages', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_predictions, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Processing statistics
        ax4 = fig.add_subplot(gs[1, 0])
        if 'processing_metadata' in analysis:
            proc_meta = analysis['processing_metadata']
            success_count = proc_meta.get('successfully_processed', 0)
            error_count = proc_meta.get('processing_errors', 0)
            
            proc_labels = ['Successful', 'Errors']
            proc_counts = [success_count, error_count]
            proc_colors = ['green', 'red']
            
            bars = ax4.bar(proc_labels, proc_counts, color=proc_colors, alpha=0.7)
            ax4.set_title('Processing Success Rate', fontweight='bold')
            ax4.set_ylabel('Number of Images')
            
            # Add percentage labels
            total_proc = success_count + error_count
            if total_proc > 0:
                for bar, count in zip(bars, proc_counts):
                    pct = count / total_proc * 100
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proc_counts)*0.01,
                            f'{pct:.1f}%', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Processing metadata\nnot available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Processing Statistics', fontweight='bold')
        
        # Plot 5: Confidence levels
        ax5 = fig.add_subplot(gs[1, 1])
        if 'confidence_statistics' in analysis:
            conf_stats = analysis['confidence_statistics'].get('confidence_levels', {})
            if conf_stats:
                conf_labels = ['High (≥0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
                conf_counts = [
                    conf_stats.get('high_confidence', {}).get('count', 0),
                    conf_stats.get('medium_confidence', {}).get('count', 0),
                    conf_stats.get('low_confidence', {}).get('count', 0)
                ]
                conf_colors = ['green', 'orange', 'red']
                
                bars = ax5.bar(conf_labels, conf_counts, color=conf_colors, alpha=0.7)
                ax5.set_title('Confidence Level Distribution', fontweight='bold')
                ax5.set_ylabel('Number of Cells')
                ax5.tick_params(axis='x', rotation=45)
                
                # Add count labels
                for bar, count in zip(bars, conf_counts):
                    if count > 0:
                        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conf_counts)*0.01,
                                str(count), ha='center', va='bottom')
            else:
                ax5.text(0.5, 0.5, 'Confidence statistics\nnot available', 
                        transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title('Confidence Distribution', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Confidence statistics\nnot available', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Confidence Distribution', fontweight='bold')
        
        # Plot 6: Decision certainty (decided vs uncertain)
        ax6 = fig.add_subplot(gs[1, 2])
        decided_count = basic_stats.get('decided_predictions_count', 0)
        uncertain_count = counts.get('uncertain', 0)
        total_count = basic_stats.get('total_images_analyzed', 0)
        
        decision_labels = ['Decided', 'Uncertain']
        decision_counts = [decided_count, uncertain_count]
        decision_colors = ['lightblue', 'lightcoral']
        
        bars = ax6.bar(decision_labels, decision_counts, color=decision_colors, alpha=0.7)
        ax6.set_title('Decision Certainty', fontweight='bold')
        ax6.set_ylabel('Number of Cells')
        
        # Add percentage labels
        if total_count > 0:
            for bar, count in zip(bars, decision_counts):
                pct = count / total_count * 100
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(decision_counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # Create probability density plots based on classification mode
        if self.config.classification_mode == "binary":
            self._add_binary_probability_plots(fig, gs, analysis)
        else:
            self._add_ternary_probability_plots(fig, gs, analysis)
        
        plt.tight_layout()
        
        # Save the plot
        mode = analysis.get('analysis_metadata', {}).get('classification_mode', 'unknown')
        plot_path = output_dir / f"{mode}_enhanced_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        logger.info(f"Enhanced visualization saved to {plot_path}")
    
    def _add_binary_probability_plots(self, fig, gs, analysis: Dict) -> None:
        """Add binary classification probability density plots."""
        detailed_analysis = analysis.get('detailed_analysis', {})
        
        # Plot 7: Probability density for all predictions
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Extract probability data
        cancerous_stats = detailed_analysis.get('cancerous_cells', {}).get('stats', {})
        non_cancerous_stats = detailed_analysis.get('non_cancerous_cells', {}).get('stats', {})
        uncertain_stats = detailed_analysis.get('uncertain_cells', {}).get('stats', {})
        
        # Create synthetic data for visualization (since we don't have raw probabilities)
        # This is an approximation based on the statistics
        x = np.linspace(0, 1, 100)
        
        # Plot density curves if we have statistics
        if cancerous_stats.get('count', 0) > 0:
            mean_canc = cancerous_stats.get('mean', 0.8)
            std_canc = max(cancerous_stats.get('std', 0.1), 0.01)  # Prevent division by zero
            y_canc = self._gaussian_kde(x, mean_canc, std_canc)
            ax7.plot(x, y_canc, label=f'Cancerous (n={cancerous_stats.get("count", 0)})', 
                    color='red', linewidth=2)
        
        if non_cancerous_stats.get('count', 0) > 0:
            mean_non_canc = non_cancerous_stats.get('mean', 0.2)
            std_non_canc = max(non_cancerous_stats.get('std', 0.1), 0.01)
            y_non_canc = self._gaussian_kde(x, mean_non_canc, std_non_canc)
            ax7.plot(x, y_non_canc, label=f'Non-Cancerous (n={non_cancerous_stats.get("count", 0)})', 
                    color='blue', linewidth=2)
        
        if uncertain_stats.get('count', 0) > 0:
            mean_unc = uncertain_stats.get('mean', 0.5)
            std_unc = max(uncertain_stats.get('std', 0.1), 0.01)
            y_unc = self._gaussian_kde(x, mean_unc, std_unc)
            ax7.plot(x, y_unc, label=f'Uncertain (n={uncertain_stats.get("count", 0)})', 
                    color='gray', linewidth=2, linestyle='--')
        
        # Add threshold lines if thresholds are available
        try:
            ax7.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                       alpha=0.7, label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
            ax7.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                       alpha=0.7, label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
        except AttributeError:
            # Use default thresholds if not available
            ax7.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Threshold (0.3)')
            ax7.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Threshold (0.7)')
        
        ax7.axvline(x=0.5, color='black', linestyle=':', alpha=0.5, label='0.5 Threshold')
        
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Density')
        ax7.set_title('Probability Density Distribution (Binary Classification)', fontweight='bold')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Decided-only probability distribution
        ax8 = fig.add_subplot(gs[3, :2])
        
        decided_stats = detailed_analysis.get('decided_only_analysis', {})
        if decided_stats.get('count', 0) > 0:
            # Plot decided predictions only
            if cancerous_stats.get('count', 0) > 0:
                ax8.plot(x, y_canc, label=f'Cancerous (decided)', color='red', linewidth=2)
            if non_cancerous_stats.get('count', 0) > 0:
                ax8.plot(x, y_non_canc, label=f'Non-Cancerous (decided)', color='blue', linewidth=2)
            
            try:
                ax8.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                           alpha=0.7, label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
                ax8.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                           alpha=0.7, label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
            except AttributeError:
                ax8.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Threshold (0.3)')
                ax8.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Threshold (0.7)')
            
            rel_canc_pct = decided_stats.get('relative_cancerous_percentage', 0)
            rel_non_canc_pct = decided_stats.get('relative_non_cancerous_percentage', 0)
            
            ax8.set_title(f'Decided Predictions Only\n(Cancerous: {rel_canc_pct:.1f}%, Non-Cancerous: {rel_non_canc_pct:.1f}%)', 
                         fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'No decided predictions\navailable', ha='center', va='center', 
                    transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Decided Predictions Only', fontweight='bold')
        
        ax8.set_xlabel('Predicted Probability')
        ax8.set_ylabel('Density')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax9 = fig.add_subplot(gs[2:, 2])
        
        # Create text summary
        basic_stats = analysis.get('basic_statistics', {})
        insights = analysis.get('insights_and_recommendations', {}).get('summary', {})
        
        summary_text = f"""BINARY CLASSIFICATION SUMMARY

Total Images: {basic_stats.get('total_images_analyzed', 'N/A')}
Success Rate: {basic_stats.get('success_rate', 0):.1f}%

OVERALL DISTRIBUTION:
Cancerous: {basic_stats.get('prediction_counts', {}).get('cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('cancerous', 0):.1f}%)
Non-Cancerous: {basic_stats.get('prediction_counts', {}).get('non-cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('non-cancerous', 0):.1f}%)
Uncertain: {basic_stats.get('prediction_counts', {}).get('uncertain', 0)} ({basic_stats.get('prediction_percentages', {}).get('uncertain', 0):.1f}%)

DECIDED PREDICTIONS ONLY:
Total Decided: {basic_stats.get('decided_predictions_count', 0)} ({basic_stats.get('decided_predictions_percentage', 0):.1f}%)
Relative Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0):.1f}%
Relative Non-Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('non-cancerous', 0):.1f}%

CONFIDENCE ASSESSMENT:
{insights.get('confidence_assessment', 'N/A')}

PRIMARY FINDING:
{insights.get('primary_finding', 'N/A')}"""
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax9.set_title('Analysis Summary', fontweight='bold')
        ax9.axis('off')
    
    def _add_ternary_probability_plots(self, fig, gs, analysis: Dict) -> None:
        """Add ternary classification probability density plots."""
        detailed_analysis = analysis.get('detailed_analysis', {})
        
        # Plot 7: Confidence distribution
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Extract confidence data
        overall_conf_stats = detailed_analysis.get('overall_confidence_distribution', {})
        decided_conf_stats = detailed_analysis.get('decided_only_analysis', {}).get('confidence_distribution', {})
        
        # Create synthetic confidence distribution
        x = np.linspace(0, 1, 100)
        
        if overall_conf_stats.get('count', 0) > 0:
            mean_conf = overall_conf_stats.get('mean', 0.5)
            std_conf = max(overall_conf_stats.get('std', 0.2), 0.01)
            y_conf = self._gaussian_kde(x, mean_conf, std_conf)
            ax7.plot(x, y_conf, label=f'All Predictions (n={overall_conf_stats.get("count", 0)})', 
                    color='purple', linewidth=2)
        
        if decided_conf_stats.get('count', 0) > 0:
            mean_dec_conf = decided_conf_stats.get('mean', 0.6)
            std_dec_conf = max(decided_conf_stats.get('std', 0.15), 0.01)
            y_dec_conf = self._gaussian_kde(x, mean_dec_conf, std_dec_conf)
            ax7.plot(x, y_dec_conf, label=f'Decided Only (n={decided_conf_stats.get("count", 0)})', 
                    color='darkblue', linewidth=2, linestyle='--')
        
        # Add threshold line
        try:
            ax7.axvline(x=self.classifier.uncertainty_threshold, color='red', linestyle='--', 
                       alpha=0.7, label=f'Uncertainty Threshold ({self.classifier.uncertainty_threshold:.2f})')
        except AttributeError:
            ax7.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Uncertainty Threshold (0.5)')
        
        ax7.set_xlabel('Confidence Level')
        ax7.set_ylabel('Density')
        ax7.set_title('Confidence Distribution (Ternary Classification)', fontweight='bold')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Class distribution for decided predictions
        ax8 = fig.add_subplot(gs[3, :2])
        
        # Get class statistics
        canc_stats = detailed_analysis.get('cancerous_cells', {})
        non_canc_stats = detailed_analysis.get('non_cancerous_cells', {})
        fp_stats = detailed_analysis.get('false_positive_cells', {})
        
        classes = ['Cancerous', 'Non-Cancerous', 'False-Positive']
        decided_counts = [
            canc_stats.get('count', 0),
            non_canc_stats.get('count', 0),
            fp_stats.get('count', 0)
        ]
        relative_pcts = [
            canc_stats.get('relative_percentage', 0),
            non_canc_stats.get('relative_percentage', 0),
            fp_stats.get('relative_percentage', 0)
        ]
        
        x_pos = np.arange(len(classes))
        bars = ax8.bar(x_pos, decided_counts, color=['red', 'blue', 'orange'], alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, relative_pcts):
            height = bar.get_height()
            if height > 0:
                ax8.text(bar.get_x() + bar.get_width()/2., height + max(decided_counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        ax8.set_xlabel('Prediction Class')
        ax8.set_ylabel('Count')
        ax8.set_title('Decided Predictions by Class', fontweight='bold')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(classes)
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax9 = fig.add_subplot(gs[2:, 2])
        
        # Create text summary
        basic_stats = analysis.get('basic_statistics', {})
        insights = analysis.get('insights_and_recommendations', {}).get('summary', {})
        
        summary_text = f"""TERNARY CLASSIFICATION SUMMARY

Total Images: {basic_stats.get('total_images_analyzed', 'N/A')}
Success Rate: {basic_stats.get('success_rate', 0):.1f}%

OVERALL DISTRIBUTION:
Cancerous: {basic_stats.get('prediction_counts', {}).get('cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('cancerous', 0):.1f}%)
Non-Cancerous: {basic_stats.get('prediction_counts', {}).get('non-cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('non-cancerous', 0):.1f}%)
False-Positive: {basic_stats.get('prediction_counts', {}).get('false-positive', 0)} ({basic_stats.get('prediction_percentages', {}).get('false-positive', 0):.1f}%)
Uncertain: {basic_stats.get('prediction_counts', {}).get('uncertain', 0)} ({basic_stats.get('prediction_percentages', {}).get('uncertain', 0):.1f}%)

DECIDED PREDICTIONS ONLY:
Total Decided: {basic_stats.get('decided_predictions_count', 0)} ({basic_stats.get('decided_predictions_percentage', 0):.1f}%)
Relative Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0):.1f}%
Relative Non-Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('non-cancerous', 0):.1f}%
Relative False-Positive: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('false-positive', 0):.1f}%

CONFIDENCE ASSESSMENT:
{insights.get('confidence_assessment', 'N/A')}

PRIMARY FINDING:
{insights.get('primary_finding', 'N/A')}"""
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax9.set_title('Analysis Summary', fontweight='bold')
        ax9.axis('off')
    
    def _gaussian_kde(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Create a simple Gaussian kernel density estimate."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient analysis of unlabeled cell images")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save analysis results")
    parser.add_argument("--batch-size", type=int, default=2048,
                       help="Batch size for processing (reduce if getting OOM errors)")
    parser.add_argument("--save-individual-results", action="store_true",
                       help="Save detailed results for each individual image")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip creating visualization plots")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"],
                       help="Image file extensions to process")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear GPU cache before starting")
    
    args = parser.parse_args()
    
    # Clear GPU cache if requested
    if args.clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    # Load configuration and update with command line arguments
    config = ClassificationConfig.from_yaml(args.config)
    
    # Override config with command line arguments
    if hasattr(args, 'model_path'):
        config.model_path = args.model_path
    if hasattr(args, 'output_path'):
        config.output_path = args.output_path
    if hasattr(args, 'img_dir'):
        config.img_dir = args.image_dir
    
    logger.info(f"Loaded configuration: {config.classification_mode} mode")
    
    # Validate batch size
    if args.batch_size < 1:
        logger.error("Batch size must be at least 1")
        return
    
    if args.batch_size > 32:
        logger.warning(f"Large batch size ({args.batch_size}) may cause memory issues. Consider using smaller batches.")
    
    # Create memory-efficient classifier
    try:
        classifier = MemoryEfficientCellClassifier(
            model_path=config.model_path,
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to create classifier: {e}")
        return
    
    # Print model info and memory status
    try:
        model_info = classifier.get_model_info()
        logger.info(f"Model loaded: {model_info['model_name']} ({model_info['classification_mode']} mode)")
        logger.info(f"Device: {model_info['device']}")
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory before analysis: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        try:
            logger.info(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            logger.info("Could not get GPU total memory info")
    
    # Create analyzer and run analysis
    analyzer = MemoryEfficientCellAnalyzer(classifier)
    
    logger.info(f"Starting memory-efficient analysis of images in {config.img_dir}")
    logger.info(f"Using batch size: {args.batch_size}")
    
    try:
        results = analyzer.analyze_image_folder(
            image_dir=Path(config.img_dir),
            extensions=args.extensions,
            batch_size=args.batch_size,
            save_individual_results=args.save_individual_results
        )
        
        if not results or 'error' in results:
            logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        # Save results
        if config.output_dir:
            output_dir = Path(config.output_dir)
        else:
            output_dir = Path(config.img_dir).parent / "analysis_results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        main_results_path = output_dir / "enhanced_analysis_results.json"
        analyzer.save_results(
            results, 
            main_results_path,
            create_visualizations=not args.no_visualizations
        )
        
        # Print enhanced summary
        print("\n" + "="*90)
        print("ENHANCED MEMORY-EFFICIENT CELL ANALYSIS SUMMARY")
        print("="*90)
        
        # Processing summary
        if 'processing_metadata' in results:
            proc_meta = results['processing_metadata']
            print(f"Batch Size Used: {proc_meta.get('batch_size_used', 'N/A')}")
            print(f"Images Found: {proc_meta.get('total_images_found', 'N/A')}")
            print(f"Successfully Processed: {proc_meta.get('successfully_processed', 'N/A')}")
            print(f"Processing Errors: {proc_meta.get('processing_errors', 'N/A')}")
            print(f"Success Rate: {100 - proc_meta.get('error_rate', 0):.1f}%")
        
        basic_stats = results.get('basic_statistics', {})
        insights = results.get('insights_and_recommendations', {})
        
        print(f"\nClassification Mode: {config.classification_mode}")
        
        if basic_stats.get('prediction_counts'):
            print("\nOVERALL PREDICTION DISTRIBUTION:")
            print("-" * 50)
            for pred, count in basic_stats['prediction_counts'].items():
                percentage = basic_stats.get('prediction_percentages', {}).get(pred, 0)
                print(f"{pred.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Show relative distribution excluding uncertain
            print(f"\nRELATIVE DISTRIBUTION (EXCLUDING UNCERTAIN):")
            print("-" * 50)
            decided_count = basic_stats.get('decided_predictions_count', 0)
            decided_pct = basic_stats.get('decided_predictions_percentage', 0)
            print(f"Total Decided Predictions: {decided_count} ({decided_pct:.1f}% of all)")
            
            for pred, percentage in basic_stats.get('relative_percentages_excluding_uncertain', {}).items():
                count = basic_stats.get('decided_prediction_counts', {}).get(pred, 0)
                print(f"{pred.capitalize()}: {count} ({percentage:.1f}% of decided)")
        
        if insights.get('summary'):
            summary = insights['summary']
            print(f"\nPRIMARY FINDING:")
            print(f"{summary.get('primary_finding', 'N/A')}")
            
            print(f"\nCONFIDENCE ASSESSMENT:")
            print(f"{summary.get('confidence_assessment', 'N/A')}")
            
            print(f"\nUNCERTAINTY LEVEL: {summary.get('uncertainty_level', 0):.1f}%")
            print(f"DECIDED PREDICTIONS: {summary.get('decided_percentage', 0):.1f}%")
            
            # Show relative findings among decided predictions
            if 'relative_findings_among_decided' in summary:
                print(f"\nRELATIVE FINDINGS AMONG DECIDED PREDICTIONS:")
                print("-" * 50)
                rel_findings = summary['relative_findings_among_decided']
                for key, value in rel_findings.items():
                    print(f"{key.replace('_', ' ').title()}: {value:.1f}%")
        
        if insights.get('recommendations'):
            print(f"\nRECOMMENDATIONS:")
            print("-" * 50)
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nOUTPUT FILES:")
        print("-" * 50)
        print(f"Detailed results: {output_dir / 'enhanced_analysis_results.json'}")
        print(f"Summary report: {output_dir / 'analysis_summary_report.txt'}")
        
        if not args.no_visualizations:
            print(f"Enhanced visualizations: {output_dir / f'{config.classification_mode}_enhanced_analysis.png'}")
        
        # Final memory status
        if torch.cuda.is_available():
            print(f"\nFINAL MEMORY STATUS:")
            print("-" * 50)
            print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            try:
                print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")
            except Exception:
                print("Could not get peak memory info")
        
        # Key insights summary
        print(f"\nKEY INSIGHTS:")
        print("-" * 50)
        if basic_stats.get('decided_predictions_count', 0) > 0:
            decided_pct = basic_stats.get('decided_predictions_percentage', 0)
            print(f"• {decided_pct:.1f}% of images resulted in confident predictions")
            
            if config.classification_mode == "binary":
                rel_canc = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0)
                if rel_canc > 50:
                    print(f"• Among confident predictions, {rel_canc:.1f}% are classified as cancerous")
                    print("• This suggests a high-risk sample population requiring clinical attention")
                else:
                    print(f"• Among confident predictions, {100-rel_canc:.1f}% are classified as non-cancerous")
                    print("• This suggests a relatively low-risk sample population")
            else:
                rel_canc = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0)
                rel_fp = basic_stats.get('relative_percentages_excluding_uncertain', {}).get('false-positive', 0)
                print(f"• Among confident predictions:")
                print(f"  - {rel_canc:.1f}% are true cancerous cells")
                print(f"  - {rel_fp:.1f}% are false-positive detections")
                
                if rel_canc > 30:
                    print("• High prevalence of true cancerous cells detected")
                elif rel_fp > 20:
                    print("• Significant false-positive rate may indicate segmentation issues")
        
        uncertain_pct = basic_stats.get('prediction_percentages', {}).get('uncertain', 0)
        if uncertain_pct > 25:
            print(f"• High uncertainty ({uncertain_pct:.1f}%) suggests challenging samples or threshold optimization needed")
    
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        raise


if __name__ == "__main__":
    main()