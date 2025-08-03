"""
Comprehensive analysis and reporting for cell classification results.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import pandas as pd
import numpy as np

from classification.memory_efficient import MemoryEfficientCellClassifier

logger = logging.getLogger(__name__)


class CellAnalyzer:
    """
    Analyzer that processes images in batches and generates comprehensive reports.
    """
    
    def __init__(self, classifier: MemoryEfficientCellClassifier):
        self.classifier = classifier
        if classifier:
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
        
        # Perform efficient batch classification
        logger.info(f"Running memory-efficient inference with batch size {batch_size}...")
        results = self.classifier.classify_batch_efficient(image_paths, batch_size=batch_size)
        
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
    
    def save_results(self, analysis_results: Dict, output_path: Path) -> None:
        """Save analysis results to files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main results to JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")
        
        # Create summary report
        self._create_summary_report(analysis_results, output_path.parent)
        
        try:
            from classification.visualization import VisualizationGenerator
            viz_gen = VisualizationGenerator(self.config.classification_mode)
            viz_gen.create_enhanced_visualizations(analysis_results, output_path.parent)
        except ImportError:
            logger.warning("Visualization module not available")
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
    
    def _create_summary_report(self, analysis: Dict, output_dir: Path) -> None:
        """Create a human-readable summary report."""
        report_path = output_dir / "analysis_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CELL CLASSIFICATION ANALYSIS REPORT\n")
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


def convert_paths_to_strings(obj):
    """Convert Path objects to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj