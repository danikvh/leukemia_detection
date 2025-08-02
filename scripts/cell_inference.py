#!/usr/bin/env python3
"""
Corrected cell inference script for unlabeled image folders.
Performs inference on cell images and provides distribution analysis and counts.

Usage:
    python cell_inference_unlabeled.py --model-path models/binary_model.pth --config configs/binary_config.yaml --image-dir path/to/images/
    python cell_inference_unlabeled.py --model-path models/ternary_model.pth --config configs/ternary_config.yaml --image-dir path/to/images/ --save-individual-results
"""

import argparse
import logging
from pathlib import Path
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from classification.config import ClassificationConfig
from classification.inference import CellClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnlabeledCellAnalyzer:
    """
    Analyzes unlabeled cell images using a trained classifier.
    Provides distribution analysis, counts, and detailed results.
    """
    
    def __init__(self, classifier: CellClassifier):
        self.classifier = classifier
        self.config = classifier.config
    
    def analyze_image_folder(self, image_dir: Path, 
                           extensions: List[str] = None,
                           save_individual_results: bool = False) -> Dict[str, Any]:
        """
        Analyze all images in a folder and provide comprehensive results.
        
        Args:
            image_dir: Directory containing images to analyze
            extensions: List of file extensions to process
            save_individual_results: Whether to save detailed results for each image
            
        Returns:
            Dict containing analysis results
        """
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"]
        
        logger.info(f"Analyzing images in {image_dir}...")
        
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
        
        # Perform batch classification
        logger.info("Running inference on all images...")
        results = self.classifier.classify_batch(image_paths)
        
        # Combine results with image information
        detailed_results = []
        for image_path, result in zip(image_paths, results):
            detailed_result = {
                "filename": image_path.name,
                "filepath": str(image_path),
                "file_size_bytes": image_path.stat().st_size,
                **result
            }
            detailed_results.append(detailed_result)
        
        # Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis(detailed_results)
        
        # Add individual results if requested
        if save_individual_results:
            analysis['individual_results'] = detailed_results
        
        return analysis
    
    def _generate_comprehensive_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis from classification results."""
        logger.info("Generating comprehensive analysis...")
        
        # Extract predictions and probabilities
        predictions = [r['prediction'] for r in results]
        
        # Count predictions
        prediction_counts = Counter(predictions)
        total_images = len(results)
        
        # Calculate basic statistics
        basic_stats = {
            'total_images_analyzed': total_images,
            'prediction_counts': dict(prediction_counts),
            'prediction_percentages': {
                pred: count / total_images * 100 
                for pred, count in prediction_counts.items()
            }
        }
        
        # Generate mode-specific analysis
        if self.config.classification_mode == "binary":
            detailed_analysis = self._analyze_binary_results(results)
        else:
            detailed_analysis = self._analyze_ternary_results(results)
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_statistics(results)
        
        # Generate summary insights
        insights = self._generate_insights(prediction_counts, total_images, detailed_analysis)
        
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
        
        detailed_analysis = {
            'cancerous_cells': {
                'count': len(cancerous),
                'percentage': len(cancerous) / len(results) * 100,
                'avg_probability': np.mean([r['probability'] for r in cancerous]) if cancerous else 0,
                'min_probability': np.min([r['probability'] for r in cancerous]) if cancerous else 0,
                'max_probability': np.max([r['probability'] for r in cancerous]) if cancerous else 0,
                'high_confidence_count': len([r for r in cancerous if r['confidence'] >= 0.8])
            },
            'non_cancerous_cells': {
                'count': len(non_cancerous),
                'percentage': len(non_cancerous) / len(results) * 100,
                'avg_probability': np.mean([r['probability'] for r in non_cancerous]) if non_cancerous else 0,
                'min_probability': np.min([r['probability'] for r in non_cancerous]) if non_cancerous else 0,
                'max_probability': np.max([r['probability'] for r in non_cancerous]) if non_cancerous else 0,
                'high_confidence_count': len([r for r in non_cancerous if r['confidence'] >= 0.8])
            },
            'uncertain_cells': {
                'count': len(uncertain),
                'percentage': len(uncertain) / len(results) * 100,
                'avg_probability': np.mean([r['probability'] for r in uncertain]) if uncertain else 0,
                'closest_to_threshold': self._find_closest_to_threshold(uncertain) if uncertain else None
            },
            'probability_distribution': {
                'mean': np.mean(all_probs),
                'median': np.median(all_probs),
                'std': np.std(all_probs),
                'quartiles': {
                    'q25': np.percentile(all_probs, 25),
                    'q75': np.percentile(all_probs, 75)
                }
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
        
        # Calculate confidence statistics for each class
        def analyze_class(class_results, class_name):
            if not class_results:
                return {
                    'count': 0,
                    'percentage': 0,
                    'avg_confidence': 0,
                    'high_confidence_count': 0
                }
            
            return {
                'count': len(class_results),
                'percentage': len(class_results) / len(results) * 100,
                'avg_confidence': np.mean([r['confidence'] for r in class_results]),
                'min_confidence': np.min([r['confidence'] for r in class_results]),
                'max_confidence': np.max([r['confidence'] for r in class_results]),
                'high_confidence_count': len([r for r in class_results if r['confidence'] >= 0.7]),
                'avg_class_probability': np.mean([
                    r['class_probabilities'][class_name] for r in class_results
                ])
            }
        
        detailed_analysis = {
            'cancerous_cells': analyze_class(cancerous, 'cancerous'),
            'non_cancerous_cells': analyze_class(non_cancerous, 'non-cancerous'),
            'false_positive_cells': analyze_class(false_positive, 'false-positive'),
            'uncertain_cells': {
                'count': len(uncertain),
                'percentage': len(uncertain) / len(results) * 100,
                'avg_confidence': np.mean([r['confidence'] for r in uncertain]) if uncertain else 0,
                'low_confidence_analysis': self._analyze_low_confidence(uncertain) if uncertain else None
            },
            'confidence_distribution': {
                'all_confidences': [r['confidence'] for r in results],
                'mean_confidence': np.mean([r['confidence'] for r in results]),
                'median_confidence': np.median([r['confidence'] for r in results]),
                'std_confidence': np.std([r['confidence'] for r in results])
            }
        }
        
        return detailed_analysis
    
    def _calculate_confidence_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence-related statistics."""
        if self.config.classification_mode == "binary":
            confidences = [r['confidence'] for r in results]
        else:
            confidences = [r['confidence'] for r in results]
        
        # Define confidence levels
        high_confidence = [c for c in confidences if c >= 0.8]
        medium_confidence = [c for c in confidences if 0.5 <= c < 0.8]
        low_confidence = [c for c in confidences if c < 0.5]
        
        return {
            'overall_confidence': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'confidence_levels': {
                'high_confidence': {
                    'count': len(high_confidence),
                    'percentage': len(high_confidence) / len(confidences) * 100
                },
                'medium_confidence': {
                    'count': len(medium_confidence),
                    'percentage': len(medium_confidence) / len(confidences) * 100
                },
                'low_confidence': {
                    'count': len(low_confidence),
                    'percentage': len(low_confidence) / len(confidences) * 100
                }
            }
        }
    
    def _find_closest_to_threshold(self, uncertain_results: List[Dict]) -> Dict[str, Any]:
        """Find uncertain results closest to decision thresholds."""
        if not uncertain_results:
            return None
        
        # Find closest to 0.5 (most uncertain)
        distances_from_half = [abs(r['probability'] - 0.5) for r in uncertain_results]
        closest_idx = np.argmin(distances_from_half)
        
        return {
            'most_uncertain': {
                'filename': uncertain_results[closest_idx]['filename'],
                'probability': uncertain_results[closest_idx]['probability'],
                'distance_from_0.5': distances_from_half[closest_idx]
            }
        }
    
    def _analyze_low_confidence(self, uncertain_results: List[Dict]) -> Dict[str, Any]:
        """Analyze low confidence results for ternary classification."""
        if not uncertain_results:
            return None
        
        # Find the most ambiguous cases
        confidences = [r['confidence'] for r in uncertain_results]
        lowest_conf_idx = np.argmin(confidences)
        
        return {
            'most_ambiguous': {
                'filename': uncertain_results[lowest_conf_idx]['filename'],
                'confidence': uncertain_results[lowest_conf_idx]['confidence'],
                'class_probabilities': uncertain_results[lowest_conf_idx]['class_probabilities']
            },
            'confidence_range': {
                'min': np.min(confidences),
                'max': np.max(confidences),
                'mean': np.mean(confidences)
            }
        }
    
    def _generate_insights(self, prediction_counts: Counter, total_images: int, 
                         detailed_analysis: Dict) -> Dict[str, Any]:
        """Generate insights and recommendations based on results."""
        insights = {
            'summary': {},
            'quality_assessment': {},
            'recommendations': []
        }
        
        if self.config.classification_mode == "binary":
            insights['summary'] = {
                'primary_finding': self._get_primary_binary_finding(prediction_counts, total_images),
                'confidence_assessment': self._assess_binary_confidence(detailed_analysis),
                'uncertainty_level': prediction_counts.get('uncertain', 0) / total_images * 100
            }
            
            insights['recommendations'] = self._generate_binary_recommendations(
                prediction_counts, total_images, detailed_analysis
            )
        else:
            insights['summary'] = {
                'primary_finding': self._get_primary_ternary_finding(prediction_counts, total_images),
                'confidence_assessment': self._assess_ternary_confidence(detailed_analysis),
                'uncertainty_level': prediction_counts.get('uncertain', 0) / total_images * 100
            }
            
            insights['recommendations'] = self._generate_ternary_recommendations(
                prediction_counts, total_images, detailed_analysis
            )
        
        return insights
    
    def _get_primary_binary_finding(self, counts: Counter, total: int) -> str:
        """Generate primary finding summary for binary classification."""
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        non_cancerous_pct = counts.get('non-cancerous', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        if cancerous_pct > 50:
            return f"High prevalence of cancerous cells detected ({cancerous_pct:.1f}%)"
        elif cancerous_pct > 20:
            return f"Moderate prevalence of cancerous cells detected ({cancerous_pct:.1f}%)"
        elif uncertain_pct > 30:
            return f"High uncertainty in classification ({uncertain_pct:.1f}% uncertain)"
        else:
            return f"Predominantly non-cancerous cells ({non_cancerous_pct:.1f}%)"
    
    def _get_primary_ternary_finding(self, counts: Counter, total: int) -> str:
        """Generate primary finding summary for ternary classification."""
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        false_pos_pct = counts.get('false-positive', 0) / total * 100
        non_cancerous_pct = counts.get('non-cancerous', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        if cancerous_pct > 30:
            return f"High prevalence of cancerous cells detected ({cancerous_pct:.1f}%)"
        elif false_pos_pct > 20:
            return f"Significant false-positive detections ({false_pos_pct:.1f}%)"
        elif uncertain_pct > 25:
            return f"High classification uncertainty ({uncertain_pct:.1f}%)"
        else:
            return f"Mixed population with {non_cancerous_pct:.1f}% non-cancerous cells"
    
    def _assess_binary_confidence(self, analysis: Dict) -> str:
        """Assess overall confidence for binary classification."""
        high_conf_cancer = analysis['cancerous_cells']['high_confidence_count']
        high_conf_normal = analysis['non_cancerous_cells']['high_confidence_count']
        total_decided = (analysis['cancerous_cells']['count'] + 
                        analysis['non_cancerous_cells']['count'])
        
        if total_decided == 0:
            return "Unable to assess confidence - all predictions uncertain"
        
        high_conf_pct = (high_conf_cancer + high_conf_normal) / total_decided * 100
        
        if high_conf_pct > 80:
            return f"High confidence in classifications ({high_conf_pct:.1f}% high confidence)"
        elif high_conf_pct > 60:
            return f"Moderate confidence in classifications ({high_conf_pct:.1f}% high confidence)"
        else:
            return f"Low confidence in classifications ({high_conf_pct:.1f}% high confidence)"
    
    def _assess_ternary_confidence(self, analysis: Dict) -> str:
        """Assess overall confidence for ternary classification."""
        mean_confidence = analysis['confidence_distribution']['mean_confidence']
        
        if mean_confidence > 0.7:
            return f"High overall confidence (mean: {mean_confidence:.3f})"
        elif mean_confidence > 0.5:
            return f"Moderate overall confidence (mean: {mean_confidence:.3f})"
        else:
            return f"Low overall confidence (mean: {mean_confidence:.3f})"
    
    def _generate_binary_recommendations(self, counts: Counter, total: int, 
                                       analysis: Dict) -> List[str]:
        """Generate recommendations for binary classification results."""
        recommendations = []
        
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        
        if uncertain_pct > 30:
            recommendations.append(
                "High uncertainty detected. Consider manual review of uncertain cases "
                "or threshold optimization to reduce ambiguous classifications."
            )
        
        if cancerous_pct > 25:
            recommendations.append(
                "Significant number of cancerous cells detected. "
                "Prioritize these samples for immediate clinical review."
            )
        
        if analysis['cancerous_cells']['high_confidence_count'] > 0:
            recommendations.append(
                f"Focus on {analysis['cancerous_cells']['high_confidence_count']} "
                "high-confidence cancerous predictions for immediate attention."
            )
        
        avg_prob = analysis['probability_distribution']['mean']
        if avg_prob > 0.7:
            recommendations.append("Overall high probability scores suggest reliable predictions.")
        elif avg_prob < 0.3:
            recommendations.append("Overall low probability scores suggest reliable non-cancerous classifications.")
        
        return recommendations
    
    def _generate_ternary_recommendations(self, counts: Counter, total: int, 
                                        analysis: Dict) -> List[str]:
        """Generate recommendations for ternary classification results."""
        recommendations = []
        
        cancerous_pct = counts.get('cancerous', 0) / total * 100
        false_pos_pct = counts.get('false-positive', 0) / total * 100
        uncertain_pct = counts.get('uncertain', 0) / total * 100
        
        if cancerous_pct > 20:
            high_conf_cancerous = analysis['cancerous_cells']['high_confidence_count']
            recommendations.append(
                f"Prioritize {high_conf_cancerous} high-confidence cancerous cases for immediate clinical review."
            )
        
        if false_pos_pct > 15:
            recommendations.append(
                f"High false-positive rate ({false_pos_pct:.1f}%) detected. "
                "Consider reviewing segmentation quality or model calibration."
            )
        
        if uncertain_pct > 25:
            recommendations.append(
                "High uncertainty suggests need for manual review or threshold optimization."
            )
        
        mean_confidence = analysis['confidence_distribution']['mean_confidence']
        if mean_confidence < 0.5:
            recommendations.append(
                "Low overall confidence suggests challenging dataset. "
                "Consider additional quality control measures."
            )
        
        return recommendations
    
    def save_results(self, analysis_results: Dict, output_path: Path, 
                    create_visualizations: bool = True) -> None:
        """Save analysis results to files and optionally create visualizations."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main results to JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")
        
        # Create summary report
        self._create_summary_report(analysis_results, output_path.parent)
        
        if create_visualizations:
            self._create_visualizations(analysis_results, output_path.parent)
    
    def _create_summary_report(self, analysis: Dict, output_dir: Path) -> None:
        """Create a human-readable summary report."""
        report_path = output_dir / "analysis_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CELL CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            basic_stats = analysis['basic_statistics']
            f.write(f"Total Images Analyzed: {basic_stats['total_images_analyzed']}\n")
            f.write(f"Classification Mode: {analysis['analysis_metadata']['classification_mode']}\n")
            f.write(f"Analysis Date: {analysis['analysis_metadata']['analysis_timestamp']}\n\n")
            
            # Prediction counts
            f.write("PREDICTION DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for pred, count in basic_stats['prediction_counts'].items():
                percentage = basic_stats['prediction_percentages'][pred]
                f.write(f"{pred.capitalize()}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Primary findings
            insights = analysis['insights_and_recommendations']
            f.write("KEY FINDINGS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Primary Finding: {insights['summary']['primary_finding']}\n")
            f.write(f"Confidence Assessment: {insights['summary']['confidence_assessment']}\n")
            f.write(f"Uncertainty Level: {insights['summary']['uncertainty_level']:.1f}%\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(insights['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Detailed statistics
            detailed = analysis['detailed_analysis']
            if analysis['analysis_metadata']['classification_mode'] == 'binary':
                f.write("DETAILED BINARY ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Cancerous Cells: {detailed['cancerous_cells']['count']} "
                       f"(avg prob: {detailed['cancerous_cells']['avg_probability']:.3f})\n")
                f.write(f"Non-cancerous Cells: {detailed['non_cancerous_cells']['count']} "
                       f"(avg prob: {detailed['non_cancerous_cells']['avg_probability']:.3f})\n")
                f.write(f"Uncertain Cells: {detailed['uncertain_cells']['count']}\n")
            else:
                f.write("DETAILED TERNARY ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Cancerous: {detailed['cancerous_cells']['count']} "
                       f"(avg conf: {detailed['cancerous_cells']['avg_confidence']:.3f})\n")
                f.write(f"Non-cancerous: {detailed['non_cancerous_cells']['count']} "
                       f"(avg conf: {detailed['non_cancerous_cells']['avg_confidence']:.3f})\n")
                f.write(f"False-positive: {detailed['false_positive_cells']['count']} "
                       f"(avg conf: {detailed['false_positive_cells']['avg_confidence']:.3f})\n")
                f.write(f"Uncertain: {detailed['uncertain_cells']['count']}\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def _create_visualizations(self, analysis: Dict, output_dir: Path) -> None:
        """Create visualization plots for the analysis results."""
        logger.info("Creating visualization plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        basic_stats = analysis['basic_statistics']
        mode = analysis['analysis_metadata']['classification_mode']
        
        # Create figure with subplots
        if mode == 'binary':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        fig.suptitle(f'Cell Classification Analysis ({mode.title()} Mode)', fontsize=16)
        
        # Plot 1: Prediction distribution (pie chart)
        ax1 = axes[0, 0] if mode == 'binary' else axes[0, 0]
        counts = basic_stats['prediction_counts']
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Prediction Distribution')
        
        # Plot 2: Prediction counts (bar chart)
        ax2 = axes[0, 1] if mode == 'binary' else axes[0, 1]
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_title('Prediction Counts')
        ax2.set_ylabel('Number of Cells')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                    str(count), ha='center', va='bottom')
        
        # Mode-specific plots
        if mode == 'binary':
            self._create_binary_plots(analysis, axes, fig)
        else:
            self._create_ternary_plots(analysis, axes, fig)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{mode}_analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create confidence distribution plot
        self._plot_confidence_distribution(analysis, output_dir)
    
    def _create_binary_plots(self, analysis: Dict, axes, fig) -> None:
        """Create binary-specific plots."""
        detailed = analysis['detailed_analysis']
        
        # Plot 3: Confidence levels
        ax3 = axes[1, 0]
        conf_stats = analysis['confidence_statistics']['confidence_levels']
        conf_labels = list(conf_stats.keys())
        conf_counts = [conf_stats[label]['count'] for label in conf_labels]
        
        bars = ax3.bar([label.replace('_', ' ').title() for label in conf_labels], 
                      conf_counts, alpha=0.7)
        ax3.set_title('Confidence Level Distribution')
        ax3.set_ylabel('Number of Cells')
        
        # Plot 4: Probability distribution histogram
        ax4 = axes[1, 1]
        prob_dist = detailed['probability_distribution']
        
        # Create sample data for histogram (since we don't store individual probabilities)
        ax4.text(0.5, 0.5, f"Mean Probability: {prob_dist['mean']:.3f}\n"
                           f"Median: {prob_dist['median']:.3f}\n"
                           f"Std Dev: {prob_dist['std']:.3f}\n"
                           f"Q25: {prob_dist['quartiles']['q25']:.3f}\n"
                           f"Q75: {prob_dist['quartiles']['q75']:.3f}",
                transform=ax4.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Probability Statistics')
        ax4.axis('off')
    
    def _create_ternary_plots(self, analysis: Dict, axes, fig) -> None:
        """Create ternary-specific plots."""
        detailed = analysis['detailed_analysis']
        
        # Plot 3: Confidence by class
        ax3 = axes[0, 2]
        class_names = ['cancerous_cells', 'non_cancerous_cells', 'false_positive_cells']
        avg_confidences = [detailed[class_name]['avg_confidence'] for class_name in class_names]
        class_labels = [name.replace('_cells', '').replace('_', '-') for name in class_names]
        
        bars = ax3.bar(class_labels, avg_confidences, alpha=0.7)
        ax3.set_title('Average Confidence by Class')
        ax3.set_ylabel('Average Confidence')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: High confidence counts
        ax4 = axes[1, 0]
        high_conf_counts = [detailed[class_name]['high_confidence_count'] for class_name in class_names]
        bars = ax4.bar(class_labels, high_conf_counts, alpha=0.7, color='green')
        ax4.set_title('High Confidence Predictions (≥0.7)')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Confidence distribution stats
        ax5 = axes[1, 1]
        conf_dist = detailed['confidence_distribution']
        ax5.text(0.5, 0.5, f"Mean Confidence: {conf_dist['mean_confidence']:.3f}\n"
                           f"Median: {conf_dist['median_confidence']:.3f}\n"
                           f"Std Dev: {conf_dist['std_confidence']:.3f}",
                transform=ax5.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax5.set_title('Overall Confidence Statistics')
        ax5.axis('off')
        
        # Plot 6: Class probability comparison (if we had individual results)
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.5, "Detailed class probability\nanalysis available in\nindividual results",
                transform=ax6.transAxes, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax6.set_title('Class Probability Analysis')
        ax6.axis('off')
    
    def _plot_confidence_distribution(self, analysis: Dict, output_dir: Path) -> None:
        """Create a separate detailed confidence distribution plot."""
        conf_stats = analysis['confidence_statistics']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot confidence level distribution
        levels = ['Low (<0.5)', 'Medium (0.5-0.8)', 'High (≥0.8)']
        counts = [
            conf_stats['confidence_levels']['low_confidence']['count'],
            conf_stats['confidence_levels']['medium_confidence']['count'],
            conf_stats['confidence_levels']['high_confidence']['count']
        ]
        percentages = [
            conf_stats['confidence_levels']['low_confidence']['percentage'],
            conf_stats['confidence_levels']['medium_confidence']['percentage'],
            conf_stats['confidence_levels']['high_confidence']['percentage']
        ]
        
        colors = ['red', 'orange', 'green']
        bars = ax.bar(levels, counts, color=colors, alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Confidence Level Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Cells')
        ax.set_xlabel('Confidence Level')
        
        # Add statistics text
        overall = conf_stats['overall_confidence']
        stats_text = f"Mean: {overall['mean']:.3f}\nStd: {overall['std']:.3f}\nRange: [{overall['min']:.3f}, {overall['max']:.3f}]"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze unlabeled cell images using trained classifier")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="Directory containing images to analyze")
    parser.add_argument("--output-dir", type=str, default="cell_analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--save-individual-results", action="store_true",
                       help="Save detailed results for each individual image")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip creating visualization plots")
    parser.add_argument("--extensions", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"],
                       help="Image file extensions to process")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return
    
    if not Path(args.image_dir).exists():
        logger.error(f"Image directory not found: {args.image_dir}")
        return
    
    # Load configuration and create classifier
    config = ClassificationConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration: {config.classification_mode} mode")
    
    classifier = CellClassifier(
        model_path=args.model_path,
        config=config
    )
    
    # Print model info
    model_info = classifier.get_model_info()
    logger.info(f"Model loaded: {model_info['model_name']} ({model_info['classification_mode']} mode)")
    
    # Create analyzer and run analysis
    analyzer = UnlabeledCellAnalyzer(classifier)
    
    logger.info(f"Starting analysis of images in {args.image_dir}")
    results = analyzer.analyze_image_folder(
        image_dir=Path(args.image_dir),
        extensions=args.extensions,
        save_individual_results=args.save_individual_results
    )
    
    if not results:
        logger.error("No analysis results generated")
        return
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    main_results_path = output_dir / "cell_analysis_results.json"
    analyzer.save_results(
        results, 
        main_results_path,
        create_visualizations=not args.no_visualizations
    )
    
    # Print summary
    print("\n" + "="*80)
    print("CELL ANALYSIS SUMMARY")
    print("="*80)
    
    basic_stats = results['basic_statistics']
    insights = results['insights_and_recommendations']
    
    print(f"Images Analyzed: {basic_stats['total_images_analyzed']}")
    print(f"Classification Mode: {config.classification_mode}")
    
    print("\nPREDICTION COUNTS:")
    print("-" * 30)
    for pred, count in basic_stats['prediction_counts'].items():
        percentage = basic_stats['prediction_percentages'][pred]
        print(f"{pred.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\nPRIMARY FINDING:")
    print(f"{insights['summary']['primary_finding']}")
    
    print(f"\nCONFIDENCE ASSESSMENT:")
    print(f"{insights['summary']['confidence_assessment']}")
    
    print(f"\nUNCERTAINTY LEVEL: {insights['summary']['uncertainty_level']:.1f}%")
    
    if insights['recommendations']:
        print(f"\nRECOMMENDATIONS:")
        print("-" * 30)
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print(f"Summary report: {output_dir / 'analysis_summary_report.txt'}")
    
    if not args.no_visualizations:
        print(f"Visualizations: {output_dir / f'{config.classification_mode}_analysis_plots.png'}")


if __name__ == "__main__":
    main()