#!/usr/bin/env python3
"""
Threshold optimization module for cell classification.
Finds optimal confidence thresholds to minimize uncertain classifications while maximizing accuracy.

Usage:
    python threshold_optimizer.py --model-path models/binary_model.pth --config configs/binary_config.yaml --test-data-dir path/to/test/data --test-labels-csv path/to/test/labels.csv
"""

import argparse
import logging
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from classification.config import ClassificationConfig
from classification.inference import CellClassifier
from classification.utils import load_classification_labels, get_image_paths_and_labels
from classification.datasets import CellClassificationDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    Optimizes confidence thresholds for cell classification to minimize uncertainty
    while maximizing classification performance.
    """
    
    def __init__(self, classifier: CellClassifier):
        self.classifier = classifier
        self.config = classifier.config
        
    def optimize_thresholds(self, test_loader: DataLoader, test_data: List[Tuple[Path, int]], 
                          output_dir: Path) -> Dict[str, Any]:
        """
        Find optimal thresholds using grid search on test set.
        
        Args:
            test_loader: DataLoader for test set
            test_data: List of (image_path, label) tuples
            output_dir: Directory to save results
            
        Returns:
            Dict containing optimization results and recommended thresholds
        """
        logger.info("Starting threshold optimization...")
        
        # Get predictions and true labels from test set
        predictions_data = self._get_test_predictions(test_loader)
        
        if self.config.classification_mode == "binary":
            return self._optimize_binary_thresholds(predictions_data, output_dir)
        else:
            return self._optimize_ternary_thresholds(predictions_data, output_dir)
    
    def _get_test_predictions(self, test_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Get raw predictions from test set."""
        logger.info("Getting predictions from test set...")
        
        self.classifier.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(test_loader, desc="Getting predictions"):
                inputs = inputs.to(self.classifier.device)
                outputs = self.classifier.model(inputs)
                
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        if self.config.classification_mode == "binary":
            # For binary: convert to probabilities
            import torch
            probabilities = torch.sigmoid(torch.tensor(all_outputs)).numpy().flatten()
            return {
                'probabilities': probabilities,
                'labels': all_labels.flatten()
            }
        else:
            # For ternary: convert to probabilities
            import torch
            probabilities = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
            return {
                'probabilities': probabilities,
                'labels': all_labels
            }
    
    def _optimize_binary_thresholds(self, predictions_data: Dict, output_dir: Path) -> Dict[str, Any]:
        """Optimize thresholds for binary classification."""
        probabilities = predictions_data['probabilities']
        true_labels = predictions_data['labels']
        
        # Define threshold ranges
        threshold_range = np.linspace(0.1, 0.9, 17)  # 0.1 to 0.9 in steps of 0.05
        
        results = []
        
        logger.info("Testing threshold combinations...")
        for low_thresh in tqdm(threshold_range, desc="Low threshold"):
            for high_thresh in threshold_range:
                if high_thresh <= low_thresh:
                    continue
                
                # Calculate metrics for this threshold combination
                metrics = self._calculate_binary_threshold_metrics(
                    probabilities, true_labels, low_thresh, high_thresh
                )
                
                results.append({
                    'low_threshold': low_thresh,
                    'high_threshold': high_thresh,
                    **metrics
                })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal thresholds based on different criteria
        optimal_thresholds = self._find_optimal_binary_thresholds(results_df)
        
        # Save results
        self._save_binary_optimization_results(results_df, optimal_thresholds, output_dir)
        
        # Create visualizations
        self._plot_binary_threshold_analysis(results_df, optimal_thresholds, output_dir)
        
        return {
            'all_results': results,
            'optimal_thresholds': optimal_thresholds,
            'recommendations': self._generate_binary_recommendations(optimal_thresholds)
        }
    
    def _calculate_binary_threshold_metrics(self, probabilities: np.ndarray, true_labels: np.ndarray,
                                          low_thresh: float, high_thresh: float) -> Dict[str, float]:
        """Calculate metrics for given binary thresholds."""
        # Classify based on thresholds
        predictions = []
        uncertain_count = 0
        
        for prob in probabilities:
            if prob >= high_thresh:
                predictions.append(1)  # Cancerous
            elif prob <= low_thresh:
                predictions.append(0)  # Non-cancerous
            else:
                predictions.append(-1)  # Uncertain
                uncertain_count += 1
        
        predictions = np.array(predictions)
        
        # Calculate metrics only for decided predictions
        decided_mask = predictions >= 0
        decided_preds = predictions[decided_mask]
        decided_labels = true_labels[decided_mask]
        
        if len(decided_preds) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'uncertain_count': uncertain_count,
                'uncertain_percentage': uncertain_count / len(probabilities),
                'decided_count': 0,
                'coverage': 0.0
            }
        
        # Standard metrics
        accuracy = accuracy_score(decided_labels, decided_preds)
        precision = precision_score(decided_labels, decided_preds, zero_division=0)
        recall = recall_score(decided_labels, decided_preds, zero_division=0)
        f1 = f1_score(decided_labels, decided_preds, zero_division=0)
        
        # Confusion matrix components
        tp = np.sum((decided_preds == 1) & (decided_labels == 1))
        fp = np.sum((decided_preds == 1) & (decided_labels == 0))
        tn = np.sum((decided_preds == 0) & (decided_labels == 0))
        fn = np.sum((decided_preds == 0) & (decided_labels == 1))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'uncertain_count': uncertain_count,
            'uncertain_percentage': uncertain_count / len(probabilities),
            'decided_count': len(decided_preds),
            'coverage': len(decided_preds) / len(probabilities)
        }
    
    def _find_optimal_binary_thresholds(self, results_df: pd.DataFrame) -> Dict[str, Dict]:
        """Find optimal thresholds based on different criteria."""
        optimal_thresholds = {}
        
        # 1. Maximize F1 score
        max_f1_idx = results_df['f1_score'].idxmax()
        optimal_thresholds['max_f1'] = {
            'low_threshold': results_df.loc[max_f1_idx, 'low_threshold'],
            'high_threshold': results_df.loc[max_f1_idx, 'high_threshold'],
            'f1_score': results_df.loc[max_f1_idx, 'f1_score'],
            'accuracy': results_df.loc[max_f1_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_f1_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_f1_idx, 'coverage'],
            'criterion': 'Maximize F1 Score'
        }
        
        # 2. Maximize accuracy
        max_acc_idx = results_df['accuracy'].idxmax()
        optimal_thresholds['max_accuracy'] = {
            'low_threshold': results_df.loc[max_acc_idx, 'low_threshold'],
            'high_threshold': results_df.loc[max_acc_idx, 'high_threshold'],
            'f1_score': results_df.loc[max_acc_idx, 'f1_score'],
            'accuracy': results_df.loc[max_acc_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_acc_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_acc_idx, 'coverage'],
            'criterion': 'Maximize Accuracy'
        }
        
        # 3. Minimize uncertainty while maintaining good performance
        # Filter for decent performance (F1 > 0.7) then minimize uncertainty
        decent_performance = results_df[results_df['f1_score'] >= 0.7]
        if not decent_performance.empty:
            min_uncertain_idx = decent_performance['uncertain_percentage'].idxmin()
            optimal_thresholds['min_uncertainty'] = {
                'low_threshold': decent_performance.loc[min_uncertain_idx, 'low_threshold'],
                'high_threshold': decent_performance.loc[min_uncertain_idx, 'high_threshold'],
                'f1_score': decent_performance.loc[min_uncertain_idx, 'f1_score'],
                'accuracy': decent_performance.loc[min_uncertain_idx, 'accuracy'],
                'uncertain_percentage': decent_performance.loc[min_uncertain_idx, 'uncertain_percentage'],
                'coverage': decent_performance.loc[min_uncertain_idx, 'coverage'],
                'criterion': 'Minimize Uncertainty (F1 ≥ 0.7)'
            }
        
        # 4. Balance performance and coverage
        # Create a composite score: F1 * Coverage - Uncertainty_penalty
        results_df['composite_score'] = (
            results_df['f1_score'] * results_df['coverage'] - 
            0.5 * results_df['uncertain_percentage']
        )
        max_composite_idx = results_df['composite_score'].idxmax()
        optimal_thresholds['balanced'] = {
            'low_threshold': results_df.loc[max_composite_idx, 'low_threshold'],
            'high_threshold': results_df.loc[max_composite_idx, 'high_threshold'],
            'f1_score': results_df.loc[max_composite_idx, 'f1_score'],
            'accuracy': results_df.loc[max_composite_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_composite_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_composite_idx, 'coverage'],
            'composite_score': results_df.loc[max_composite_idx, 'composite_score'],
            'criterion': 'Balance Performance and Coverage'
        }
        
        # 5. High precision (minimize false positives)
        high_precision = results_df[results_df['precision'] >= 0.8]
        if not high_precision.empty:
            max_precision_f1_idx = high_precision['f1_score'].idxmax()
            optimal_thresholds['high_precision'] = {
                'low_threshold': high_precision.loc[max_precision_f1_idx, 'low_threshold'],
                'high_threshold': high_precision.loc[max_precision_f1_idx, 'high_threshold'],
                'f1_score': high_precision.loc[max_precision_f1_idx, 'f1_score'],
                'accuracy': high_precision.loc[max_precision_f1_idx, 'accuracy'],
                'precision': high_precision.loc[max_precision_f1_idx, 'precision'],
                'uncertain_percentage': high_precision.loc[max_precision_f1_idx, 'uncertain_percentage'],
                'coverage': high_precision.loc[max_precision_f1_idx, 'coverage'],
                'criterion': 'High Precision (≥ 0.8)'
            }
        
        # 6. High recall (minimize false negatives)
        high_recall = results_df[results_df['recall'] >= 0.8]
        if not high_recall.empty:
            max_recall_f1_idx = high_recall['f1_score'].idxmax()
            optimal_thresholds['high_recall'] = {
                'low_threshold': high_recall.loc[max_recall_f1_idx, 'low_threshold'],
                'high_threshold': high_recall.loc[max_recall_f1_idx, 'high_threshold'],
                'f1_score': high_recall.loc[max_recall_f1_idx, 'f1_score'],
                'accuracy': high_recall.loc[max_recall_f1_idx, 'accuracy'],
                'recall': high_recall.loc[max_recall_f1_idx, 'recall'],
                'uncertain_percentage': high_recall.loc[max_recall_f1_idx, 'uncertain_percentage'],
                'coverage': high_recall.loc[max_recall_f1_idx, 'coverage'],
                'criterion': 'High Recall (≥ 0.8)'
            }
        
        return optimal_thresholds
    
    def _save_binary_optimization_results(self, results_df: pd.DataFrame, 
                                        optimal_thresholds: Dict, output_dir: Path):
        """Save binary optimization results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        results_df.to_csv(output_dir / "threshold_optimization_results.csv", index=False)
        
        # Save optimal thresholds
        with open(output_dir / "optimal_thresholds.json", 'w') as f:
            json.dump(optimal_thresholds, f, indent=4, default=str)
        
        logger.info(f"Optimization results saved to {output_dir}")
    
    def _plot_binary_threshold_analysis(self, results_df: pd.DataFrame, 
                                      optimal_thresholds: Dict, output_dir: Path):
        """Create visualization plots for binary threshold analysis."""
        # Create heatmaps for different metrics
        metrics_to_plot = ['f1_score', 'accuracy', 'uncertain_percentage', 'coverage']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            # Create pivot table for heatmap
            pivot_data = results_df.pivot_table(
                values=metric, 
                index='high_threshold', 
                columns='low_threshold', 
                aggfunc='mean'
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot_data, 
                annot=False, 
                cmap='viridis' if metric != 'uncertain_percentage' else 'viridis_r',
                ax=axes[i],
                cbar_kws={'label': metric.replace('_', ' ').title()}
            )
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} vs Thresholds')
            axes[i].set_xlabel('Low Threshold')
            axes[i].set_ylabel('High Threshold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot optimal thresholds comparison
        self._plot_optimal_thresholds_comparison(optimal_thresholds, output_dir)
    
    def _plot_optimal_thresholds_comparison(self, optimal_thresholds: Dict, output_dir: Path):
        """Plot comparison of optimal thresholds."""
        if not optimal_thresholds:
            return
        
        # Prepare data for plotting
        threshold_names = list(optimal_thresholds.keys())
        metrics = ['f1_score', 'accuracy', 'uncertain_percentage', 'coverage']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [optimal_thresholds[name].get(metric, 0) for name in threshold_names]
            
            bars = axes[i].bar(threshold_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Optimization Strategy')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "optimal_thresholds_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_binary_recommendations(self, optimal_thresholds: Dict) -> Dict[str, str]:
        """Generate recommendations based on optimization results."""
        recommendations = {}
        
        if 'balanced' in optimal_thresholds:
            recommendations['general'] = (
                f"For general use, recommend balanced thresholds: "
                f"low={optimal_thresholds['balanced']['low_threshold']:.2f}, "
                f"high={optimal_thresholds['balanced']['high_threshold']:.2f}. "
                f"This provides F1={optimal_thresholds['balanced']['f1_score']:.3f} "
                f"with {optimal_thresholds['balanced']['uncertain_percentage']:.1%} uncertainty."
            )
        
        if 'high_precision' in optimal_thresholds:
            recommendations['clinical'] = (
                f"For clinical applications requiring high precision: "
                f"low={optimal_thresholds['high_precision']['low_threshold']:.2f}, "
                f"high={optimal_thresholds['high_precision']['high_threshold']:.2f}. "
                f"This minimizes false positives with precision≥0.8."
            )
        
        if 'min_uncertainty' in optimal_thresholds:
            recommendations['automated'] = (
                f"For automated screening with minimal manual review: "
                f"low={optimal_thresholds['min_uncertainty']['low_threshold']:.2f}, "
                f"high={optimal_thresholds['min_uncertainty']['high_threshold']:.2f}. "
                f"This minimizes uncertain cases to {optimal_thresholds['min_uncertainty']['uncertain_percentage']:.1%}."
            )
        
        return recommendations
    
    def _optimize_ternary_thresholds(self, predictions_data: Dict, output_dir: Path) -> Dict[str, Any]:
        """Optimize thresholds for ternary classification."""
        probabilities = predictions_data['probabilities']  # Shape: (n_samples, 3)
        true_labels = predictions_data['labels']
        
        # Define threshold ranges
        confidence_range = np.linspace(0.1, 0.9, 17)
        uncertainty_range = np.linspace(0.05, 0.3, 11)
        
        results = []
        
        logger.info("Testing threshold combinations for ternary classification...")
        for conf_high in tqdm(confidence_range, desc="Confidence high"):
            for conf_low in confidence_range:
                if conf_low >= conf_high:
                    continue
                for uncert_thresh in uncertainty_range:
                    metrics = self._calculate_ternary_threshold_metrics(
                        probabilities, true_labels, conf_low, conf_high, uncert_thresh
                    )
                    
                    results.append({
                        'confidence_threshold_low': conf_low,
                        'confidence_threshold_high': conf_high,
                        'uncertainty_threshold': uncert_thresh,
                        **metrics
                    })
        
        # Convert to DataFrame and find optimal thresholds
        results_df = pd.DataFrame(results)
        optimal_thresholds = self._find_optimal_ternary_thresholds(results_df)
        
        # Save results and create visualizations
        self._save_ternary_optimization_results(results_df, optimal_thresholds, output_dir)
        self._plot_ternary_threshold_analysis(results_df, optimal_thresholds, output_dir)
        
        return {
            'all_results': results,
            'optimal_thresholds': optimal_thresholds,
            'recommendations': self._generate_ternary_recommendations(optimal_thresholds)
        }
    
    def _calculate_ternary_threshold_metrics(self, probabilities: np.ndarray, true_labels: np.ndarray,
                                           conf_low: float, conf_high: float, uncert_thresh: float) -> Dict[str, float]:
        """Calculate metrics for given ternary thresholds."""
        predictions = []
        uncertain_count = 0
        
        for prob_array in probabilities:
            max_prob_idx = np.argmax(prob_array)
            max_prob = prob_array[max_prob_idx]
            
            # Calculate confidence as difference between highest and second highest
            sorted_probs = np.sort(prob_array)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]
            
            # Determine prediction based on thresholds
            if confidence < uncert_thresh:
                predictions.append(-1)  # Uncertain
                uncertain_count += 1
            elif max_prob >= conf_high:
                predictions.append(max_prob_idx)
            elif max_prob <= conf_low:
                predictions.append(-1)  # Uncertain
                uncertain_count += 1
            else:
                predictions.append(max_prob_idx)
        
        predictions = np.array(predictions)
        
        # Calculate metrics only for decided predictions
        decided_mask = predictions >= 0
        decided_preds = predictions[decided_mask]
        decided_labels = true_labels[decided_mask]
        
        if len(decided_preds) == 0:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'uncertain_count': uncertain_count,
                'uncertain_percentage': uncertain_count / len(probabilities),
                'decided_count': 0,
                'coverage': 0.0
            }
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(decided_labels, decided_preds)
        f1_macro = f1_score(decided_labels, decided_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(decided_labels, decided_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(decided_labels, decided_preds, average='macro', zero_division=0)
        recall_macro = recall_score(decided_labels, decided_preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'uncertain_count': uncertain_count,
            'uncertain_percentage': uncertain_count / len(probabilities),
            'decided_count': len(decided_preds),
            'coverage': len(decided_preds) / len(probabilities)
        }
    
    def _find_optimal_ternary_thresholds(self, results_df: pd.DataFrame) -> Dict[str, Dict]:
        """Find optimal thresholds for ternary classification."""
        optimal_thresholds = {}
        
        # 1. Maximize F1 macro score
        max_f1_idx = results_df['f1_macro'].idxmax()
        optimal_thresholds['max_f1_macro'] = {
            'confidence_threshold_low': results_df.loc[max_f1_idx, 'confidence_threshold_low'],
            'confidence_threshold_high': results_df.loc[max_f1_idx, 'confidence_threshold_high'],
            'uncertainty_threshold': results_df.loc[max_f1_idx, 'uncertainty_threshold'],
            'f1_macro': results_df.loc[max_f1_idx, 'f1_macro'],
            'accuracy': results_df.loc[max_f1_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_f1_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_f1_idx, 'coverage'],
            'criterion': 'Maximize F1 Macro Score'
        }
        
        # 2. Maximize accuracy
        max_acc_idx = results_df['accuracy'].idxmax()
        optimal_thresholds['max_accuracy'] = {
            'confidence_threshold_low': results_df.loc[max_acc_idx, 'confidence_threshold_low'],
            'confidence_threshold_high': results_df.loc[max_acc_idx, 'confidence_threshold_high'],
            'uncertainty_threshold': results_df.loc[max_acc_idx, 'uncertainty_threshold'],
            'f1_macro': results_df.loc[max_acc_idx, 'f1_macro'],
            'accuracy': results_df.loc[max_acc_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_acc_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_acc_idx, 'coverage'],
            'criterion': 'Maximize Accuracy'
        }
        
        # 3. Minimize uncertainty
        decent_performance = results_df[results_df['f1_macro'] >= 0.6]
        if not decent_performance.empty:
            min_uncertain_idx = decent_performance['uncertain_percentage'].idxmin()
            optimal_thresholds['min_uncertainty'] = {
                'confidence_threshold_low': decent_performance.loc[min_uncertain_idx, 'confidence_threshold_low'],
                'confidence_threshold_high': decent_performance.loc[min_uncertain_idx, 'confidence_threshold_high'],
                'uncertainty_threshold': decent_performance.loc[min_uncertain_idx, 'uncertainty_threshold'],
                'f1_macro': decent_performance.loc[min_uncertain_idx, 'f1_macro'],
                'accuracy': decent_performance.loc[min_uncertain_idx, 'accuracy'],
                'uncertain_percentage': decent_performance.loc[min_uncertain_idx, 'uncertain_percentage'],
                'coverage': decent_performance.loc[min_uncertain_idx, 'coverage'],
                'criterion': 'Minimize Uncertainty (F1 ≥ 0.6)'
            }
        
        # 4. Balanced approach
        results_df['composite_score'] = (
            results_df['f1_macro'] * results_df['coverage'] - 
            0.3 * results_df['uncertain_percentage']
        )
        max_composite_idx = results_df['composite_score'].idxmax()
        optimal_thresholds['balanced'] = {
            'confidence_threshold_low': results_df.loc[max_composite_idx, 'confidence_threshold_low'],
            'confidence_threshold_high': results_df.loc[max_composite_idx, 'confidence_threshold_high'],
            'uncertainty_threshold': results_df.loc[max_composite_idx, 'uncertainty_threshold'],
            'f1_macro': results_df.loc[max_composite_idx, 'f1_macro'],
            'accuracy': results_df.loc[max_composite_idx, 'accuracy'],
            'uncertain_percentage': results_df.loc[max_composite_idx, 'uncertain_percentage'],
            'coverage': results_df.loc[max_composite_idx, 'coverage'],
            'composite_score': results_df.loc[max_composite_idx, 'composite_score'],
            'criterion': 'Balance Performance and Coverage'
        }
        
        return optimal_thresholds
    
    def _save_ternary_optimization_results(self, results_df: pd.DataFrame, 
                                         optimal_thresholds: Dict, output_dir: Path):
        """Save ternary optimization results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / "ternary_threshold_optimization_results.csv", index=False)
        
        with open(output_dir / "ternary_optimal_thresholds.json", 'w') as f:
            json.dump(optimal_thresholds, f, indent=4, default=str)
        
        logger.info(f"Ternary optimization results saved to {output_dir}")
    
    def _plot_ternary_threshold_analysis(self, results_df: pd.DataFrame, 
                                       optimal_thresholds: Dict, output_dir: Path):
        """Create visualization plots for ternary threshold analysis."""
        # Since we have 3 threshold parameters, we'll create several 2D projections
        
        # 1. F1 vs uncertainty percentage for different parameter combinations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: F1 macro vs uncertainty percentage
        scatter = axes[0].scatter(results_df['uncertain_percentage'], results_df['f1_macro'], 
                                c=results_df['accuracy'], cmap='viridis', alpha=0.6)
        axes[0].set_xlabel('Uncertainty Percentage')
        axes[0].set_ylabel('F1 Macro Score')
        axes[0].set_title('F1 Macro vs Uncertainty (colored by Accuracy)')
        fig.colorbar(scatter, ax=axes[0], label='Accuracy')
        
        # Plot 2: Coverage vs F1 macro
        scatter2 = axes[1].scatter(results_df['coverage'], results_df['f1_macro'], 
                                 c=results_df['uncertain_percentage'], cmap='plasma', alpha=0.6)
        axes[1].set_xlabel('Coverage')
        axes[1].set_ylabel('F1 Macro Score')
        axes[1].set_title('Coverage vs F1 Macro (colored by Uncertainty %)')
        fig.colorbar(scatter2, ax=axes[1], label='Uncertainty %')
        
        # Plot 3: Confidence thresholds impact
        for uncert_val in [0.1, 0.15, 0.2]:
            subset = results_df[np.abs(results_df['uncertainty_threshold'] - uncert_val) < 0.01]
            if not subset.empty:
                axes[2].scatter(subset['confidence_threshold_high'] - subset['confidence_threshold_low'], 
                              subset['f1_macro'], alpha=0.6, 
                              label=f'Uncertainty thresh = {uncert_val}')
        axes[2].set_xlabel('Confidence Threshold Range (High - Low)')
        axes[2].set_ylabel('F1 Macro Score')
        axes[2].set_title('Threshold Range Impact on F1')
        axes[2].legend()
        
        # Plot 4: Optimal thresholds comparison
        if optimal_thresholds:
            threshold_names = list(optimal_thresholds.keys())
            f1_values = [optimal_thresholds[name].get('f1_macro', 0) for name in threshold_names]
            uncert_values = [optimal_thresholds[name].get('uncertain_percentage', 0) for name in threshold_names]
            
            bars = axes[3].bar(threshold_names, f1_values, alpha=0.7)
            axes[3].set_title('F1 Macro by Optimization Strategy')
            axes[3].set_ylabel('F1 Macro Score')
            axes[3].tick_params(axis='x', rotation=45)
            
            # Add uncertainty percentage as text on bars
            for bar, uncert in zip(bars, uncert_values):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{uncert:.1%}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ternary_threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_ternary_recommendations(self, optimal_thresholds: Dict) -> Dict[str, str]:
        """Generate recommendations for ternary classification."""
        recommendations = {}
        
        if 'balanced' in optimal_thresholds:
            ot = optimal_thresholds['balanced']
            recommendations['general'] = (
                f"For general use, recommend balanced thresholds: "
                f"confidence_low={ot['confidence_threshold_low']:.2f}, "
                f"confidence_high={ot['confidence_threshold_high']:.2f}, "
                f"uncertainty={ot['uncertainty_threshold']:.2f}. "
                f"This provides F1={ot['f1_macro']:.3f} "
                f"with {ot['uncertain_percentage']:.1%} uncertainty."
            )
        
        if 'max_accuracy' in optimal_thresholds:
            ot = optimal_thresholds['max_accuracy']
            recommendations['accuracy_focused'] = (
                f"For maximum accuracy: "
                f"confidence_low={ot['confidence_threshold_low']:.2f}, "
                f"confidence_high={ot['confidence_threshold_high']:.2f}, "
                f"uncertainty={ot['uncertainty_threshold']:.2f}. "
                f"Achieves {ot['accuracy']:.1%} accuracy."
            )
        
        if 'min_uncertainty' in optimal_thresholds:
            ot = optimal_thresholds['min_uncertainty']
            recommendations['automated'] = (
                f"For minimal manual review: "
                f"confidence_low={ot['confidence_threshold_low']:.2f}, "
                f"confidence_high={ot['confidence_threshold_high']:.2f}, "
                f"uncertainty={ot['uncertainty_threshold']:.2f}. "
                f"Reduces uncertainty to {ot['uncertain_percentage']:.1%}."
            )
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Optimize confidence thresholds for cell classification")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--test-data-dir", type=str, required=True,
                       help="Directory containing test images")
    parser.add_argument("--test-labels-csv", type=str, required=True,
                       help="CSV file with test labels")
    parser.add_argument("--output-dir", type=str, default="threshold_optimization_results",
                       help="Directory to save optimization results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return
    
    if not Path(args.test_data_dir).exists():
        logger.error(f"Test data directory not found: {args.test_data_dir}")
        return
    
    if not Path(args.test_labels_csv).exists():
        logger.error(f"Test labels CSV not found: {args.test_labels_csv}")
        return
    
    # Load configuration and create classifier
    config = ClassificationConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration: {config.classification_mode} mode")
    
    classifier = CellClassifier(
        model_path=args.model_path,
        config=config
    )
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data_dir}")
    test_label_map = load_classification_labels(Path(args.test_labels_csv), config.classification_mode)
    test_data = get_image_paths_and_labels(Path(args.test_data_dir), test_label_map)
    
    if not test_data:
        logger.error("No test data found!")
        return
    
    logger.info(f"Found {len(test_data)} test samples")
    
    # Create test dataset and dataloader
    test_dataset = CellClassificationDataset(
        data_samples=test_data,
        image_size=config.image_size,
        mean=config.normalize_mean,
        std=config.normalize_std,
        classification_mode=config.classification_mode,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create optimizer and run optimization
    optimizer = ThresholdOptimizer(classifier)
    output_dir = Path(args.output_dir)
    
    results = optimizer.optimize_thresholds(test_loader, test_data, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("="*80)
    
    optimal_thresholds = results['optimal_thresholds']
    recommendations = results['recommendations']
    
    print(f"Classification Mode: {config.classification_mode}")
    print(f"Test Samples: {len(test_data)}")
    print(f"Results saved to: {output_dir}")
    
    print("\nOPTIMAL THRESHOLDS:")
    print("-" * 40)
    for strategy, thresholds in optimal_thresholds.items():
        print(f"\n{strategy.upper()} ({thresholds['criterion']}):")
        if config.classification_mode == "binary":
            print(f"  Low Threshold: {thresholds['low_threshold']:.3f}")
            print(f"  High Threshold: {thresholds['high_threshold']:.3f}")
        else:
            print(f"  Confidence Low: {thresholds['confidence_threshold_low']:.3f}")
            print(f"  Confidence High: {thresholds['confidence_threshold_high']:.3f}")
            print(f"  Uncertainty: {thresholds['uncertainty_threshold']:.3f}")
        
        print(f"  F1 Score: {thresholds.get('f1_score', thresholds.get('f1_macro', 0)):.3f}")
        print(f"  Accuracy: {thresholds['accuracy']:.3f}")
        print(f"  Uncertainty: {thresholds['uncertain_percentage']:.1%}")
        print(f"  Coverage: {thresholds['coverage']:.1%}")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    for use_case, recommendation in recommendations.items():
        print(f"\n{use_case.upper()}:")
        print(f"  {recommendation}")
    
    print(f"\nDetailed results and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()