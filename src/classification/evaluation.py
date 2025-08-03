"""
Evaluation utilities for cell classification models.
Separated from core inference logic for better modularity.
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from classification.inference import CellClassifier

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles comprehensive evaluation of cell classification models.
    """
    
    def __init__(self, classifier: CellClassifier):
        self.classifier = classifier
        self.config = classifier.config
    
    def evaluate_test_set(self, test_loader, test_data: List[Tuple[Path, int]], 
                         output_dir: Path, visualize: bool = True, 
                         num_visualize_each: int = 5) -> Dict[str, Union[float, int, List[Dict]]]:
        """
        Comprehensive evaluation of the test set with detailed analysis.
        
        Args:
            test_loader: DataLoader for test set
            test_data: List of (image_path, label) tuples for the test set
            output_dir: Directory to save outputs
            visualize: Whether to create visualizations
            num_visualize_each: Number of examples to visualize for each category
            
        Returns:
            Dict containing comprehensive evaluation metrics and analysis
        """
        logger.info("--- Evaluating on the entire Test Set ---")
        self.classifier.model.eval()
        
        if self.config.classification_mode == "binary":
            return self._evaluate_binary_test_set(test_loader, test_data, output_dir, visualize, num_visualize_each)
        else:
            return self._evaluate_ternary_test_set(test_loader, test_data, output_dir, visualize, num_visualize_each)
    
    def _evaluate_binary_test_set(self, test_loader, test_data: List[Tuple[Path, int]], 
                                 output_dir: Path, visualize: bool, num_visualize_each: int) -> Dict:
        """Evaluate binary classification test set."""
        # Initialize tracking variables
        all_preds_binary_at_0_5 = []
        all_labels_test = []
        all_probs_test = []
        
        # For detailed analysis based on thresholds
        undecided_test_count = 0
        undecided_predictions_details = []
        
        false_positives_decided = 0
        false_negatives_decided = 0
        true_positives_decided = 0
        true_negatives_decided = 0
        
        # Lists to store misclassified samples
        misclassified_false_positives = []
        misclassified_false_negatives = []
        
        # Map filenames to test data
        test_data_map = {test_data[i][0].stem: (test_data[i][0], test_data[i][1]) for i in range(len(test_data))}
        
        with torch.no_grad():
            for inputs, labels, filenames_batch in tqdm(test_loader, desc="Evaluating Test Set"):
                inputs, labels = inputs.to(self.classifier.device), labels.to(self.classifier.device)
                outputs = self.classifier.model(inputs)
                probs = torch.sigmoid(outputs)
                
                # For standard metrics (at 0.5 threshold)
                all_preds_binary_at_0_5.extend((probs > 0.5).long().cpu().numpy().flatten())
                all_labels_test.extend(labels.cpu().numpy().flatten())
                all_probs_test.extend(probs.cpu().numpy().flatten())
                
                # Detailed analysis based on custom thresholds
                for i, prob_val in enumerate(probs.cpu().numpy().flatten()):
                    current_filename_stem = filenames_batch[i]
                    original_img_path, original_true_label = test_data_map.get(current_filename_stem, (None, None))
                    
                    # Determine the model's 'decided' prediction based on thresholds
                    predicted_category = "uncertain"
                    if prob_val >= self.classifier.confidence_threshold_high:
                        predicted_category = "cancerous"
                    elif prob_val <= self.classifier.confidence_threshold_low:
                        predicted_category = "non-cancerous"
                    
                    # Check for misclassifications among decided predictions
                    if predicted_category == "cancerous":
                        if original_true_label == 0:  # FP
                            false_positives_decided += 1
                            misclassified_false_positives.append({
                                "img_path": original_img_path,
                                "true_label": original_true_label,
                                "predicted_label": predicted_category,
                                "probability": prob_val
                            })
                        else:  # TP
                            true_positives_decided += 1
                    elif predicted_category == "non-cancerous":
                        if original_true_label == 1:  # FN
                            false_negatives_decided += 1
                            misclassified_false_negatives.append({
                                "img_path": original_img_path,
                                "true_label": original_true_label,
                                "predicted_label": predicted_category,
                                "probability": prob_val
                            })
                        else:  # TN
                            true_negatives_decided += 1
                    else:  # Undecided
                        undecided_test_count += 1
                        if original_img_path:
                            undecided_predictions_details.append({
                                "img_path": original_img_path,
                                "true_label": original_true_label,
                                "probability": prob_val,
                                "distance_from_0_5": abs(prob_val - 0.5)
                            })
        
        # Calculate standard metrics
        test_accuracy = accuracy_score(all_labels_test, all_preds_binary_at_0_5)
        test_precision = precision_score(all_labels_test, all_preds_binary_at_0_5, zero_division=0)
        test_recall = recall_score(all_labels_test, all_preds_binary_at_0_5, zero_division=0) if len(all_labels_test) > 0 else 0.0
        test_f1 = f1_score(all_labels_test, all_preds_binary_at_0_5, zero_division=0)
        test_auc = roc_auc_score(all_labels_test, all_probs_test) if len(np.unique(all_labels_test)) > 1 else 0.0
        
        total_test_samples = len(all_labels_test)
        undecided_test_percentage = (undecided_test_count / total_test_samples) if total_test_samples > 0 else 0.0
        
        # Calculate decided samples metrics (excluding uncertain predictions)
        total_decided = true_positives_decided + false_positives_decided + true_negatives_decided + false_negatives_decided
        decided_percentage = (total_decided / total_test_samples) if total_test_samples > 0 else 0.0
        
        # Relative percentages among decided predictions only
        if total_decided > 0:
            relative_cancerous_decided = (true_positives_decided + false_positives_decided) / total_decided * 100
            relative_non_cancerous_decided = (true_negatives_decided + false_negatives_decided) / total_decided * 100
            
            # Accuracy among decided predictions
            decided_accuracy = (true_positives_decided + true_negatives_decided) / total_decided if total_decided > 0 else 0.0
        else:
            relative_cancerous_decided = 0.0
            relative_non_cancerous_decided = 0.0
            decided_accuracy = 0.0
        
        # Log results
        logger.info(f"\nFinal Test Set Metrics (at 0.5 binarization threshold):")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1 Score: {test_f1:.4f}")
        logger.info(f"  AUC: {test_auc:.4f}")
        
        logger.info(f"\nClassification Breakdown (using thresholds {self.classifier.confidence_threshold_low:.2f}-{self.classifier.confidence_threshold_high:.2f}):")
        logger.info(f"  True Positives (Decided): {true_positives_decided}")
        logger.info(f"  False Positives (Decided): {false_positives_decided}")
        logger.info(f"  True Negatives (Decided): {true_negatives_decided}")
        logger.info(f"  False Negatives (Decided): {false_negatives_decided}")
        logger.info(f"  Total Decided Predictions: {total_decided} ({decided_percentage:.2%})")
        logger.info(f"  Total Undecided Predictions: {undecided_test_count} ({undecided_test_percentage:.2%})")
        logger.info(f"  Total Samples Evaluated: {total_test_samples}")
        
        logger.info(f"\nRelative Distribution Among Decided Predictions:")
        logger.info(f"  Predicted Cancerous: {relative_cancerous_decided:.1f}%")
        logger.info(f"  Predicted Non-Cancerous: {relative_non_cancerous_decided:.1f}%")
        logger.info(f"  Accuracy Among Decided: {decided_accuracy:.4f}")
        
        # Visualization and analysis
        if visualize:
            self._visualize_undecided_predictions(undecided_predictions_details, num_visualize_each)
            self._visualize_misclassified_samples(misclassified_false_positives, misclassified_false_negatives, num_visualize_each)
            self._plot_probability_distribution(all_probs_test, all_labels_test, output_dir)
            self._plot_confidence_distribution(all_probs_test, output_dir)
        
        # Results
        results = {
            "standard_metrics": {
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
                "auc": test_auc
            },
            "threshold_analysis": {
                "true_positives_decided": true_positives_decided,
                "false_positives_decided": false_positives_decided,
                "true_negatives_decided": true_negatives_decided,
                "false_negatives_decided": false_negatives_decided,
                "undecided_count": undecided_test_count,
                "undecided_percentage": undecided_test_percentage,
                "total_samples": total_test_samples,
                "total_decided": total_decided,
                "decided_percentage": decided_percentage,
                "decided_accuracy": decided_accuracy
            },
            "relative_analysis": {
                "relative_cancerous_decided": relative_cancerous_decided,
                "relative_non_cancerous_decided": relative_non_cancerous_decided,
                "decided_only_metrics": {
                    "accuracy": decided_accuracy,
                    "precision": (true_positives_decided / (true_positives_decided + false_positives_decided)) if (true_positives_decided + false_positives_decided) > 0 else 0.0,
                    "recall": (true_positives_decided / (true_positives_decided + false_negatives_decided)) if (true_positives_decided + false_negatives_decided) > 0 else 0.0
                }
            },
            "detailed_analysis": {
                "undecided_details": undecided_predictions_details,
                "false_positives": misclassified_false_positives,
                "false_negatives": misclassified_false_negatives
            }
        }
        
        # Save results to JSON
        self._save_evaluation_results(results, output_dir, "binary")
        
        return results
    
    def _evaluate_ternary_test_set(self, test_loader, test_data: List[Tuple[Path, int]], 
                                  output_dir: Path, visualize: bool, num_visualize_each: int) -> Dict:
        """Evaluate ternary classification test set."""
        all_preds = []
        all_labels_test = []
        all_probs_test = []
        all_confidences = []
        
        uncertain_predictions = []
        misclassified_samples = []
        decided_predictions = []
        
        # Map filenames to test data
        test_data_map = {test_data[i][0].stem: (test_data[i][0], test_data[i][1]) for i in range(len(test_data))}
        
        with torch.no_grad():
            for inputs, labels, filenames_batch in tqdm(test_loader, desc="Evaluating Test Set"):
                inputs, labels = inputs.to(self.classifier.device), labels.to(self.classifier.device)
                outputs = self.classifier.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels_test.extend(labels.cpu().numpy())
                all_probs_test.extend(probs.cpu().numpy())
                
                # Detailed analysis
                for i, (pred, true_label, prob_array) in enumerate(zip(preds.cpu().numpy(), 
                                                                      labels.cpu().numpy(), 
                                                                      probs.cpu().numpy())):
                    current_filename_stem = filenames_batch[i]
                    original_img_path, _ = test_data_map.get(current_filename_stem, (None, None))
                    
                    max_prob = np.max(prob_array)
                    sorted_probs = np.sort(prob_array)[::-1]
                    confidence = sorted_probs[0] - sorted_probs[1]
                    all_confidences.append(confidence)
                    
                    # Check if prediction is uncertain based on thresholds
                    if confidence < self.classifier.uncertainty_threshold or max_prob <= self.classifier.confidence_threshold_low:
                        uncertain_predictions.append({
                            "img_path": original_img_path,
                            "true_label": true_label,
                            "predicted_label": pred,
                            "probabilities": prob_array.tolist(),
                            "confidence": confidence
                        })
                    else:
                        # This is a decided prediction
                        decided_predictions.append({
                            "img_path": original_img_path,
                            "true_label": true_label,
                            "predicted_label": pred,
                            "probabilities": prob_array.tolist(),
                            "confidence": confidence
                        })
                        
                        if pred != true_label:  # Misclassified among decided
                            misclassified_samples.append({
                                "img_path": original_img_path,
                                "true_label": true_label,
                                "predicted_label": pred,
                                "probabilities": prob_array.tolist(),
                                "confidence": confidence
                            })
        
        # Calculate metrics
        test_accuracy = accuracy_score(all_labels_test, all_preds)
        test_precision_macro = precision_score(all_labels_test, all_preds, average='macro', zero_division=0)
        test_recall_macro = recall_score(all_labels_test, all_preds, average='macro', zero_division=0)
        test_f1_macro = f1_score(all_labels_test, all_preds, average='macro', zero_division=0)
        
        test_precision_weighted = precision_score(all_labels_test, all_preds, average='weighted', zero_division=0)
        test_recall_weighted = recall_score(all_labels_test, all_preds, average='weighted', zero_division=0)
        test_f1_weighted = f1_score(all_labels_test, all_preds, average='weighted', zero_division=0)
        
        # Calculate decided-only metrics
        total_samples = len(all_labels_test)
        decided_count = len(decided_predictions)
        uncertain_count = len(uncertain_predictions)
        decided_percentage = (decided_count / total_samples) if total_samples > 0 else 0.0
        uncertain_percentage = (uncertain_count / total_samples) if total_samples > 0 else 0.0
        
        # Relative distribution among decided predictions
        if decided_count > 0:
            decided_pred_counts = {}
            class_names = ["non-cancerous", "cancerous", "false-positive"]
            for class_idx in range(3):
                count = sum(1 for dp in decided_predictions if dp['predicted_label'] == class_idx)
                decided_pred_counts[class_names[class_idx]] = count
            
            relative_percentages = {
                class_name: (count / decided_count * 100) for class_name, count in decided_pred_counts.items()
            }
            
            # Accuracy among decided predictions
            decided_accuracy = sum(1 for dp in decided_predictions if dp['predicted_label'] == dp['true_label']) / decided_count
        else:
            decided_pred_counts = {name: 0 for name in ["non-cancerous", "cancerous", "false-positive"]}
            relative_percentages = {name: 0.0 for name in ["non-cancerous", "cancerous", "false-positive"]}
            decided_accuracy = 0.0
        
        # Log results
        logger.info(f"\nFinal Test Set Metrics (Ternary Classification):")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision (Macro): {test_precision_macro:.4f}")
        logger.info(f"  Recall (Macro): {test_recall_macro:.4f}")
        logger.info(f"  F1 Score (Macro): {test_f1_macro:.4f}")
        logger.info(f"  Precision (Weighted): {test_precision_weighted:.4f}")
        logger.info(f"  Recall (Weighted): {test_recall_weighted:.4f}")
        logger.info(f"  F1 Score (Weighted): {test_f1_weighted:.4f}")
        
        logger.info(f"\nClassification Breakdown:")
        logger.info(f"  Total Decided Predictions: {decided_count} ({decided_percentage:.2%})")
        logger.info(f"  Total Uncertain Predictions: {uncertain_count} ({uncertain_percentage:.2%})")
        logger.info(f"  Decided-Only Accuracy: {decided_accuracy:.4f}")
        
        logger.info(f"\nRelative Distribution Among Decided Predictions:")
        for class_name, percentage in relative_percentages.items():
            logger.info(f"  {class_name.capitalize()}: {decided_pred_counts[class_name]} ({percentage:.1f}%)")
        
        logger.info(f"\nMisclassified among decided: {len(misclassified_samples)}")
        
        # Visualization
        if visualize:
            self._visualize_ternary_analysis(uncertain_predictions, misclassified_samples, num_visualize_each)
            self._plot_ternary_probability_distribution(all_probs_test, all_labels_test, output_dir)
            self._plot_confidence_distribution_ternary(all_confidences, output_dir)
        
        # Prepare results
        results = {
            "standard_metrics": {
                "accuracy": test_accuracy,
                "precision_macro": test_precision_macro,
                "recall_macro": test_recall_macro,
                "f1_macro": test_f1_macro,
                "precision_weighted": test_precision_weighted,
                "recall_weighted": test_recall_weighted,
                "f1_weighted": test_f1_weighted
            },
            "threshold_analysis": {
                "decided_count": decided_count,
                "uncertain_count": uncertain_count,
                "decided_percentage": decided_percentage,
                "uncertain_percentage": uncertain_percentage,
                "decided_accuracy": decided_accuracy,
                "total_samples": total_samples
            },
            "relative_analysis": {
                "decided_prediction_counts": decided_pred_counts,
                "relative_percentages_decided": relative_percentages,
                "decided_only_metrics": {
                    "accuracy": decided_accuracy,
                    "total_decided": decided_count
                }
            },
            "detailed_analysis": {
                "uncertain_predictions": uncertain_predictions,
                "misclassified_samples": misclassified_samples,
                "decided_predictions": decided_predictions
            }
        }
        
        # Save results
        self._save_evaluation_results(results, output_dir, "ternary")
        
        return results
    
    def _visualize_undecided_predictions(self, undecided_details: List[Dict], num_to_show: int):
        """Visualize undecided predictions for binary classification."""
        if not undecided_details:
            logger.info("\nNo undecided predictions found in the test set to visualize.")
            return
        
        logger.info("\n--- Visualizing Undecided Predictions ---")
        
        # Sort by distance from 0.5
        undecided_details.sort(key=lambda x: x['distance_from_0_5'])
        
        num_to_show_each = min(num_to_show, len(undecided_details) // 2)
        if num_to_show_each == 0 and len(undecided_details) > 0:
            num_to_show_each = 1
        
        logger.info(f"\n--- {num_to_show_each} Most Undecided Predictions (closest to 0.5) ---")
        for detail in undecided_details[:num_to_show_each]:
            logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                       f"Pred: uncertain (Prob: {detail['probability']:.4f})")
            try:
                from classification.utils import visualize_cell
                visualize_cell(
                    detail['img_path'],
                    detail['true_label'],
                    predicted_label="uncertain",
                    probability=detail['probability'],
                    classification_mode=self.config.classification_mode
                )
            except ImportError:
                logger.warning("Visualization utilities not available")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        logger.info(f"\n--- {num_to_show_each} Least Undecided Predictions (closest to thresholds) ---")
        start_index = max(0, len(undecided_details) - num_to_show_each)
        for detail in undecided_details[start_index:]:
            logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                       f"Pred: uncertain (Prob: {detail['probability']:.4f})")
            try:
                from classification.utils import visualize_cell
                visualize_cell(
                    detail['img_path'],
                    detail['true_label'],
                    predicted_label="uncertain",
                    probability=detail['probability'],
                    classification_mode=self.config.classification_mode
                )
            except ImportError:
                logger.warning("Visualization utilities not available")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
    
    def _visualize_misclassified_samples(self, false_positives: List[Dict], false_negatives: List[Dict], num_to_show: int):
        """Visualize misclassified samples for binary classification."""
        logger.info("\n--- Visualizing Misclassified Cells (False Positives and False Negatives) ---")
        
        num_fp_to_show = min(num_to_show, len(false_positives))
        if num_fp_to_show > 0:
            logger.info(f"\n--- {num_fp_to_show} False Positives (Predicted Cancerous, True Non-Cancerous) ---")
            for detail in false_positives[:num_fp_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                           f"Pred: {detail['predicted_label']} (Prob: {detail['probability']:.4f})")
                try:
                    from classification.utils import visualize_cell
                    visualize_cell(
                        detail['img_path'],
                        detail['true_label'],
                        predicted_label=detail['predicted_label'],
                        probability=detail['probability'],
                        classification_mode=self.config.classification_mode
                    )
                except ImportError:
                    logger.warning("Visualization utilities not available")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
        else:
            logger.info("\nNo False Negatives found among decided predictions.")

        num_fn_to_show = min(num_to_show, len(false_negatives))
        if num_fn_to_show > 0:
            logger.info(f"\n--- {num_fn_to_show} False Negatives (Predicted Non-Cancerous, True Cancerous) ---")
            for detail in false_negatives[:num_fn_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                           f"Pred: {detail['predicted_label']} (Prob: {detail['probability']:.4f})")
                try:
                    from classification.utils import visualize_cell
                    visualize_cell(
                        detail['img_path'],
                        detail['true_label'],
                        predicted_label=detail['predicted_label'],
                        probability=detail['probability'],
                        classification_mode=self.config.classification_mode
                    )
                except ImportError:
                    logger.warning("Visualization utilities not available")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
    
    def _visualize_ternary_analysis(self, uncertain_predictions: List[Dict], misclassified_samples: List[Dict], num_to_show: int):
        """Visualize uncertain and misclassified samples for ternary classification."""
        logger.info("\n--- Visualizing Ternary Classification Analysis ---")
        
        class_names = ["non-cancerous", "cancerous", "false-positive"]
        
        # Visualize uncertain predictions
        num_uncertain_to_show = min(num_to_show, len(uncertain_predictions))
        if num_uncertain_to_show > 0:
            logger.info(f"\n--- {num_uncertain_to_show} Uncertain Predictions ---")
            for detail in uncertain_predictions[:num_uncertain_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {class_names[detail['true_label']]}, "
                           f"Confidence: {detail['confidence']:.4f}")
                try:
                    from classification.utils import visualize_cell
                    visualize_cell(
                        detail['img_path'],
                        detail['true_label'],
                        predicted_label="uncertain",
                        probability=detail['probabilities'],
                        classification_mode=self.config.classification_mode
                    )
                except ImportError:
                    logger.warning("Visualization utilities not available")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
        else:
            logger.info("\nNo uncertain predictions found.")
        
        # Visualize misclassified samples
        num_misclassified_to_show = min(num_to_show, len(misclassified_samples))
        if num_misclassified_to_show > 0:
            logger.info(f"\n--- {num_misclassified_to_show} Misclassified Samples ---")
            for detail in misclassified_samples[:num_misclassified_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {class_names[detail['true_label']]}, "
                           f"Pred: {class_names[detail['predicted_label']]}, Confidence: {detail['confidence']:.4f}")
                try:
                    from classification.utils import visualize_cell
                    visualize_cell(
                        detail['img_path'],
                        detail['true_label'],
                        predicted_label=class_names[detail['predicted_label']],
                        probability=detail['probabilities'],
                        classification_mode=self.config.classification_mode
                    )
                except ImportError:
                    logger.warning("Visualization utilities not available")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
        else:
            logger.info("\nNo misclassified samples found.")
    
    def _plot_probability_distribution(self, all_probs: List[float], all_labels: List[int], output_dir: Path):
        """Plot probability distribution for binary classification."""
        logger.info("\n--- Probability Distribution and Thresholds ---")
        
        if len(all_probs) == 0:
            logger.warning("No probabilities to plot for the test set.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Probability distribution by true class
        probs_positive = [p for i, p in enumerate(all_probs) if all_labels[i] == 1]
        probs_negative = [p for i, p in enumerate(all_probs) if all_labels[i] == 0]
        
        ax1.hist(probs_negative, bins=50, alpha=0.6, label='True Negative', 
                color='skyblue', density=True)
        ax1.hist(probs_positive, bins=50, alpha=0.6, label='True Positive', 
                color='lightcoral', density=True)
        
        # Plot thresholds
        ax1.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                   label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
        ax1.axvline(x=0.5, color='gray', linestyle=':', label='Standard 0.5 Threshold')
        ax1.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                   label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
        
        ax1.set_title('Probability Distribution by True Class')
        ax1.set_xlabel('Predicted Probability of being Cancerous')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence distribution (distance from 0.5)
        confidences = [max(p, 1-p) for p in all_probs]
        ax2.hist(confidences, bins=50, alpha=0.7, color='purple', density=True)
        ax2.axvline(x=0.5, color='gray', linestyle=':', label='Minimum Confidence')
        ax2.axvline(x=np.mean(confidences), color='orange', linestyle='-', 
                   label=f'Mean Confidence ({np.mean(confidences):.3f})')
        ax2.set_title('Model Confidence Distribution')
        ax2.set_xlabel('Confidence (Distance from 0.5)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "binary_probability_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_distribution(self, all_probs: List[float], output_dir: Path):
        """Plot detailed confidence analysis for binary classification."""
        if len(all_probs) == 0:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate confidence levels
        confidences = [max(p, 1-p) for p in all_probs]
        
        # Plot 1: Confidence histogram with threshold regions
        ax1.hist(confidences, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                   label=f'High Confidence ({self.classifier.confidence_threshold_high:.2f})')
        ax1.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                   label=f'Low Confidence ({self.classifier.confidence_threshold_low:.2f})')
        ax1.set_title('Confidence Distribution')
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Probability vs Confidence scatter
        ax2.scatter(all_probs, confidences, alpha=0.5, s=20)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Probability vs Confidence')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Decision regions
        decisions = []
        for p in all_probs:
            if p >= self.classifier.confidence_threshold_high:
                decisions.append('Cancerous')
            elif p <= self.classifier.confidence_threshold_low:
                decisions.append('Non-Cancerous')
            else:
                decisions.append('Uncertain')
        
        decision_counts = {d: decisions.count(d) for d in ['Cancerous', 'Non-Cancerous', 'Uncertain']}
        colors = ['red', 'blue', 'gray']
        ax3.pie(decision_counts.values(), labels=decision_counts.keys(), colors=colors, autopct='%1.1f%%')
        ax3.set_title('Decision Distribution')
        
        # Plot 4: Box plot of probabilities by decision
        prob_by_decision = {
            'Cancerous': [p for p in all_probs if p >= self.classifier.confidence_threshold_high],
            'Non-Cancerous': [p for p in all_probs if p <= self.classifier.confidence_threshold_low],
            'Uncertain': [p for p in all_probs if self.classifier.confidence_threshold_low < p < self.classifier.confidence_threshold_high]
        }
        
        box_data = [prob_by_decision[key] for key in ['Non-Cancerous', 'Uncertain', 'Cancerous'] if prob_by_decision[key]]
        box_labels = [key for key in ['Non-Cancerous', 'Uncertain', 'Cancerous'] if prob_by_decision[key]]
        
        if box_data:
            ax4.boxplot(box_data, labels=box_labels)
            ax4.set_title('Probability Distribution by Decision')
            ax4.set_ylabel('Probability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "binary_confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_ternary_probability_distribution(self, all_probs: List[np.ndarray], all_labels: List[int], output_dir: Path):
        """Plot probability distribution for ternary classification."""
        if len(all_probs) == 0:
            return
            
        class_names = ["Non-Cancerous", "Cancerous", "False-Positive"]
        colors = ['blue', 'red', 'orange']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot class-specific probability distributions
        for class_idx in range(3):
            ax = axes[class_idx]
            
            # Get probabilities for this class, separated by true label
            for true_label in range(3):
                probs_for_class = [probs[class_idx] for i, probs in enumerate(all_probs) if all_labels[i] == true_label]
                if probs_for_class:
                    ax.hist(probs_for_class, bins=30, alpha=0.6, 
                           label=f'True {class_names[true_label]}', 
                           density=True)
            
            ax.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                      label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
            ax.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                      label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
            
            ax.set_title(f'Predicted Probability Distribution: {class_names[class_idx]}')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Overall confidence distribution
        ax4 = axes[3]
        confidences = []
        for probs in all_probs:
            sorted_probs = np.sort(probs)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]
            confidences.append(confidence)
        
        ax4.hist(confidences, bins=50, alpha=0.7, color='purple', density=True)
        ax4.axvline(x=self.classifier.uncertainty_threshold, color='red', linestyle='--', 
                   label=f'Uncertainty Threshold ({self.classifier.uncertainty_threshold:.2f})')
        ax4.axvline(x=np.mean(confidences), color='orange', linestyle='-', 
                   label=f'Mean Confidence ({np.mean(confidences):.3f})')
        ax4.set_title('Model Confidence Distribution')
        ax4.set_xlabel('Confidence (Prob Difference)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ternary_probability_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_distribution_ternary(self, all_confidences: List[float], output_dir: Path):
        """Plot detailed confidence analysis for ternary classification."""
        if len(all_confidences) == 0:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Confidence histogram
        ax1.hist(all_confidences, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=self.classifier.uncertainty_threshold, color='red', linestyle='--', 
                   label=f'Uncertainty Threshold ({self.classifier.uncertainty_threshold:.2f})')
        ax1.axvline(x=np.mean(all_confidences), color='orange', linestyle='-', 
                   label=f'Mean ({np.mean(all_confidences):.3f})')
        ax1.set_title('Confidence Distribution')
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence levels
        high_conf = [c for c in all_confidences if c >= 0.7]
        medium_conf = [c for c in all_confidences if 0.3 <= c < 0.7]
        low_conf = [c for c in all_confidences if c < 0.3]
        
        conf_counts = [len(high_conf), len(medium_conf), len(low_conf)]
        labels = ['High (≥0.7)', 'Medium (0.3-0.7)', 'Low (<0.3)']
        colors = ['green', 'orange', 'red']
        
        ax2.pie(conf_counts, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Confidence Level Distribution')
        
        # Plot 3: Confidence vs Decision
        decisions = ['Certain' if c >= self.classifier.uncertainty_threshold else 'Uncertain' for c in all_confidences]
        decision_counts = {d: decisions.count(d) for d in ['Certain', 'Uncertain']}
        
        ax3.bar(decision_counts.keys(), decision_counts.values(), color=['green', 'red'], alpha=0.7)
        ax3.set_title('Decision Certainty Distribution')
        ax3.set_ylabel('Count')
        
        # Add percentage labels
        total = sum(decision_counts.values())
        for i, (key, value) in enumerate(decision_counts.items()):
            pct = value / total * 100 if total > 0 else 0
            ax3.text(i, value + total * 0.01, f'{pct:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Confidence statistics
        stats_text = f"""Confidence Statistics:
        
Mean: {np.mean(all_confidences):.4f}
Median: {np.median(all_confidences):.4f}
Std: {np.std(all_confidences):.4f}
Min: {np.min(all_confidences):.4f}
Max: {np.max(all_confidences):.4f}

Percentiles:
25%: {np.percentile(all_confidences, 25):.4f}
75%: {np.percentile(all_confidences, 75):.4f}
90%: {np.percentile(all_confidences, 90):.4f}
95%: {np.percentile(all_confidences, 95):.4f}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Confidence Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "ternary_confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_evaluation_results(self, results: Dict, output_dir: Path, mode: str):
        """Save evaluation results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths_to_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_paths_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        serializable_results = convert_paths_to_strings(results)
        
        output_file = output_dir / f"{mode}_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4, default=str)
        
        logger.info(f"Evaluation results saved to {output_file}")


def evaluate_model_on_test_set(classifier: CellClassifier, test_loader, test_data: List[Tuple[Path, int]], 
                              output_dir: Path, visualize: bool = True, 
                              num_visualize_each: int = 5) -> Dict:
    """
    Convenience function to evaluate a model on a test set.
    
    Args:
        classifier: CellClassifier instance
        test_loader: DataLoader for test set
        test_data: List of (image_path, label) tuples for the test set
        output_dir: Directory to save outputs
        visualize: Whether to create visualizations
        num_visualize_each: Number of examples to visualize for each category
        
    Returns:
        Dict containing comprehensive evaluation metrics and analysis
    """
    evaluator = ModelEvaluator(classifier)
    return evaluator.evaluate_test_set(test_loader, test_data, output_dir, visualize, num_visualize_each)
