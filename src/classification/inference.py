import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import json

from classification.models import get_classification_model
from classification.transforms import get_classification_transforms
from classification.config import ClassificationConfig
from classification.utils import visualize_cell

logger = logging.getLogger(__name__)

class CellClassifier:
    """
    Handles inference for the cell classification model.
    Supports both binary and ternary classification modes.
    """
    def __init__(
        self,
        model_path: Union[str, Path],
        config: ClassificationConfig,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device if device else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = get_classification_model(
            model_name=config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            input_channels=3, # Assuming RGB input
            input_image_size=config.image_size,
            classification_mode=config.classification_mode
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        logger.info(f"Classification model loaded from {model_path} on device: {self.device}")
        logger.info(f"Classification mode: {config.classification_mode}")

        self.transform = get_classification_transforms(
            image_size=config.image_size,
            mean=tuple(config.normalize_mean),
            std=tuple(config.normalize_std),
            is_train=False # Always use validation/inference transforms
        )
        
        self.confidence_threshold_high = config.confidence_threshold_high
        self.confidence_threshold_low = config.confidence_threshold_low
        self.uncertainty_threshold = config.uncertainty_threshold
        
    def _preprocess_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Helper to load and preprocess a single image."""
        if isinstance(img_input, (str, Path)):
            img_path = Path(img_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            if img_path.suffix.lower() == '.npy':
                img_np = np.load(img_path)
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = (img_np / img_np.max() * 255).astype(np.uint8)

                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                elif img_np.ndim == 3 and img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                elif img_np.ndim == 3 and img_np.shape[2] == 1:
                    img_np = np.repeat(img_np, 3, axis=2)
                elif img_np.ndim == 3 and img_np.shape[2] not in [1, 3]:
                    raise ValueError(f"Unsupported number of channels for .npy image: {img_np.shape[2]} at {img_path}")
                img = Image.fromarray(img_np)
            else:
                img = Image.open(img_path).convert("RGB") # Ensure RGB
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                if img_input.max() <= 1.0:
                    img_input = (img_input * 255).astype(np.uint8)
                else:
                    img_input = (img_input / img_input.max() * 255).astype(np.uint8)

            if img_input.ndim == 2:
                img_input = np.stack([img_input]*3, axis=-1)
            elif img_input.ndim == 3 and img_input.shape[2] == 4:
                img_input = img_input[:, :, :3]
            elif img_input.ndim == 3 and img_input.shape[2] == 1:
                img_input = np.repeat(img_input, 3, axis=2)
            elif img_input.ndim == 3 and img_input.shape[2] not in [1, 3]:
                raise ValueError(f"Unsupported number of channels for numpy array: {img_input.shape[2]}")
            img = Image.fromarray(img_input)
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")
            
        return self.transform(img).unsqueeze(0) # Add batch dimension

    def classify_single_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Union[str, float, List[float]]]:
        """
        Classifies a single cell image.
        
        Args:
            img_input (Union[str, Path, np.ndarray, Image.Image]): Path to the image file,
                                                                   or a loaded NumPy array/PIL Image.
                                                                   
        Returns:
            Dict[str, Union[str, float, List[float]]]: A dictionary with classification results.
                For binary: {'prediction': str, 'probability': float}
                For ternary: {'prediction': str, 'probabilities': List[float], 'confidence': float}
        """
        img_tensor = self._preprocess_image(img_input).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            
            if self.config.classification_mode == "binary":
                return self._process_binary_output(outputs)
            else:
                return self._process_ternary_output(outputs)

    def _process_binary_output(self, outputs: torch.Tensor) -> Dict[str, Union[str, float]]:
        """Process outputs for binary classification."""
        probability = torch.sigmoid(outputs).item()

        if probability >= self.confidence_threshold_high:
            prediction = "cancerous"
        elif probability <= self.confidence_threshold_low:
            prediction = "non-cancerous"
        else:
            prediction = "uncertain"
        
        return {
            "prediction": prediction, 
            "probability": probability,
            "confidence": max(probability, 1.0 - probability)  # Distance from 0.5
        }

    def _process_ternary_output(self, outputs: torch.Tensor) -> Dict[str, Union[str, List[float], float]]:
        """Process outputs for ternary classification."""
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        
        # Calculate confidence as difference between highest and second highest probability
        sorted_probs = np.sort(probabilities)[::-1]  # Sort in descending order
        confidence = sorted_probs[0] - sorted_probs[1]
        
        class_names = ["non-cancerous", "cancerous", "false-positive"]
        
        # Determine prediction based on confidence and thresholds
        if confidence < self.uncertainty_threshold:
            prediction = "uncertain"
        elif max_prob >= self.confidence_threshold_high:
            prediction = class_names[max_prob_idx]
        elif max_prob <= self.confidence_threshold_low:
            prediction = "uncertain"
        else:
            prediction = class_names[max_prob_idx]
        
        return {
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
            "confidence": confidence,
            "class_probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }

    def classify_batch(self, img_inputs: List[Union[str, Path, np.ndarray, Image.Image]]) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        Classifies a batch of cell images.
        
        Args:
            img_inputs (List[Union[str, Path, np.ndarray, Image.Image]]): List of image inputs.
                                                                           
        Returns:
            List[Dict[str, Union[str, float, List[float]]]]: A list of classification results for each image.
        """
        if not img_inputs:
            return []

        processed_tensors = [self._preprocess_image(img).squeeze(0) for img in img_inputs] # Remove batch dim temporarily
        # Check if all tensors have same shape before stacking
        if not all(t.shape == processed_tensors[0].shape for t in processed_tensors):
            logger.error("Images in batch have differing processed shapes. Cannot batch.")
            # Fallback to single image classification if shapes differ after preprocessing
            return [self.classify_single_image(img_input) for img_input in img_inputs]

        batch_tensor = torch.stack(processed_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            
            if self.config.classification_mode == "binary":
                return self._process_binary_batch_output(outputs)
            else:
                return self._process_ternary_batch_output(outputs)

    def _process_binary_batch_output(self, outputs: torch.Tensor) -> List[Dict[str, Union[str, float]]]:
        """Process batch outputs for binary classification."""
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        results = []
        for prob in probabilities:
            if prob >= self.confidence_threshold_high:
                prediction = "cancerous"
            elif prob <= self.confidence_threshold_low:
                prediction = "non-cancerous"
            else:
                prediction = "uncertain"
            results.append({
                "prediction": prediction, 
                "probability": float(prob),
                "confidence": max(prob, 1.0 - prob)
            })
            
        return results

    def _process_ternary_batch_output(self, outputs: torch.Tensor) -> List[Dict[str, Union[str, List[float], float]]]:
        """Process batch outputs for ternary classification."""
        probabilities_batch = torch.softmax(outputs, dim=1).cpu().numpy()
        class_names = ["non-cancerous", "cancerous", "false-positive"]
        
        results = []
        for probabilities in probabilities_batch:
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            
            # Calculate confidence
            sorted_probs = np.sort(probabilities)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]
            
            # Determine prediction
            if confidence < self.uncertainty_threshold:
                prediction = "uncertain"
            elif max_prob >= self.confidence_threshold_high:
                prediction = class_names[max_prob_idx]
            elif max_prob <= self.confidence_threshold_low:
                prediction = "uncertain"
            else:
                prediction = class_names[max_prob_idx]
            
            results.append({
                "prediction": prediction,
                "probabilities": probabilities.tolist(),
                "confidence": float(confidence),
                "class_probabilities": {
                    class_names[i]: float(prob) for i, prob in enumerate(probabilities)
                }
            })
            
        return results

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
        self.model.eval()
        
        # Initialize tracking variables
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
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
                    if prob_val >= self.confidence_threshold_high:
                        predicted_category = "cancerous"
                    elif prob_val <= self.confidence_threshold_low:
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
        
        # Log results
        logger.info(f"\nFinal Test Set Metrics (at 0.5 binarization threshold):")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1 Score: {test_f1:.4f}")
        logger.info(f"  AUC: {test_auc:.4f}")
        
        logger.info(f"\nClassification Breakdown (using thresholds {self.config.confidence_threshold_low:.2f}-{self.config.confidence_threshold_high:.2f}):")
        logger.info(f"  True Positives (Decided): {true_positives_decided}")
        logger.info(f"  False Positives (Decided): {false_positives_decided}")
        logger.info(f"  True Negatives (Decided): {true_negatives_decided}")
        logger.info(f"  False Negatives (Decided): {false_negatives_decided}")
        logger.info(f"  Total Decided Predictions: {true_positives_decided + false_positives_decided + true_negatives_decided + false_negatives_decided}")
        logger.info(f"  Total Undecided Predictions: {undecided_test_count} ({undecided_test_percentage:.2%})")
        logger.info(f"  Total Samples Evaluated: {total_test_samples}")
        
        # Visualization and analysis
        if visualize:
            self._visualize_undecided_predictions(undecided_predictions_details, num_visualize_each)
            self._visualize_misclassified_samples(misclassified_false_positives, misclassified_false_negatives, num_visualize_each)
            self._plot_probability_distribution(all_probs_test, all_labels_test, output_dir)
        
        # Prepare results dictionary
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
                "total_samples": total_test_samples
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
        
        uncertain_predictions = []
        misclassified_samples = []
        
        # Map filenames to test data
        test_data_map = {test_data[i][0].stem: (test_data[i][0], test_data[i][1]) for i in range(len(test_data))}
        
        with torch.no_grad():
            for inputs, labels, filenames_batch in tqdm(test_loader, desc="Evaluating Test Set"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
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
                    
                    # Check if prediction is uncertain based on thresholds
                    if confidence < self.uncertainty_threshold or max_prob <= self.confidence_threshold_low:
                        uncertain_predictions.append({
                            "img_path": original_img_path,
                            "true_label": true_label,
                            "predicted_label": pred,
                            "probabilities": prob_array.tolist(),
                            "confidence": confidence
                        })
                    elif pred != true_label:  # Misclassified
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
        
        # Log results
        logger.info(f"\nFinal Test Set Metrics (Ternary Classification):")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Precision (Macro): {test_precision_macro:.4f}")
        logger.info(f"  Recall (Macro): {test_recall_macro:.4f}")
        logger.info(f"  F1 Score (Macro): {test_f1_macro:.4f}")
        logger.info(f"  Precision (Weighted): {test_precision_weighted:.4f}")
        logger.info(f"  Recall (Weighted): {test_recall_weighted:.4f}")
        logger.info(f"  F1 Score (Weighted): {test_f1_weighted:.4f}")
        
        logger.info(f"\nUncertain predictions: {len(uncertain_predictions)}")
        logger.info(f"Misclassified samples: {len(misclassified_samples)}")
        
        # Visualization
        if visualize:
            self._visualize_ternary_analysis(uncertain_predictions, misclassified_samples, num_visualize_each)
        
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
            "detailed_analysis": {
                "uncertain_predictions": uncertain_predictions,
                "misclassified_samples": misclassified_samples
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
            visualize_cell(
                detail['img_path'],
                detail['true_label'],
                predicted_label="uncertain",
                probability=detail['probability'],
                classification_mode=self.config.classification_mode
            )
        
        logger.info(f"\n--- {num_to_show_each} Least Undecided Predictions (closest to thresholds) ---")
        start_index = max(0, len(undecided_details) - num_to_show_each)
        for detail in undecided_details[start_index:]:
            logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                       f"Pred: uncertain (Prob: {detail['probability']:.4f})")
            visualize_cell(
                detail['img_path'],
                detail['true_label'],
                predicted_label="uncertain",
                probability=detail['probability'],
                classification_mode=self.config.classification_mode
            )
    
    def _visualize_misclassified_samples(self, false_positives: List[Dict], false_negatives: List[Dict], num_to_show: int):
        """Visualize misclassified samples for binary classification."""
        logger.info("\n--- Visualizing Misclassified Cells (False Positives and False Negatives) ---")
        
        num_fp_to_show = min(num_to_show, len(false_positives))
        if num_fp_to_show > 0:
            logger.info(f"\n--- {num_fp_to_show} False Positives (Predicted Cancerous, True Non-Cancerous) ---")
            for detail in false_positives[:num_fp_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {detail['true_label']}, "
                           f"Pred: {detail['predicted_label']} (Prob: {detail['probability']:.4f})")
                visualize_cell(
                    detail['img_path'],
                    detail['true_label'],
                    predicted_label=detail['predicted_label'],
                    probability=detail['probability'],
                    classification_mode=self.config.classification_mode
                )
        else:
            logger.info("\nNo False Negatives found among decided predictions.")
    
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
                visualize_cell(
                    detail['img_path'],
                    detail['true_label'],
                    predicted_label="uncertain",
                    probability=detail['probabilities'],
                    classification_mode=self.config.classification_mode
                )
        else:
            logger.info("\nNo uncertain predictions found.")
        
        # Visualize misclassified samples
        num_misclassified_to_show = min(num_to_show, len(misclassified_samples))
        if num_misclassified_to_show > 0:
            logger.info(f"\n--- {num_misclassified_to_show} Misclassified Samples ---")
            for detail in misclassified_samples[:num_misclassified_to_show]:
                logger.info(f"  Image: {detail['img_path'].name}, True: {class_names[detail['true_label']]}, "
                           f"Pred: {class_names[detail['predicted_label']]}, Confidence: {detail['confidence']:.4f}")
                visualize_cell(
                    detail['img_path'],
                    detail['true_label'],
                    predicted_label=class_names[detail['predicted_label']],
                    probability=detail['probabilities'],
                    classification_mode=self.config.classification_mode
                )
        else:
            logger.info("\nNo misclassified samples found.")
    
    def _plot_probability_distribution(self, all_probs: List[float], all_labels: List[int], output_dir: Path):
        """Plot probability distribution for binary classification."""
        logger.info("\n--- Probability Distribution and Thresholds ---")
        
        if len(all_probs) == 0:
            logger.warning("No probabilities to plot for the test set.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by true label
        probs_positive = [p for i, p in enumerate(all_probs) if all_labels[i] == 1]
        probs_negative = [p for i, p in enumerate(all_probs) if all_labels[i] == 0]
        
        plt.hist(probs_negative, bins=30, alpha=0.6, label='True Negative Probabilities', 
                color='skyblue', density=True)
        plt.hist(probs_positive, bins=30, alpha=0.6, label='True Positive Probabilities', 
                color='lightcoral', density=True)
        
        # Plot thresholds
        plt.axvline(x=self.config.confidence_threshold_low, color='green', linestyle='--', 
                   label=f'Low Threshold ({self.config.confidence_threshold_low:.2f})')
        plt.axvline(x=0.5, color='gray', linestyle=':', label='Standard 0.5 Threshold')
        plt.axvline(x=self.config.confidence_threshold_high, color='red', linestyle='--', 
                   label=f'High Threshold ({self.config.confidence_threshold_high:.2f})')
        
        plt.title('Distribution of Predicted Probabilities by True Class')
        plt.xlabel('Predicted Probability of being Cancerous')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "probability_distribution.png", dpi=300, bbox_inches='tight')
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
    
    def run_sample_inference(self, test_data: List[Tuple[Path, int]], num_single: int = 5, 
                           num_batch: int = 5, visualize: bool = True) -> Dict:
        """
        Run inference on sample test data (both single and batch).
        
        Args:
            test_data: List of (image_path, label) tuples
            num_single: Number of samples for single inference
            num_batch: Number of samples for batch inference
            visualize: Whether to visualize results
            
        Returns:
            Dict containing inference results
        """
        logger.info("--- Running Sample Inference ---")
        
        # Single image inference
        logger.info("Classifying individual test samples:")
        single_results = []
        
        samples_to_classify_single = test_data[:min(num_single, len(test_data))]
        for img_path, true_label in samples_to_classify_single:
            result = self.classify_single_image(img_path)
            single_results.append({
                "img_path": img_path,
                "true_label": true_label,
                "result": result
            })
            
            if self.config.classification_mode == "binary":
                logger.info(f"  Image: {img_path.name}, True Label: {true_label}, "
                           f"Predicted: {result['prediction']} (Prob: {result['probability']:.4f})")
            else:
                logger.info(f"  Image: {img_path.name}, True Label: {true_label}, "
                           f"Predicted: {result['prediction']} (Confidence: {result['confidence']:.4f})")
            
            # Visualize individual results
            if visualize:
                if self.config.classification_mode == "binary":
                    visualize_cell(
                        img_path, 
                        true_label, 
                        predicted_label=result['prediction'], 
                        probability=result['probability'],
                        classification_mode=self.config.classification_mode
                    )
                else:
                    visualize_cell(
                        img_path, 
                        true_label, 
                        predicted_label=result['prediction'], 
                        probability=result['probabilities'],
                        classification_mode=self.config.classification_mode
                    )
        
        # Batch inference
        batch_results = []
        if len(test_data) > num_single:
            samples_to_classify_batch_paths = [sample[0] for sample in test_data[num_single:min(num_single + num_batch, len(test_data))]]
            samples_to_classify_batch_labels = [sample[1] for sample in test_data[num_single:min(num_single + num_batch, len(test_data))]]
            
            if samples_to_classify_batch_paths:
                logger.info("\nClassifying a batch of test samples:")
                batch_classification_results = self.classify_batch(samples_to_classify_batch_paths)
                
                for i, result in enumerate(batch_classification_results):
                    img_path = samples_to_classify_batch_paths[i]
                    true_label = samples_to_classify_batch_labels[i]
                    
                    batch_results.append({
                        "img_path": img_path,
                        "true_label": true_label,
                        "result": result
                    })
                    
                    if self.config.classification_mode == "binary":
                        logger.info(f"  Image: {img_path.name}, True Label: {true_label}, "
                                   f"Predicted: {result['prediction']} (Prob: {result['probability']:.4f})")
                    else:
                        logger.info(f"  Image: {img_path.name}, True Label: {true_label}, "
                                   f"Predicted: {result['prediction']} (Confidence: {result['confidence']:.4f})")
                    
                    # Visualize batch results
                    if visualize:
                        if self.config.classification_mode == "binary":
                            visualize_cell(
                                img_path, 
                                true_label, 
                                predicted_label=result['prediction'], 
                                probability=result['probability'],
                                classification_mode=self.config.classification_mode
                            )
                        else:
                            visualize_cell(
                                img_path, 
                                true_label, 
                                predicted_label=result['prediction'], 
                                probability=result['probabilities'],
                                classification_mode=self.config.classification_mode
                            )
        
        return {
            "single_inference": single_results,
            "batch_inference": batch_results
        }

    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Returns information about the loaded model and configuration.
        
        Returns:
            Dict[str, Union[str, int, float]]: Model information.
        """
        return {
            "classification_mode": self.config.classification_mode,
            "model_name": self.config.model_name,
            "num_classes": self.config.num_classes,
            "image_size": self.config.image_size,
            "confidence_threshold_high": self.confidence_threshold_high,
            "confidence_threshold_low": self.confidence_threshold_low,
            "uncertainty_threshold": self.uncertainty_threshold,
            "class_names": self.config.class_names,
            "device": str(self.device)
        }