"""
Refactored inference module with core classification logic separated from evaluation and visualization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
import logging
import json

from classification.models import get_classification_model
from classification.transforms import get_classification_transforms
from classification.config import ClassificationConfig

logger = logging.getLogger(__name__)


class CellClassifier:
    """
    Core cell classification inference handler.
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
        
        # Load model
        self.model = get_classification_model(
            model_name=config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            input_channels=3,  # Assuming RGB input
            input_image_size=config.image_size,
            classification_mode=config.classification_mode
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Classification model loaded from {model_path} on device: {self.device}")
        logger.info(f"Classification mode: {config.classification_mode}")

        # Setup transforms
        self.transform = get_classification_transforms(
            image_size=config.image_size,
            mean=tuple(config.normalize_mean),
            std=tuple(config.normalize_std),
            is_train=False
        )
        
        # Set thresholds
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
                img = Image.open(img_path).convert("RGB")
                
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
            
        return self.transform(img).unsqueeze(0)  # Add batch dimension

    def classify_single_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Union[str, float, List[float]]]:
        """
        Classifies a single cell image.
        
        Args:
            img_input: Path to the image file, or a loaded NumPy array/PIL Image.
                                                                   
        Returns:
            Dict with classification results.
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
            img_inputs: List of image inputs.
                                                                           
        Returns:
            List of classification results for each image.
        """
        if not img_inputs:
            return []

        processed_tensors = [self._preprocess_image(img).squeeze(0) for img in img_inputs]
        
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
            
            # Visualize individual results if requested
            if visualize:
                try:
                    from classification.utils import visualize_cell
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
                except ImportError:
                    logger.warning("Visualization utilities not available")
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
        
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
                    
                    # Visualize batch results if requested
                    if visualize:
                        try:
                            from classification.utils import visualize_cell
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
                        except ImportError:
                            logger.warning("Visualization utilities not available")
                        except Exception as e:
                            logger.warning(f"Visualization failed: {e}")
        
        return {
            "single_inference": single_results,
            "batch_inference": batch_results
        }

    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Returns information about the loaded model and configuration.
        
        Returns:
            Dict with model information.
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