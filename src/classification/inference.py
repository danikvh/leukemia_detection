import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
import logging

from classification.models import get_classification_model
from classification.transforms import get_classification_transforms
from classification.config import ClassificationConfig

logger = logging.getLogger(__name__)

class CellClassifier:
    """
    Handles inference for the cell classification model.
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
            num_classes=1, # Always 1 for binary classification with BCEWithLogitsLoss
            input_channels=3, # Assuming RGB input
            input_image_size=config.image_size # For CustomCellClassifier
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        logger.info(f"Classification model loaded from {model_path} on device: {self.device}")

        self.transform = get_classification_transforms(
            image_size=config.image_size,
            mean=tuple(config.normalize_mean),
            std=tuple(config.normalize_std),
            is_train=False # Always use validation/inference transforms
        )
        
        self.confidence_threshold_high = config.confidence_threshold_high
        self.confidence_threshold_low = config.confidence_threshold_low
        
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

    def classify_single_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Classifies a single cell image.
        
        Args:
            img_input (Union[str, Path, np.ndarray, Image.Image]): Path to the image file,
                                                                   or a loaded NumPy array/PIL Image.
                                                                   
        Returns:
            Dict[str, Union[str, float]]: A dictionary with 'prediction' (str) and 'probability' (float).
                                          Prediction can be 'cancerous', 'non-cancerous', or 'uncertain'.
        """
        img_tensor = self._preprocess_image(img_input).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probability = torch.sigmoid(logits).item() # Get scalar probability

        if probability >= self.confidence_threshold_high:
            prediction = "cancerous"
        elif probability <= self.confidence_threshold_low:
            prediction = "non-cancerous"
        else:
            prediction = "uncertain"
        
        return {"prediction": prediction, "probability": probability}

    def classify_batch(self, img_inputs: List[Union[str, Path, np.ndarray, Image.Image]]) -> List[Dict[str, Union[str, float]]]:
        """
        Classifies a batch of cell images.
        
        Args:
            img_inputs (List[Union[str, Path, np.ndarray, Image.Image]]): List of image inputs.
                                                                           
        Returns:
            List[Dict[str, Union[str, float]]]: A list of classification results for each image.
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
            logits = self.model(batch_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

        results = []
        for prob in probabilities:
            prediction = ""
            if prob >= self.confidence_threshold_high:
                prediction = "cancerous"
            elif prob <= self.confidence_threshold_low:
                prediction = "non-cancerous"
            else:
                prediction = "uncertain"
            results.append({"prediction": prediction, "probability": prob})
            
        return results