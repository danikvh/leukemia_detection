"""
Memory-efficient extensions for cell classification inference.
Provides batch processing with explicit memory management.
"""

import torch
import gc
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from classification.inference import CellClassifier

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