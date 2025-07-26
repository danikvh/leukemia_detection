"""Training visualization utilities."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path


class TrainingVisualizer:
    """Handles visualization during training."""
    
    def __init__(self, output_dir: str, save_enabled: bool = True):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations.
            save_enabled: Whether to save visualizations to disk.
        """
        self.output_dir = Path(output_dir)
        self.save_enabled = save_enabled
        if save_enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_stage1_predictions(self, 
                                   image_tensor: torch.Tensor,
                                   pred_boxes: torch.Tensor,
                                   gt_boxes: torch.Tensor,
                                   epoch: int,
                                   batch_idx: int = 0,
                                   fold: int = 0,
                                   phase: str = "train") -> None:
        """
        Visualize Stage 1 (DETR) predictions.
        
        Args:
            image_tensor: Input image tensor (C, H, W).
            pred_boxes: Predicted boxes in normalized cxcywh format.
            gt_boxes: Ground truth boxes in normalized cxcywh format.
            epoch: Current epoch.
            batch_idx: Batch index.
            fold: Fold number.
            phase: Training phase (train/val).
        """
        if not self.save_enabled:
            return
            
        # Convert tensor to numpy and adjust format
        if image_tensor.ndim == 3:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor.cpu().numpy()
            
        # Ensure image is in proper format for display
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)
        
        height, width = image_np.shape[:2]
        
        # Plot ground truth boxes (green)
        for box in gt_boxes.cpu().numpy():
            cx, cy, w, h = box
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            w *= width
            h *= height
            rect = patches.Rectangle((x, y), w, h, linewidth=2.0, 
                                   edgecolor='lime', facecolor='none', label='GT')
            ax.add_patch(rect)
        
        # Plot predicted boxes (red)
        for box in pred_boxes.cpu().numpy():
            cx, cy, w, h = box
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            w *= width
            h *= height
            rect = patches.Rectangle((x, y), w, h, linewidth=2.0, 
                                   edgecolor='red', facecolor='none', label='Pred')
            ax.add_patch(rect)
        
        ax.set_title(f"Stage 1 - Epoch {epoch+1} - {phase.title()}")
        ax.axis('off')
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        save_path = self.output_dir / f"fold_{fold+1}" / phase / "stage1_boxes"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"epoch_{epoch+1}_batch_{batch_idx}.png", 
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    def visualize_stage2_masks(self,
                             original_image: torch.Tensor,
                             gt_prompt_boxes: torch.Tensor,
                             predicted_masks: np.ndarray,
                             gt_masks: np.ndarray,
                             epoch: int,
                             batch_idx: int = 0,
                             image_idx: int = 0,
                             fold: int = 0,
                             phase: str = "train") -> None:
        """
        Visualize Stage 2 (SAM) mask predictions.
        
        Args:
            original_image: Original image tensor (C, H, W).
            gt_prompt_boxes: Ground truth prompt boxes.
            predicted_masks: Predicted instance masks.
            gt_masks: Ground truth instance masks.
            epoch: Current epoch.
            batch_idx: Batch index.
            image_idx: Image index within batch.
            fold: Fold number.
            phase: Training phase.
        """
        if not self.save_enabled:
            return
            
        try:
            pil_img = TF.to_pil_image(original_image.cpu())
        except Exception:
            # Fallback for problematic tensors
            pil_img = Image.new('RGB', (original_image.shape[2], original_image.shape[1]), 
                              color='grey')
        
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image with prompt boxes
        axs[0].imshow(pil_img)
        axs[0].set_title("Image + GT Prompts")
        axs[0].axis('off')
        
        if gt_prompt_boxes is not None and gt_prompt_boxes.numel() > 0:
            for box in gt_prompt_boxes.cpu().numpy():
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       linewidth=1, edgecolor='cyan', facecolor='none')
                axs[0].add_patch(rect)
        
        # Ground truth masks
        axs[1].imshow(gt_masks, cmap='nipy_spectral', vmin=0)
        axs[1].set_title("Ground Truth Masks")
        axs[1].axis('off')
        
        # Predicted masks
        axs[2].imshow(predicted_masks, cmap='nipy_spectral', vmin=0)
        axs[2].set_title("Predicted Masks")
        axs[2].axis('off')
        
        plt.suptitle(f"Stage 2 - E{epoch+1}_B{batch_idx}_I{image_idx}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = self.output_dir / f"fold_{fold+1}" / phase / "stage2_masks"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"epoch_{epoch+1}_batch_{batch_idx}_img_{image_idx}.png",
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: Optional[List[float]] = None,
                           title: str = "Training Loss",
                           save_name: str = "training_curve.png") -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: List of training losses per epoch.
            val_losses: List of validation losses per epoch.
            title: Plot title.
            save_name: Filename to save the plot.
        """
        if not self.save_enabled or not train_losses:
            return
            
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses and len(val_losses) == len(train_losses):
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / save_name, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_loss_components(self,
                           loss_components: Dict[str, List[float]],
                           title: str = "Loss Components",
                           save_name: str = "loss_components.png") -> None:
        """
        Plot individual loss components over time.
        
        Args:
            loss_components: Dictionary mapping loss names to values over epochs.
            title: Plot title.
            save_name: Filename to save the plot.
        """
        if not self.save_enabled or not loss_components:
            return
            
        plt.figure(figsize=(12, 8))
        
        for loss_name, values in loss_components.items():
            if values:
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=loss_name, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often helps with multiple loss components
        
        plt.savefig(self.output_dir / save_name, bbox_inches='tight', dpi=150)
        plt.close()
    
    def visualize_single_mask(self,
                            mask: torch.Tensor,
                            title: str = "Mask",
                            save_path: Optional[str] = None) -> None:
        """
        Visualize a single mask tensor.
        
        Args:
            mask: Mask tensor to visualize.
            title: Title for the visualization.
            save_path: Optional path to save the visualization.
        """
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask
            
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_np, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        
        if save_path and self.save_enabled:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if not save_path:
            plt.show()
        else:
            plt.close()


def create_training_visualizer(output_dir: str, 
                             fold: int = 0,
                             save_enabled: bool = True) -> TrainingVisualizer:
    """
    Factory function to create a training visualizer.
    
    Args:
        output_dir: Base output directory.
        fold: Fold number for k-fold training.
        save_enabled: Whether to save visualizations.
        
    Returns:
        TrainingVisualizer instance.
    """
    viz_dir = os.path.join(output_dir, f"fold_{fold+1}", "visualizations")
    return TrainingVisualizer(viz_dir, save_enabled)


# Backward compatibility functions
def visualize_predictions(image_tensor: torch.Tensor, 
                        pred_boxes: torch.Tensor, 
                        gt_boxes: torch.Tensor, 
                        save_path: str,
                        max_images: int = 4) -> None:
    """
    Legacy function for visualizing predictions.
    Maintained for backward compatibility.
    """
    visualizer = TrainingVisualizer(os.path.dirname(save_path))
    visualizer.visualize_stage1_predictions(
        image_tensor, pred_boxes, gt_boxes, 
        epoch=0, batch_idx=0, fold=0
    )


def display_image(image_tensor: torch.Tensor, 
                 save_path: Optional[str] = None,
                 title: str = "Image") -> None:
    """
    Display or save a single image tensor.
    
    Args:
        image_tensor: Image tensor to display.
        save_path: Optional path to save the image.
        title: Title for the image.
    """
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 4:  # Batch dimension
            image_tensor = image_tensor.squeeze(0)
        
        # Convert to numpy
        if image_tensor.dim() == 3 and image_tensor.shape[0] in [1, 3]:
            # CHW format
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already HW or HWC
            image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    # Normalize to 0-1 if needed
    if image_np.max() > 1.0:
        image_np = image_np / image_np.max()
    
    plt.figure(figsize=(8, 8))
    if len(image_np.shape) == 2:  # Grayscale
        plt.imshow(image_np, cmap='gray')
    else:
        plt.imshow(image_np)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()