"""
Visualization utilities for medical image analysis.

This module provides comprehensive visualization tools for displaying medical images,
segmentation masks, and analysis results with proper handling of different tensor
formats and display options.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch import Tensor

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class ImageVisualizer:
    """Comprehensive image visualization class for medical imaging."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5), dpi: int = 100):
        """
        Initialize visualizer with default settings.
        
        Args:
            figsize: Default figure size (width, height).
            dpi: Figure resolution.
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = 'viridis'
        
    @staticmethod
    def _validate_inputs(*tensors: Optional[Tensor]) -> None:
        """Validate input tensors."""
        for i, tensor in enumerate(tensors):
            if tensor is not None and not isinstance(tensor, (torch.Tensor, np.ndarray)):
                raise VisualizationError(
                    f"Input {i} must be a torch.Tensor or numpy.ndarray, got {type(tensor)}"
                )
    
    @staticmethod
    def _prepare_tensor(tensor: Optional[Tensor]) -> Optional[np.ndarray]:
        """
        Convert tensor to numpy array with proper format.
        
        Args:
            tensor: Input tensor (torch.Tensor or np.ndarray).
            
        Returns:
            Prepared numpy array or None.
        """
        if tensor is None:
            return None
            
        # Convert to numpy if it's a torch tensor
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor.copy()
        
        # Remove batch dimension if present
        if array.ndim == 4:
            array = array[0]
        elif array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        
        return array
    
    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image array for display.
        
        Args:
            image: Input image array.
            
        Returns:
            Normalized image array.
        """
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW format
            image = image.transpose(1, 2, 0)
        
        # Handle single channel images
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        
        # Normalize to [0, 255] range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)
        
        return image
    
    def display_single_image(
        self,
        image: Tensor,
        title: str = "Image",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        colormap: str = None
    ) -> Optional[plt.Figure]:
        """
        Display a single image.
        
        Args:
            image: Input image tensor.
            title: Image title.
            save_path: Path to save the figure.
            show: Whether to display the figure.
            colormap: Colormap for grayscale images.
            
        Returns:
            Figure object if not showing, None otherwise.
        """
        self._validate_inputs(image)
        
        if image is None:
            logger.error("Cannot display None image")
            return None
        
        image_np = self._prepare_tensor(image)
        image_np = self._normalize_image(image_np)
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        
        if image_np.ndim == 2:  # Grayscale
            cmap = colormap or 'gray'
            ax.imshow(image_np, cmap=cmap)
        else:  # RGB
            ax.imshow(image_np)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def display_image_with_mask(
        self,
        image: Tensor,
        mask: Tensor,
        title: str = "Image with Mask",
        mask_alpha: float = 0.5,
        mask_color: str = 'red',
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Display image with overlay mask.

        Args:
            image: Input image tensor.
            mask: Mask tensor.
            title: Figure title.
            mask_alpha: Mask transparency (0-1).
            mask_color: Mask color.
            save_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Figure object if not showing, None otherwise.
        """
        self._validate_inputs(image, mask)

        if image is None or mask is None:
            logger.error("Cannot display: Image or mask is None")
            return None

        image_np = self._prepare_tensor(image)
        mask_np = self._prepare_tensor(mask)

        image_np = self._normalize_image(image_np)

        # Ensure mask is 2D
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        # Display base image
        if image_np.ndim == 2:
            ax.imshow(image_np, cmap='gray')
        else:
            ax.imshow(image_np)

        # Create colored mask overlay
        # Define a fully transparent color using an RGBA tuple
        transparent_color = (0, 0, 0, 0)

        # You can keep the `mask_color` as a string for the non-transparent part
        # as Matplotlib will convert named colors.
        colors = [transparent_color, mask_color]

        cmap = ListedColormap(colors)
        ax.imshow(mask_np > 0, cmap=cmap, alpha=mask_alpha)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show:
            plt.show()
            return None
        else:
            return fig
    
    def display_comparison(
        self,
        image: Tensor,
        mask: Tensor,
        additional_image: Optional[Tensor] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Display image, mask, and overlay comparison.
        
        Args:
            image: Input image tensor.
            mask: Mask tensor.
            additional_image: Optional third image to display.
            titles: List of titles for subplots.
            save_path: Path to save the figure.
            show: Whether to display the figure.
            
        Returns:
            Figure object if not showing, None otherwise.
        """
        self._validate_inputs(image, mask, additional_image)
        
        if image is None or mask is None:
            logger.error("Cannot display: Image or mask is None")
            return None
        
        # Prepare data
        image_np = self._normalize_image(self._prepare_tensor(image))
        mask_np = self._prepare_tensor(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # Create overlay
        colored_mask = np.zeros_like(image_np)
        if image_np.ndim == 3:
            colored_mask[..., 0] = (mask_np > 0) * 255  # Red channel
        else:
            colored_mask = (mask_np > 0) * 255
        
        overlay = np.where(
            colored_mask > 0,
            (0.5 * image_np + 0.5 * colored_mask),
            image_np
        ).astype(np.uint8)
        
        # Determine number of subplots
        n_plots = 3 if additional_image is None else 4
        
        # Default titles
        if titles is None:
            titles = ["Original Image", "Mask", "Overlay"]
            if additional_image is not None:
                titles.append("Additional")
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), dpi=self.dpi)
        
        if n_plots == 1:
            axes = [axes]
        
        # Display original image
        if image_np.ndim == 2:
            axes[0].imshow(image_np, cmap='gray')
        else:
            axes[0].imshow(image_np)
        axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display mask
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Display overlay
        if overlay.ndim == 2:
            axes[2].imshow(overlay, cmap='gray')
        else:
            axes[2].imshow(overlay)
        axes[2].set_title(titles[2], fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Display additional image if provided
        if additional_image is not None:
            additional_np = self._normalize_image(self._prepare_tensor(additional_image))
            if additional_np.ndim == 2:
                axes[3].imshow(additional_np, cmap='gray')
            else:
                axes[3].imshow(additional_np)
            axes[3].set_title(titles[3], fontsize=12, fontweight='bold')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def display_individual_masks(
        self,
        image: Tensor,
        mask: Tensor,
        max_masks: int = 16,
        cols: int = 4,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Display individual instance masks over the image.
        
        Args:
            image: Input image tensor.
            mask: Instance mask tensor with unique IDs.
            max_masks: Maximum number of masks to display.
            cols: Number of columns in the grid.
            save_path: Path to save the figure.
            show: Whether to display the figure.
            
        Returns:
            Figure object if not showing, None otherwise.
        """
        self._validate_inputs(image, mask)
        
        if image is None or mask is None:
            logger.error("Cannot display: Image or mask is None")
            return None
        
        image_np = self._normalize_image(self._prepare_tensor(image))
        mask_np = self._prepare_tensor(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # Get unique instance IDs (skip background = 0)
        unique_ids = np.unique(mask_np)
        unique_ids = unique_ids[unique_ids != 0]
        
        if len(unique_ids) == 0:
            logger.warning("No instance masks found")
            return None
        
        # Limit number of masks
        unique_ids = unique_ids[:max_masks]
        n_masks = len(unique_ids)
        
        logger.info(f"Displaying {n_masks} individual masks")
        
        # Calculate grid dimensions
        rows = int(np.ceil(n_masks / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=self.dpi)
        
        # Ensure 'axes' is always a 2D array for consistent indexing
        if n_masks == 1:
            axes = np.array([[axes]]) # Wrap single Axes object in a 2D array
        elif rows == 1:
            axes = axes.reshape(1, -1) # Ensure 1D array becomes 2D (1, cols)
        # else: axes is already 2D if rows > 1 and cols > 1 (e.g., 2,2; 3,4)


        # Define the highlight color and its transparency
        # Using a bright, distinct color like 'lime' or 'red'
        highlight_color = 'lime' # You can change this to 'red', 'blue', 'yellow', etc.
        overlay_alpha = 0.5 # Adjust transparency (0.0 to 1.0)

        # Create a colormap for the binary mask: transparent for 0, highlight_color for 1
        # (R, G, B, Alpha) for transparent, then the actual color string/tuple
        colors_for_cmap = [(0, 0, 0, 0), highlight_color]
        cmap_overlay = ListedColormap(colors_for_cmap)

        for idx, obj_id in enumerate(unique_ids):
            row = idx // cols
            col = idx % cols
            current_ax = axes[row, col] # Get the specific subplot axis

            # Display the base image first
            if image_np.ndim == 2:  # Grayscale
                current_ax.imshow(image_np, cmap='gray')
            else:  # RGB
                current_ax.imshow(image_np)

            # Create binary mask for this instance (using boolean for direct use with imshow alpha)
            binary_mask = (mask_np == obj_id)

            # Overlay the binary mask with transparency using the custom colormap
            current_ax.imshow(binary_mask, cmap=cmap_overlay, alpha=overlay_alpha)

            current_ax.set_title(f"Instance {obj_id}", fontsize=10, fontweight='bold')
            current_ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_masks, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def create_segmentation_summary(
        self,
        image: Tensor,
        mask: Tensor,
        predictions: Optional[Dict[str, Any]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create a comprehensive segmentation analysis summary.
        
        Args:
            image: Input image tensor.
            mask: Segmentation mask tensor.
            predictions: Optional dictionary with prediction metrics.
            save_path: Path to save the figure.
            show: Whether to display the figure.
            
        Returns:
            Figure object if not showing, None otherwise.
        """
        self._validate_inputs(image, mask)
        
        if image is None or mask is None:
            logger.error("Cannot create summary: Image or mask is None")
            return None
        
        image_np = self._normalize_image(self._prepare_tensor(image))
        mask_np = self._prepare_tensor(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # Calculate basic statistics
        unique_ids = np.unique(mask_np)
        n_instances = len(unique_ids) - (1 if 0 in unique_ids else 0)
        mask_coverage = np.sum(mask_np > 0) / mask_np.size * 100
        
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        if image_np.ndim == 2:
            ax1.imshow(image_np, cmap='gray')
        else:
            ax1.imshow(image_np)
        ax1.set_title("Original Image", fontweight='bold')
        ax1.axis('off')
        
        # Mask
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask_np, cmap='jet')
        ax2.set_title("Segmentation Mask", fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = image_np.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_np > 0, 0] = 255  # Red overlay
        
        overlay_combined = np.where(
            colored_mask > 0,
            (0.7 * overlay + 0.3 * colored_mask),
            overlay
        ).astype(np.uint8)
        
        ax3.imshow(overlay_combined)
        ax3.set_title("Overlay", fontweight='bold')
        ax3.axis('off')
        
        # Statistics
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        stats_text = f"""
        Segmentation Summary:
        • Number of instances: {n_instances}
        • Mask coverage: {mask_coverage:.2f}%
        • Image size: {image_np.shape[:2]}
        • Unique mask values: {len(unique_ids)}
        """
        
        if predictions:
            stats_text += "\n        Prediction Metrics:\n"
            for key, value in predictions.items():
                stats_text += f"        • {key}: {value}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        if save_path:
            self._save_figure(fig, save_path)
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    @staticmethod
    def _save_figure(fig: plt.Figure, save_path: Union[str, Path]) -> None:
        """Save figure to specified path."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white')
            logger.info(f"Figure saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
        finally:
            plt.close(fig)


# Backward compatibility functions
def display_image(image_tensor: Optional[Tensor], save_path: Optional[str] = None) -> None:
    """Legacy function for backward compatibility."""
    warnings.warn(
        "display_image is deprecated. Use ImageVisualizer.display_single_image() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if image_tensor is None:
        logger.error("Error: Image is None, cannot display.")
        return
    
    visualizer = ImageVisualizer()
    visualizer.display_single_image(
        image_tensor, 
        save_path=save_path,
        show=(save_path is None)
    )


def display_comparison(
    image_tensor: Optional[Tensor], 
    mask_tensor: Optional[Tensor], 
    additional_image_tensor: Optional[Tensor] = None
) -> None:
    """Legacy function for backward compatibility."""
    warnings.warn(
        "display_image_mask_cellpose is deprecated. Use ImageVisualizer.display_comparison() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if image_tensor is None or mask_tensor is None:
        logger.error("Error: Image or Mask is None, cannot display.")
        return
    
    visualizer = ImageVisualizer()
    visualizer.display_comparison(
        image_tensor, 
        mask_tensor, 
        additional_image_tensor
    )


def display_individual_masks(image_tensor: Optional[Tensor], mask_tensor: Optional[Tensor]) -> None:
    """Legacy function for backward compatibility."""
    warnings.warn(
        "display_individual_masks is deprecated. Use ImageVisualizer.display_individual_masks() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if image_tensor is None or mask_tensor is None:
        logger.error("Error: Image or Mask is None, cannot display.")
        return
    
    visualizer = ImageVisualizer()
    visualizer.display_individual_masks(image_tensor, mask_tensor)


# Additional utility functions
def create_color_palette(n_colors: int) -> List[str]:
    """
    Create a color palette for visualization.
    
    Args:
        n_colors: Number of colors needed.
        
    Returns:
        List of color strings.
    """
    import matplotlib.cm as cm
    colormap = cm.get_cmap('tab20')
    return [colormap(i / n_colors) for i in range(n_colors)]


def save_batch_visualization(
    images: List[Tensor],
    masks: List[Tensor],
    save_dir: Union[str, Path],
    prefix: str = "sample"
) -> None:
    """
    Save visualizations for a batch of images and masks.
    
    Args:
        images: List of image tensors.
        masks: List of mask tensors.
        save_dir: Directory to save visualizations.
        prefix: Filename prefix.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ImageVisualizer()
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        save_path = save_dir / f"{prefix}_{i:03d}.png"
        visualizer.display_comparison(
            image, mask, 
            save_path=save_path, 
            show=False
        )
    
    logger.info(f"Saved {len(images)} visualizations to {save_dir}")


def create_grid_visualization(
    images: List[Tensor],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of image tensors.
        titles: Optional list of titles.
        cols: Number of columns in grid.
        save_path: Path to save the figure.
        show: Whether to display the figure.
        
    Returns:
        Figure object if not showing, None otherwise.
    """
    n_images = len(images)
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=100)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    visualizer = ImageVisualizer()
    
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        
        image_np = visualizer._normalize_image(visualizer._prepare_tensor(image))
        
        if image_np.ndim == 2:
            axes[row, col].imshow(image_np, cmap='gray')
        else:
            axes[row, col].imshow(image_np)
        
        title = titles[i] if titles and i < len(titles) else f"Image {i+1}"
        axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        ImageVisualizer._save_figure(fig, save_path)
    
    if show:
        plt.show()
        return None
    else:
        return fig