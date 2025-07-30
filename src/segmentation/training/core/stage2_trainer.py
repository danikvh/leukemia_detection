"""
Stage 2 trainer for CellSAM (SAM neck fine-tuning).
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from cellSAM.AnchorDETR.util.box_ops import masks_to_boxes

from segmentation.training.core.base_trainer import BaseTrainer
from segmentation.training.config.stage2_config import Stage2Config
from segmentation.training.losses.stage2_losses import CombinedSegmentationLoss
from segmentation.training.utils.online_hard_example_mining import OnlineHardExampleMining


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2: SAM neck fine-tuning."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Stage2Config,
        device: torch.device,
        output_dir: str,
        fold: int = 0
    ):
        super().__init__(model, config, device, output_dir, fold)
        self.stage2_config = config
        self.device = device
        self.loss_manager = CombinedSegmentationLoss(config)
        self.epochs = self.stage2_config.epochs_s2
        
        # OHEM if enabled
        self.ohem = None
        if config.online_hard_negative_mining:
            self.ohem = OnlineHardExampleMining(
                fraction=config.ohem_fraction,
                hard_weight=config.ohem_hard_weight,
                min_hard_weight=config.ohem_hard_weight_min,
                max_hard_weight=config.ohem_hard_weight_max,
                weighted=config.online_hard_negative_mining_weighted
            )
    
    def setup_training_components(self) -> None:
        """Setup optimizer, scheduler, and criterion for Stage 2."""
        # Freeze all parameters first
        self.freeze_parameters(self.model)
        
        # Unfreeze only the neck
        neck = self.model.model.image_encoder.neck
        self.unfreeze_parameters(neck)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            neck.parameters(),
            lr=self.stage2_config.learning_rate,
            weight_decay=self.stage2_config.weight_decay
        )
        
        # Setup scheduler (optional)
        if self.stage2_config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        
        self.logger.info("Stage 2 setup complete - Training SAM neck")
    
    def forward_pass(self, batch: Tuple) -> Tuple[Dict, torch.Tensor]:
        """Perform forward pass for Stage 2."""
        original_images, instance_masks, filenames = batch
        
        # Move to device
        instance_masks = instance_masks.to(self.device)
        
        # Preprocess images for SAM
        list_of_images = [img for img in original_images]
        preprocessed_images, batch_paddings = self.model.sam_bbox_preprocessing(
            list_of_images, device=self.device
        )
        
        # Get image embeddings (through ViT + neck)
        batch_embeddings = self.model.model.image_encoder(preprocessed_images)
        
        return {
            'original_images': original_images,
            'instance_masks': instance_masks,
            'batch_embeddings': batch_embeddings,
            'batch_paddings': batch_paddings,
            'filenames': filenames
        }, instance_masks
    
    def compute_loss(
        self,
        outputs: Dict,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage 2 losses for all images in batch."""
        batch_losses = {
            'focal_loss': 0.0,
            'dice_loss': 0.0,
            'boundary_loss': 0.0,
            'total_loss': 0.0
        }
        
        original_images = outputs['original_images']
        instance_masks = outputs['instance_masks']
        batch_embeddings = outputs['batch_embeddings']
        batch_paddings = outputs['batch_paddings']
        
        total_prompts = 0
        all_prompt_losses = []
        
        # Process each image in the batch
        for b_idx in range(len(original_images)):
            image_losses, num_prompts, prompt_losses = self._process_single_image(
                original_images[b_idx],
                instance_masks[b_idx],
                batch_embeddings[b_idx].unsqueeze(0),
                batch_paddings[b_idx],
                b_idx
            )
            
            # Accumulate losses
            for key in batch_losses:
                batch_losses[key] += image_losses[key]
            
            total_prompts += num_prompts
            all_prompt_losses.extend(prompt_losses)
        
        # Apply OHEM if enabled
        if self.ohem is not None and all_prompt_losses:
            batch_losses = self.ohem.apply_mining(all_prompt_losses, batch_losses)
        
        # Average over batch
        if total_prompts > 0:
            for key in batch_losses:
                if isinstance(batch_losses[key], torch.Tensor):
                    batch_losses[key] = batch_losses[key] / len(original_images)
                else:
                    batch_losses[key] = torch.tensor(
                        batch_losses[key] / len(original_images), 
                        device=self.device
                    )
        
        return batch_losses
    
    def _process_single_image(
        self,
        original_image: torch.Tensor,
        instance_mask: torch.Tensor,
        image_embedding: torch.Tensor,
        padding: Tuple[int, int],
        image_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], int, List[Dict]]:
        """Process a single image and return losses."""
        current_mask = instance_mask.squeeze()
        original_h, original_w = original_image.shape[-2:]
        
        # Get unique instance IDs
        unique_ids = torch.unique(current_mask.int())
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        
        image_losses = {
            'focal_loss': torch.tensor(0.0, device=self.device),
            'dice_loss': torch.tensor(0.0, device=self.device),
            'boundary_loss': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device)
        }
        
        prompt_losses = []
        num_prompts = 0
        
        # Handle empty images (background only)
        if len(unique_ids) == 0:
            unique_ids = [-1]  # Synthetic ID for background-only case
        
        # Process each instance
        for inst_id in unique_ids:
            losses, valid = self._process_single_instance(
                original_image, current_mask, image_embedding, padding,
                inst_id, original_h, original_w, image_idx, num_prompts
            )
            
            if valid:
                prompt_losses.append(losses)
                num_prompts += 1

                # Accumulate image losses
                for key in image_losses:
                    if key in losses:
                        image_losses[key] += losses[key]
        
        return image_losses, num_prompts, prompt_losses
    
    def _process_single_instance(
        self,
        original_image: torch.Tensor,
        current_mask: torch.Tensor,
        image_embedding: torch.Tensor,
        padding: Tuple[int, int],
        inst_id: int,
        original_h: int,
        original_w: int,
        image_idx: int,
        prompt_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Process a single instance and compute its losses."""
        
        # Handle background-only case
        if inst_id == -1:
            gt_mask = torch.zeros((original_h, original_w), dtype=torch.float32)
            gt_box = torch.tensor([[0.0, 0.0, original_w, original_h]])
        else:
            # Get individual GT mask
            gt_mask = (current_mask.int() == inst_id).float()
            if not torch.any(gt_mask):
                return {}, False
            
            # Get bounding box from mask
            gt_box = masks_to_boxes(gt_mask.unsqueeze(0), self.device)
            if gt_box.numel() == 0 or torch.any(torch.isnan(gt_box)):
                return {}, False
        
        # Move GT mask to device
        gt_mask_gpu = gt_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Scale box for SAM input
        scaled_box = self._scale_box_for_sam(
            gt_box, original_h, original_w, padding
        )
        
        # Get SAM prediction
        predicted_mask = self._get_sam_prediction(
            image_embedding, scaled_box, original_h, original_w
        )
        
        # Compute losses
        losses = self.loss_manager.forward(
            predicted_mask, gt_mask_gpu
        )

        for k in list(losses.keys()):
            if k != "total_loss":
                losses[k] = losses[k].detach()
        
        return losses, True
    
    def _scale_box_for_sam(
        self,
        box_xyxy: torch.Tensor,
        original_h: int,
        original_w: int,
        padding: Tuple[int, int]
    ) -> torch.Tensor:
        """Scale bounding box to SAM input coordinates."""
        img_size = self.model.model.image_encoder.img_size
        
        # Calculate dimensions after resize
        img_h_after_resize = img_size - padding[0]
        img_w_after_resize = img_size - padding[1]
        
        # Calculate scaling factors
        scale_h = img_h_after_resize / original_h
        scale_w = img_w_after_resize / original_w
        
        # Scale box coordinates
        scaled_box = box_xyxy.clone()
        scaled_box[:, 0] *= scale_w  # xmin
        scaled_box[:, 2] *= scale_w  # xmax
        scaled_box[:, 1] *= scale_h  # ymin
        scaled_box[:, 3] *= scale_h  # ymax
        
        return scaled_box.to(self.device)
    
    def _get_sam_prediction(
        self,
        image_embedding: torch.Tensor,
        box_prompt: torch.Tensor,
        original_h: float,
        original_w: float
    ) -> torch.Tensor:
        """Get SAM mask prediction for given box prompt."""
        with torch.no_grad():  # freeze everything except neck
            # Encode prompt
            sparse_embeddings, _ = self.model.model.prompt_encoder(
                points=None,
                boxes=box_prompt,
                masks=None
            )
            # Get positional encoding
            image_pe = self.model.model.prompt_encoder.get_dense_pe()
            # Create dummy dense embeddings
            dense_embeddings = torch.zeros_like(image_embedding)

            # Only SAM mask decoder & prompt encoder are frozen
            low_res_mask, _ = self.model.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
        
        # Postprocess to original size (no gradient needed)
        predicted_mask = self.model.model.postprocess_masks(
            low_res_mask,
            input_size=torch.tensor([original_h, original_w], device=self.device),
            original_size=(original_h, original_w)
        )
        
        return predicted_mask
    
    def visualize_predictions(
        self,
        batch: Any,
        outputs: Dict,
        targets: Any,
        epoch: int,
        batch_idx: int,
        stage: str
    ) -> None:
        """Visualize predictions for debugging."""
        if not self.config.debug:
            return

        original_images = outputs['original_images']
        instance_masks = outputs['instance_masks']
        batch_embeddings = outputs['batch_embeddings']
        batch_paddings = outputs['batch_paddings']

        # Visualize each image in the batch
        for img_idx in range(len(original_images)):
            self._visualize_single_image(
                original_images[img_idx],
                instance_masks[img_idx],
                batch_embeddings[img_idx].unsqueeze(0),
                batch_paddings[img_idx],
                img_idx,
                epoch,
                batch_idx,
                stage
            )
    
    def _visualize_single_image(
        self,
        original_image: torch.Tensor,
        instance_mask: torch.Tensor,
        image_embedding: torch.Tensor,
        padding: Tuple[int, int],
        img_idx: int,
        epoch: int,
        batch_idx: int,
        stage: str
    ) -> None:
        """Create visualization for a single image."""
        # Debug tensor information (optional)
        if self.config.debug:
            self.logger.debug(
                f"Image {img_idx}: shape={original_image.shape}, "
                f"min={original_image.min():.3f}, max={original_image.max():.3f}"
            )
        
        current_mask = instance_mask.squeeze()
        original_h, original_w = original_image.shape[-2:]
        
        # Get unique instance IDs
        unique_ids = torch.unique(current_mask.int())
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        
        if len(unique_ids) == 0:
            return  # Skip empty images for visualization
        
        # Collect GT masks, predicted masks, and boxes
        gt_masks = []
        pred_masks = []
        gt_boxes = []
        
        for inst_id in unique_ids:
            # Get GT mask and box
            gt_mask = (current_mask.int() == inst_id).float()
            if not torch.any(gt_mask):
                continue
                
            gt_box = masks_to_boxes(gt_mask.unsqueeze(0), self.device)
            if gt_box.numel() == 0 or torch.any(torch.isnan(gt_box)):
                continue
            
            # Get prediction
            scaled_box = self._scale_box_for_sam(gt_box, original_h, original_w, padding)
            predicted_mask = self._get_sam_prediction(
                image_embedding, scaled_box, original_h, original_w
            )
            
            # Convert to binary mask
            pred_binary = (torch.sigmoid(predicted_mask).squeeze().detach().cpu().numpy() > 0.5)
            
            # Store for visualization
            gt_masks.append(gt_mask.cpu().numpy())
            pred_masks.append(pred_binary)
            gt_boxes.append(gt_box.squeeze().cpu().numpy())
        
        # Create visualization
        if gt_masks:  # Only visualize if we have valid instances
            self._create_combined_visualization(
                original_image.cpu(),
                gt_masks,
                pred_masks,
                gt_boxes,
                img_idx,
                epoch,
                batch_idx,
                stage
            )
    
    def _create_combined_visualization(
        self,
        original_image: torch.Tensor,
        gt_masks: List[np.ndarray],
        pred_masks: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        img_idx: int,
        epoch: int,
        batch_idx: int,
        stage: str
    ) -> None:
        """Create combined visualization for all instances in an image."""
        # Handle batch dimension like Stage 1
        if len(original_image.shape) == 4:  # e.g., [1, 3, H, W]
            if original_image.shape[0] == 1:
                original_image = original_image[0]  # remove batch dimension
            else:
                raise ValueError(f"Expected single image in batch but got {original_image.shape[0]} images")

        # Convert to numpy like Stage 1
        if original_image.ndim == 3:
            image_np = original_image.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected image_tensor shape: {original_image.shape}")
        
        # Ensure proper display format
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image with GT boxes
        axes[0].imshow(image_np)
        axes[0].set_title("Original + GT Boxes")
        axes[0].axis('off')
        
        height, width = image_np.shape[:2]
        
        # Add GT boxes (convert from xyxy to matplotlib format)
        for box in gt_boxes:
            if len(box) >= 4:
                xmin, ymin, xmax, ymax = box[:4]
                rect = patches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                axes[0].add_patch(rect)
        
        # Combined GT masks
        combined_gt = np.zeros_like(gt_masks[0], dtype=np.int16)
        for i, mask in enumerate(gt_masks):
            combined_gt[mask > 0] = i + 1
        
        axes[1].imshow(combined_gt, cmap='nipy_spectral', vmin=0)
        axes[1].set_title(f"Ground Truth Masks ({len(gt_masks)} instances)")
        axes[1].axis('off')
        
        # Combined predicted masks
        combined_pred = np.zeros_like(pred_masks[0], dtype=np.int16)
        for i, mask in enumerate(pred_masks):
            combined_pred[mask > 0] = i + 1
        
        axes[2].imshow(combined_pred, cmap='nipy_spectral', vmin=0)
        axes[2].set_title(f"Predicted Masks ({len(pred_masks)} instances)")
        axes[2].axis('off')
        
        plt.suptitle(f"Stage 2 - Epoch {epoch+1}, Batch {batch_idx}, Image {img_idx} ({stage})")
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(
            self.output_dir, f"fold_{self.fold+1}", "visualizations",
            f"{stage}_stage2_epoch{epoch+1}_batch{batch_idx}_img{img_idx}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _post_step_hook(self, batch_idx: int, batch: Any) -> None:
        """Hook called after each optimization step for Stage 2 specific logic."""
        # Clear GPU cache after each step to manage memory
        torch.cuda.empty_cache()
        if self.gpu_monitor:
            self.gpu_monitor.log_memory("End of epoch after empty cache")