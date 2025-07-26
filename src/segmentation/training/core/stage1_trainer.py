"""
Stage 1 trainer for CellSAM (CellFinder head and backbone training).
"""

from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from cellSAM.AnchorDETR.util.misc import nested_tensor_from_tensor_list, NestedTensor
from cellSAM.AnchorDETR.util.box_ops import masks_to_boxes, box_xyxy_to_cxcywh
from cellSAM.AnchorDETR.models.anchor_detr import SetCriterion
from cellSAM.AnchorDETR.models.matcher import HungarianMatcher

from segmentation.training.core.base_trainer import BaseTrainer
from segmentation.training.config.stage1_config import Stage1Config
from segmentation.training.losses.loss_factory import LossFactory


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1: CellFinder head and backbone training."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Stage1Config,
        device: torch.device,
        output_dir: str,
        fold: int = 0
    ):
        super().__init__(model, config, device, output_dir, fold)
        self.stage1_config = config
        self.loss_manager = LossFactory().create_stage1_loss(config)
        
        # Stage 1 specific components
        self.matcher = self.loss_manager.matcher
        
    def setup_training_components(self) -> None:
        """Setup optimizer, scheduler, and criterion for Stage 1."""
        # Freeze all parameters first
        self.freeze_parameters(self.model)
        
        # Get components
        transformer = self.model.cellfinder.decode_head.transformer
        backbone = self.model.cellfinder.decode_head.backbone
        image_encoder = self.model.model.image_encoder
        
        # Setup trainable parameters based on strategy
        param_groups = self._setup_parameter_groups(
            transformer, backbone, image_encoder
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.stage1_config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 1.0 if epoch < int(0.7 * self.config.epochs) else 0.9
        )
        
        # Setup criterion
        self.criterion = self.loss_manager.criterion.to(self.device)
        
        self.logger.info(f"Stage 1 setup complete - Strategy: {self.stage1_config.training_strategy}")
    
    def _setup_parameter_groups(
        self,
        transformer: nn.Module,
        backbone: nn.Module,
        image_encoder: nn.Module
    ) -> List[Dict[str, Any]]:
        """Setup parameter groups based on training strategy."""
        # Head parameters
        head_params = (
            list(transformer.class_embed.parameters()) +
            list(transformer.bbox_embed.parameters()) +
            list(transformer.pattern.parameters()) +
            list(transformer.position.parameters()) +
            list(transformer.adapt_pos2d.parameters()) +
            list(transformer.adapt_pos1d.parameters())
        )
        
        # Unfreeze head
        for param in head_params:
            param.requires_grad = True
        
        param_groups = [
            {"params": head_params, "lr": self.stage1_config.lr}
        ]
        
        # Add backbone parameters based on strategy
        strategy = self.stage1_config.training_strategy
        
        if "cellfinder" in strategy:
            self.unfreeze_parameters(backbone)
            param_groups.append({
                "params": backbone.parameters(),
                "lr": self.stage1_config.backbone_lr
            })
            self.logger.info("Training CellFinder backbone")
            
        if "image_encoder" in strategy:
            image_encoder_params = (
                [image_encoder.pos_embed] +
                list(image_encoder.patch_embed.parameters()) +
                list(image_encoder.blocks.parameters())
            )
            for param in image_encoder_params:
                param.requires_grad = True
            param_groups.append({
                "params": image_encoder_params,
                "lr": self.stage1_config.backbone_lr
            })
            self.logger.info("Training image encoder backbone")
        
        return param_groups
    
    def convert_to_targets(
        self,
        images_batch: torch.Tensor,
        instance_masks_batch: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert instance masks to DETR targets format."""
        targets = []
        
        for b_idx in range(images_batch.shape[0]):
            mask = instance_masks_batch[b_idx].squeeze()
            
            # Ensure integer type
            if mask.dtype.is_floating_point:
                mask = mask.round().int()
            else:
                mask = mask.int()
            
            image_height, image_width = mask.shape
            unique_ids = torch.unique(mask)
            unique_ids = unique_ids[unique_ids != 0]  # Exclude background
            
            if len(unique_ids) == 0:
                targets.append({
                    'labels': torch.empty((0,), dtype=torch.long, device=self.device),
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=self.device)
                })
                continue
            
            # Create binary masks for each instance
            binary_masks = []
            for inst_id in unique_ids:
                binary_mask = (mask == inst_id).float()
                if torch.any(binary_mask):
                    binary_masks.append(binary_mask)
            
            if not binary_masks:
                targets.append({
                    'labels': torch.empty((0,), dtype=torch.long, device=self.device),
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=self.device)
                })
                continue
            
            stacked_masks = torch.stack(binary_masks, dim=0)
            N_valid = stacked_masks.shape[0]
            
            # Get bounding boxes
            boxes_xyxy = masks_to_boxes(stacked_masks, self.device)
            if boxes_xyxy.numel() == 0:
                targets.append({
                    'labels': torch.empty((0,), dtype=torch.long, device=self.device),
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=self.device)
                })
                continue
            
            # Convert to cxcywh and normalize
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
            boxes_cxcywh[:, 0] /= image_width   # norm_cx
            boxes_cxcywh[:, 1] /= image_height  # norm_cy
            boxes_cxcywh[:, 2] /= image_width   # norm_w
            boxes_cxcywh[:, 3] /= image_height  # norm_h
            
            # Ensure positive dimensions
            boxes_cxcywh[:, 2:] = torch.clamp(boxes_cxcywh[:, 2:], min=1e-6)
            
            # Create labels (all class 0 for 'cell')
            labels = torch.zeros(N_valid, dtype=torch.long, device=self.device)
            
            targets.append({
                'labels': labels,
                'boxes': boxes_cxcywh
            })
        
        return targets
    
    def forward_pass(self, batch: Tuple) -> Tuple[Dict, List[Dict]]:
        """Perform forward pass for Stage 1."""
        images, masks, filenames = batch
        
        # Preprocess images
        list_of_images = [img.to(self.device) for img in images]
        preprocessed_images, _ = self.model.sam_bbox_preprocessing(
            list_of_images, device=self.device
        )
        preprocessed_images = preprocessed_images.to(self.device)
        
        # Convert masks to targets
        instance_masks = masks.to(self.device)
        targets = self.convert_to_targets(preprocessed_images, instance_masks)
        
        # Create NestedTensor for DETR
        if not isinstance(preprocessed_images, NestedTensor):
            images_nt = nested_tensor_from_tensor_list(preprocessed_images)
        else:
            images_nt = preprocessed_images
        
        # Forward pass through CellFinder
        outputs = self.model.cellfinder.decode_head(images_nt.to(self.device))
        
        return outputs, targets
    

    def compute_loss(
        self,
        outputs: Dict,
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage 1 losses."""
        # Get total loss (already properly weighted by the loss object)
        total_loss = self.criterion(outputs, targets)
        
        # Get individual components for logging
        loss_components = self.criterion.get_loss_components(outputs, targets)
        
        # Return structured dictionary
        result = {'total_loss': total_loss}
        result.update(loss_components)
        
        return result
    
    def visualize_predictions(
        self,
        images: torch.Tensor,
        outputs: Dict,
        targets: List[Dict],
        epoch: int,
        batch_idx: int,
        stage: str
    ) -> None:
        """Visualize predictions for debugging."""
        if not self.config.debug:
            return
        
        # Get matches
        indices = self.matcher.forward(outputs, targets)
        
        for img_idx in range(len(images)):
            image_tensor = images[img_idx].cpu()
            gt_boxes = targets[img_idx]['boxes'].clone().cpu()
            
            # Get matched predictions
            pred_inds, gt_inds = indices[img_idx]
            if len(pred_inds) > 0:
                pred_boxes = outputs['pred_boxes'][img_idx][pred_inds].detach().cpu()
            else:
                pred_boxes = torch.empty(0, 4)
            
            self._save_bbox_visualization(
                image_tensor, pred_boxes, gt_boxes,
                f"{self.output_dir}/fold_{self.fold+1}/visualizations/{stage}_stage1_epoch{epoch+1}_batch{batch_idx}_img{img_idx}.png"
            )
    
    def _save_bbox_visualization(
        self,
        image_tensor: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        save_path: str
    ) -> None:
        """Save bounding box visualization."""
        # Convert to numpy
        if image_tensor.ndim == 3:
            image_np = image_tensor.permute(1, 2, 0).numpy()
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np.astype("uint8"))
        
        height, width = image_np.shape[:2]
        
        # Plot ground truth boxes (green)
        for box in gt_boxes:
            cx, cy, w, h = box
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            w *= width
            h *= height
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2.0, 
                edgecolor='lime', facecolor='none', label='GT'
            )
            ax.add_patch(rect)
        
        # Plot predicted boxes (red)
        for box in pred_boxes:
            cx, cy, w, h = box
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            w *= width
            h *= height
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2.0, 
                edgecolor='red', facecolor='none', label='Pred'
            )
            ax.add_patch(rect)
        
        ax.set_title("Stage 1 Predictions: Green=GT, Red=Pred")
        ax.axis('off')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    
    def perform_weight_transfer(self) -> None:
        """Transfer weights from CellFinder backbone to SAM image encoder."""
        if "transfer" not in self.stage1_config.training_strategy:
            return
        
        self.logger.info("Transferring ViT weights from CellFinder to SAM image encoder")
        
        # Source: CellFinder's ViT
        source_vit = self.model.cellfinder.decode_head.backbone.body
        source_state_dict = source_vit.state_dict()
        
        # Target: SAM's image encoder
        target_encoder = self.model.model.image_encoder
        target_state_dict = target_encoder.state_dict()
        
        # Transfer compatible weights
        transferred_keys = []
        for key in source_state_dict.keys():
            if key in target_state_dict:
                target_state_dict[key] = source_state_dict[key]
                transferred_keys.append(key)
        
        # Load the updated state dict
        target_encoder.load_state_dict(target_state_dict)
        
        self.logger.info(f"Transferred {len(transferred_keys)} parameter groups")
        self.logger.debug(f"Transferred keys: {transferred_keys}")
    
    def finalize_stage(self) -> None:
        """Finalize Stage 1 training."""
        # Perform weight transfer if specified
        self.perform_weight_transfer()
        
        # Load best model
        best_model_path = self.checkpoint_manager.get_best_model_path()
        if os.path.exists(best_model_path):
            self.logger.info(f"Loading best Stage 1 model from {best_model_path}")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        else:
            self.logger.warning("No best Stage 1 model found, using current state")
        
        self.logger.info("Stage 1 training finalized")