"""
Stage 1 (DETR) specific losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cellSAM.AnchorDETR.models.anchor_detr import SetCriterion
from cellSAM.AnchorDETR.models.matcher import HungarianMatcher
from .base_losses import BaseLoss


class DETRCombinedLoss(BaseLoss):
    """Combined DETR loss using SetCriterion."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 ce_weight: float = 1.0,
                 bbox_weight: float = 5.0,
                 giou_weight: float = 2.0,
                 focal_alpha: float = 0.25):
        super().__init__()
        
        self.num_classes = num_classes

        self.matcher = HungarianMatcher()
        
        weight_dict = {
            'loss_ce': ce_weight,
            'loss_bbox': bbox_weight,
            'loss_giou': giou_weight
        }
        
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            losses=['labels', 'boxes'],
            focal_alpha=focal_alpha
        )
        
        self.weight_dict = weight_dict
    
    def forward(self, predictions, targets, **kwargs):
        """
        Args:
            predictions: Model outputs from DETR head
            targets: List of target dictionaries with 'labels' and 'boxes'
        """
        loss_dict = self.criterion(predictions, targets)
        
        # Calculate weighted total loss
        total_loss = sum(
            loss_dict[k] * self.weight_dict[k] 
            for k in loss_dict.keys() 
            if k in self.weight_dict
        )
        
        return total_loss
    
    def get_loss_components(self, predictions, targets):
        """Get individual loss components for logging."""
        loss_dict = self.criterion(predictions, targets)
        return {
            'ce_loss': loss_dict['loss_ce'].item(),
            'bbox_loss': loss_dict['loss_bbox'].item(),
            'giou_loss': loss_dict['loss_giou'].item()
        }


class ClassificationLoss(BaseLoss):
    """Cross-entropy loss for classification."""
    
    def __init__(self, weight: float = 1.0, focal_alpha: float = 0.25):
        super().__init__(weight=weight)
        self.focal_alpha = focal_alpha
    
    def forward(self, predictions, targets, **kwargs):
        # Extract class predictions
        pred_logits = predictions['pred_logits']  # (batch_size, num_queries, num_classes)
        
        # Flatten for loss calculation
        pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])
        target_labels = torch.cat([t['labels'] for t in targets])
        
        # Cross-entropy loss
        loss = F.cross_entropy(pred_logits_flat, target_labels, reduction='mean')
        
        return loss


class BBoxRegressionLoss(BaseLoss):
    """Bounding box regression loss (L1 + GIoU)."""
    
    def __init__(self, l1_weight: float = 1.0, giou_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
    
    def forward(self, predictions, targets, **kwargs):
        from cellSAM.AnchorDETR.util.box_ops import box_cxcywh_to_xyxy
        from torchvision.ops import generalized_box_iou
        
        pred_boxes = predictions['pred_boxes']  # (batch_size, num_queries, 4)
        target_boxes = torch.cat([t['boxes'] for t in targets])
        
        # L1 loss
        l1_loss = F.l1_loss(pred_boxes.flatten(0, 1), target_boxes, reduction='mean')
        
        # GIoU loss
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes.flatten(0, 1))
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        giou_matrix = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        giou_loss = 1 - torch.diag(giou_matrix).mean()
        
        total_loss = self.l1_weight * l1_loss + self.giou_weight * giou_loss
        
        return total_loss