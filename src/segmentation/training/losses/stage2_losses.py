"""
Stage 2 (SAM segmentation) specific losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_losses import BaseLoss


class DiceLoss(BaseLoss):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0, weight: float = 1.0):
        super().__init__(weight=weight, name="dice")
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_boxes=None, **kwargs):
        """
        Args:
            predictions: Model logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)
            num_boxes: Number of boxes (for normalization)
        """
        inputs = predictions.sigmoid()
        inputs = inputs.flatten(1)
        targets_flattened = targets.flatten(1)
        
        numerator = 2 * (inputs * targets_flattened).sum(1)
        denominator = inputs.sum(-1) + targets_flattened.sum(-1)
        loss = 1 - (numerator + self.smooth) / (denominator + self.smooth)
        
        return loss.sum()


class FocalLoss(BaseLoss):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: float = 1.0):
        super().__init__(weight=weight, name="focal")
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets, num_boxes=None, **kwargs):
        """
        Args:
            predictions: Model logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)
            num_boxes: Number of boxes (for normalization)
        """
        logits_flattened = predictions.flatten(1)
        targets_flattened = targets.flatten(1)
        
        prob = logits_flattened.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            logits_flattened, targets_flattened, reduction="none"
        )
        
        p_t = prob * targets_flattened + (1 - prob) * (1 - targets_flattened)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets_flattened + (1 - self.alpha) * (1 - targets_flattened)
            loss = alpha_t * loss
        
        return loss.mean(1).sum()


class BoundaryLoss(BaseLoss):
    """Boundary-aware loss using Sobel edge detection."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight, name="boundary")
        
        # Sobel X kernel
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[[[-1., 0., 1.],
                                                   [-2., 0., 2.],
                                                   [-1., 0., 1.]]]], dtype=torch.float32)
        self.sobel_x.weight.requires_grad = False
        
        # Sobel Y kernel
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor([[[[-1., -2., -1.],
                                                   [0., 0., 0.],
                                                   [1., 2., 1.]]]], dtype=torch.float32)
        self.sobel_y.weight.requires_grad = False
    
    def forward(self, predictions, targets, num_boxes=None, **kwargs):
        """
        Args:
            predictions: Model logits (B, 1, H, W)
            targets: Ground truth binary masks (B, 1, H, W)
        """
        pred_probs = torch.sigmoid(predictions)
        
        # Move kernels to the same device as input
        self.sobel_x.to(pred_probs.device)
        self.sobel_y.to(pred_probs.device)
        
        # Apply Sobel to predicted probabilities
        pred_edge_x = self.sobel_x(pred_probs)
        pred_edge_y = self.sobel_y(pred_probs)
        pred_edge_magnitude = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        # Apply Sobel to GT masks
        gt_edge_x = self.sobel_x(targets.float())
        gt_edge_y = self.sobel_y(targets.float())
        gt_edge_magnitude = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-6)
        
        # L1 loss between edge magnitudes
        loss = F.l1_loss(pred_edge_magnitude, gt_edge_magnitude, reduction='mean')
        
        return loss


class CombinedSegmentationLoss(BaseLoss):
    """Combined loss for segmentation (Focal + Dice + Boundary)."""
    
    def __init__(self, 
                 focal_weight: float = 1.0,
                 dice_weight: float = 1.0, 
                 boundary_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__(name="combined_segmentation")
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=focal_weight)
        self.dice_loss = DiceLoss(weight=dice_weight)
        self.boundary_loss = BoundaryLoss(weight=boundary_weight)
        
        self.weights = {
            'focal': focal_weight,
            'dice': dice_weight,
            'boundary': boundary_weight
        }
    
    def forward(self, predictions, targets, num_boxes=None, **kwargs):
        """Combined segmentation loss."""
        focal_loss = self.focal_loss(predictions, targets, num_boxes, **kwargs)
        dice_loss = self.dice_loss(predictions, targets, num_boxes, **kwargs)
        boundary_loss = self.boundary_loss(predictions, targets, num_boxes, **kwargs)
        
        total_loss = focal_loss + dice_loss + boundary_loss
        
        return total_loss
    
    def get_loss_components(self, predictions, targets, num_boxes=None, **kwargs):
        """Get individual loss components for logging."""
        with torch.no_grad():
            focal_loss = self.focal_loss(predictions, targets, num_boxes, **kwargs)
            dice_loss = self.dice_loss(predictions, targets, num_boxes, **kwargs)  
            boundary_loss = self.boundary_loss(predictions, targets, num_boxes, **kwargs)
        
        return {
            'focal_loss': focal_loss.item(),
            'dice_loss': dice_loss.item(),
            'boundary_loss': boundary_loss.item()
        }