import torch
import numpy as np
import albumentations as A
import cv2 # For albumentations border_mode
from typing import Tuple

def get_cell_augmentations(complex_augmentation: bool = False) -> A.Compose:
    """
    Defines and returns a composition of Albumentations transforms suitable for
    cell instance segmentation, addressing newer API recommendations.
    """
    if complex_augmentation:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                shear=(-5, 5),
                p=0.75,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0
            ),

            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.07,
                approximate=False,
                p=0.4,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                cval=0,
                cval_mask=0
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.6,
            ),
        ])
    else:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])


def augment_image_and_mask(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    augmentation_pipeline: A.Compose
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies data augmentation to an image and its corresponding instance mask.

    Args:
        image_tensor (torch.Tensor): Input image tensor. Expected to be in CHW format,
                                     torch.float, with values in [0, 255].
        mask_tensor (torch.Tensor): Input mask tensor. Expected to be in CHW format
                                    (e.g., 1HW for single channel instance mask),
                                    torch.long, with integer instance labels.
        augmentation_pipeline (A.Compose): An Albumentations composition pipeline.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented image
                                           tensor and augmented mask tensor, in the
                                           same format and device as the input.
    """
    original_device = image_tensor.device
    original_image_dtype = image_tensor.dtype
    original_mask_dtype = mask_tensor.dtype

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    if image_np.max() <= 1.0 and original_image_dtype == torch.float32:
        image_np = (image_np * 255)
    image_np = image_np.astype(np.uint8)

    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_np = mask_tensor.squeeze(0).cpu().numpy()
    elif mask_tensor.ndim == 2:
        mask_np = mask_tensor.cpu().numpy()
    else:
        raise ValueError(
            f"Mask tensor has unexpected shape: {mask_tensor.shape}. "
            "Expected CHW (1HW for single channel) or HW."
        )
    if not np.issubdtype(mask_np.dtype, np.integer):
        mask_np = mask_np.astype(np.int32)

    try:
        augmented = augmentation_pipeline(image=image_np, mask=mask_np)
        augmented_image_np = augmented['image']
        augmented_mask_np = augmented['mask']
    except Exception as e:
        print(f"Error during albumentations: {e}")
        return image_tensor, mask_tensor

    augmented_image_tensor = torch.from_numpy(augmented_image_np).permute(2, 0, 1)
    if original_image_dtype == torch.float32 :
         augmented_image_tensor = augmented_image_tensor.float()

    augmented_mask_tensor = torch.from_numpy(augmented_mask_np).unsqueeze(0)
    if original_mask_dtype == torch.long:
        augmented_mask_tensor = augmented_mask_tensor.long()

    return augmented_image_tensor.to(original_device), augmented_mask_tensor.to(original_device)