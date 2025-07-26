# src/transforms/stain_deconvolution.py
from typing import Tuple, Optional
import torch
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from skimage import exposure
from .base import ImageOnlyTransform, ParameterizedTransform


class StainDeconvolution(ImageOnlyTransform, ParameterizedTransform):
    """
    Perform stain deconvolution for histological images.
    
    Supports:
    - H&E: Hematoxylin (nuclei) + Eosin (cytoplasm)  
    - IHC: Hematoxylin (nuclei) + DAB (target protein)
    """
    
    def __init__(
        self, 
        stain_type: str = 'ihc',  # 'ihc' or 'he'
        gamma: float = 2.1,
        enhance_nuclei: bool = True,
        enhance_cytoplasm: bool = True,
        invert_stains: bool = False,
        nuclear_channel: str = 'green',  # 'red', 'green', 'blue'
        cytoplasm_channel: str = 'blue'
    ):
        super().__init__(
            stain_type=stain_type, gamma=gamma, enhance_nuclei=enhance_nuclei,
            enhance_cytoplasm=enhance_cytoplasm, invert_stains=invert_stains,
            nuclear_channel=nuclear_channel, cytoplasm_channel=cytoplasm_channel
        )
        
        self.stain_type = stain_type.lower()
        self.gamma = gamma
        self.enhance_nuclei = enhance_nuclei
        self.enhance_cytoplasm = enhance_cytoplasm
        self.invert_stains = invert_stains
        
        # Channel mapping
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        self.nuclear_channel_idx = channel_map[nuclear_channel.lower()]
        self.cytoplasm_channel_idx = channel_map[cytoplasm_channel.lower()]
        
        # Validate stain type
        if self.stain_type not in ['ihc', 'he']:
            raise ValueError(f"Unsupported stain_type: {stain_type}. Must be 'ihc' or 'he'")
    
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply stain deconvolution to RGB image."""
        # Convert to numpy for skimage processing
        if image.dtype == torch.uint8:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            # Assume float in [0, 255] range
            img_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Perform stain deconvolution
        deconvolved = self._deconvolve_stains(img_np)
        
        # Convert back to tensor
        result = torch.from_numpy(deconvolved).permute(2, 0, 1).to(
            device=image.device, dtype=image.dtype
        )
        
        return result
    
    def _deconvolve_stains(self, image_rgb: np.ndarray) -> np.ndarray:
        """Perform the actual stain deconvolution."""
        # Normalize to [0, 1] for skimage
        image_float = image_rgb.astype(np.float64) / 255.0
        
        # Perform HED deconvolution
        hed = rgb2hed(image_float)
        
        # Extract individual stains
        hematoxylin = self._process_stain_channel(hed[:, :, 0], 'hematoxylin')
        eosin = self._process_stain_channel(hed[:, :, 1], 'eosin')  
        dab = self._process_stain_channel(hed[:, :, 2], 'dab')
        
        # Create output channels based on stain type
        rgb_output = np.zeros_like(image_float)
        
        if self.enhance_nuclei:
            rgb_output[:, :, self.nuclear_channel_idx] = hematoxylin
        
        if self.enhance_cytoplasm:
            if self.stain_type == 'ihc':
                cytoplasm_stain = dab
            else:  # H&E
                cytoplasm_stain = eosin
            
            rgb_output[:, :, self.cytoplasm_channel_idx] = cytoplasm_stain
        
        # Convert back to uint8
        return (rgb_output * 255).astype(np.uint8)
    
    def _process_stain_channel(self, stain_channel: np.ndarray, stain_name: str) -> np.ndarray:
        """Process individual stain channel with normalization and gamma correction."""
        # Normalize to [0, 1]
        normalized = exposure.rescale_intensity(stain_channel, out_range=(0.0, 1.0))
        
        # Optional inversion
        if self.invert_stains:
            normalized = 1.0 - normalized
        
        # Gamma correction
        gamma_corrected = exposure.adjust_gamma(normalized, gamma=self.gamma)
        
        return gamma_corrected


class AdaptiveStainNormalization(ImageOnlyTransform, ParameterizedTransform):
    """Adaptive stain normalization using reference statistics."""
    
    def __init__(
        self, 
        reference_image: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.15,
        clip_limit: float = 0.01
    ):
        super().__init__(reference_image=reference_image, alpha=alpha, beta=beta, clip_limit=clip_limit)
        self.reference_image = reference_image
        self.alpha = alpha
        self.beta = beta
        self.clip_limit = clip_limit
        
        if reference_image is not None:
            self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute reference statistics from reference image."""
        if self.reference_image is None:
            return
        
        # Convert to numpy for processing
        ref_np = self.reference_image.permute(1, 2, 0).cpu().numpy() / 255.0
        ref_hed = rgb2hed(ref_np)
        
        # Compute mean and std for each stain channel
        self.ref_means = np.mean(ref_hed.reshape(-1, 3), axis=0)
        self.ref_stds = np.std(ref_hed.reshape(-1, 3), axis=0)
    
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply adaptive stain normalization."""
        if self.reference_image is None:
            return image
        
        # Convert to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy() / 255.0
        
        # Convert to HED
        hed = rgb2hed(img_np)
        
        # Normalize each channel
        normalized_hed = np.zeros_like(hed)
        for i in range(3):
            channel = hed[:, :, i]
            channel_mean = np.mean(channel)
            channel_std = np.std(channel)
            
            # Avoid division by zero
            if channel_std > 1e-6:
                normalized_channel = (channel - channel_mean) * (self.ref_stds[i] / channel_std) + self.ref_means[i]
                normalized_hed[:, :, i] = normalized_channel
            else:
                normalized_hed[:, :, i] = channel
        
        # Convert back to RGB
        normalized_rgb = hed2rgb(normalized_hed)
        normalized_rgb = np.clip(normalized_rgb, 0, 1)
        
        # Convert back to tensor
        result = torch.from_numpy((normalized_rgb * 255).astype(np.uint8))
        result = result.permute(2, 0, 1).to(device=image.device, dtype=image.dtype)
        
        return result


class StainAugmentation(ImageOnlyTransform, ParameterizedTransform):
    """
    Augment staining appearance by modifying stain concentrations.
    Useful for domain adaptation and data augmentation.
    """
    
    def __init__(
        self,
        hematoxylin_range: Tuple[float, float] = (0.8, 1.2),
        eosin_range: Tuple[float, float] = (0.8, 1.2), 
        dab_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5
    ):
        super().__init__(
            hematoxylin_range=hematoxylin_range, eosin_range=eosin_range,
            dab_range=dab_range, probability=probability
        )
        self.hematoxylin_range = hematoxylin_range
        self.eosin_range = eosin_range
        self.dab_range = dab_range
        self.probability = probability
    
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply stain augmentation."""
        if np.random.random() > self.probability:
            return image
        
        # Convert to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy() / 255.0
        
        # Convert to HED
        hed = rgb2hed(img_np)
        
        # Generate random scaling factors
        h_factor = np.random.uniform(*self.hematoxylin_range)
        e_factor = np.random.uniform(*self.eosin_range)
        d_factor = np.random.uniform(*self.dab_range)
        
        # Scale stain channels
        hed[:, :, 0] *= h_factor  # Hematoxylin
        hed[:, :, 1] *= e_factor  # Eosin
        hed[:, :, 2] *= d_factor  # DAB
        
        # Convert back to RGB
        augmented_rgb = hed2rgb(hed)
        augmented_rgb = np.clip(augmented_rgb, 0, 1)
        
        # Convert back to tensor
        result = torch.from_numpy((augmented_rgb * 255).astype(np.uint8))
        result = result.permute(2, 0, 1).to(device=image.device, dtype=image.dtype)
        
        return result
