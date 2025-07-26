from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict
import torch
import numpy as np


class Transform(ABC):
    """Base class for all transforms."""
    
    @abstractmethod
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply transform to image and optionally mask."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ParameterizedTransform(Transform):
    """Base class for transforms with parameters."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def __repr__(self) -> str:
        param_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"


class PairedTransform(Transform):
    """Base class for transforms that require both image and mask."""
    
    @abstractmethod
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transform to both image and mask."""
        pass


class ImageOnlyTransform(Transform):
    """Base class for transforms that only affect the image."""
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        transformed_image = self.transform_image(image)
        return transformed_image, mask
    
    @abstractmethod
    def transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """Transform only the image."""
        pass


class Compose(Transform):
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
    
    def __repr__(self) -> str:
        transform_strs = [str(t) for t in self.transforms]
        return f"Compose([{', '.join(transform_strs)}])"


class ConditionalTransform(Transform):
    """Apply transform based on a condition."""
    
    def __init__(self, transform: Transform, condition: callable, probability: float = 1.0):
        self.transform = transform
        self.condition = condition
        self.probability = probability
    
    def __call__(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.random() < self.probability and self.condition(image, mask):
            return self.transform(image, mask)
        return image, mask