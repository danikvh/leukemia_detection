"""Configuration management for datasets."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
from common.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    """Configuration class for dataset parameters."""

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            # Paths
            "data_root": "data",
            "img_path": None,
            "mask_path": None,

            # Dataset parameters
            "do_augmentation": False,
            "complex_augmentation": False,
            "stain_transform": True,
            "rgb_transform": False,
            "eosin": 0.0,
            "dab": 1.0,
            "gamma": 2.1,
            "normalize": True,  # (kept final one)
            "normalize_inf": True,
            "inversion": False,
            "only_nuclei": False,
            "debug": False,

            # Dataloader parameters
            "device": "auto",
            "num_workers": 0,
            "batch_size": 1,
            "pin_memory": True,

            # Image processing
            "target_size": None,  # (tuple or None)

            # Validation splits
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,

            # File patterns
            "image_extensions": [".npy", ".png", ".jpg", ".tiff"],
            "mask_extensions": [".npy", ".png"],
        }