from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from pathlib import Path
import yaml

@dataclass
class ClassificationConfig:
    """Configuration class for cell classification."""

    # Classification mode
    classification_mode: str = "binary"  # Options: "binary", "ternary"
    
    # Data paths
    output_dir: str = "results/cell_classification" # Output directory
    data_dir: str = "data/classified_cells" # Directory containing individual cell images (output from segmentation)
    labels_csv: str = "data/classified_cells/labels.csv" # Path to CSV with labels (e.g., filename, label)

    # Image preprocessing
    image_size: int = 96 # Target size (e.g., 96x96) after padding and resizing
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406]) # ImageNet mean
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225]) # ImageNet std

    # Training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    optimizer: str = "AdamW" # Options: "AdamW", "SGD"
    # NEW: Modified to support both binary and multi-class
    loss_function: str = "auto"  # Options: "auto", "BCEWithLogitsLoss", "CrossEntropyLoss"
    weight_decay: float = 1e-5
    
    # Class imbalance handling
    use_class_weights: bool = True # Apply class weighting to loss function

    # Dataloader parameters
    num_workers: int = 4
    pin_memory: bool = True

    # Model parameters
    model_name: str = "resnet18" # Options: "custom_cnn", "resnet18", "resnet34", "resnet50"
    pretrained: bool = True # Use ImageNet pretrained weights if using torchvision models

    # Early Stopping
    patience: int = 10 # Number of epochs to wait for improvement before stopping
    early_stopping_metric: str = "val_loss" # Metric to monitor for early stopping ('val_loss', 'val_accuracy', 'val_f1', etc.)

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    # Inference parameters - updated for ternary support
    confidence_threshold_high: float = 0.9 # Probability threshold for 'cancerous' (binary) or highest confidence (ternary)
    confidence_threshold_low: float = 0.3  # Probability threshold for 'non-cancerous' (binary) or lowest confidence (ternary)
    # NEW: Additional threshold for ternary classification uncertainty
    uncertainty_threshold: float = 0.1  # If max probability - second_max probability < this, classify as uncertain

    def __post_init__(self):
        """Validate configuration after initialization."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not 0 <= self.train_split <= 1:
            raise ValueError("Train split must be between 0 and 1")
        
        if not 0 <= self.confidence_threshold_low <= self.confidence_threshold_high <= 1:
            raise ValueError("Confidence thresholds must be between 0 and 1 and low <= high")
            
        if self.classification_mode not in ["binary", "ternary"]:
            raise ValueError("Classification mode must be 'binary' or 'ternary'")
            
        # Auto-set loss function based on classification mode
        if self.loss_function == "auto":
            if self.classification_mode == "binary":
                self.loss_function = "BCEWithLogitsLoss"
            else:  # ternary
                self.loss_function = "CrossEntropyLoss"
        
        # Validate loss function compatibility
        if self.classification_mode == "binary" and self.loss_function == "CrossEntropyLoss":
            raise ValueError("CrossEntropyLoss is not compatible with binary classification mode")
        if self.classification_mode == "ternary" and self.loss_function == "BCEWithLogitsLoss":
            raise ValueError("BCEWithLogitsLoss is not compatible with ternary classification mode")

    @property
    def num_classes(self) -> int:
        """Returns the number of classes based on classification mode."""
        return 1 if self.classification_mode == "binary" else 3

    @property
    def class_names(self) -> List[str]:
        """Returns class names based on classification mode."""
        if self.classification_mode == "binary":
            return ["non-cancerous", "cancerous"]
        else:
            return ["non-cancerous", "cancerous", "false-positive"]

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ClassificationConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        config_dict = self.__dict__.copy()
        
        # Convert Path objects to strings for YAML serialization if any
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

def create_default_classification_config(output_dir: Union[str, Path] = "configs/classification/", mode: str = "binary") -> None:
    """Create a default example classification configuration file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = ClassificationConfig(classification_mode=mode)
    suffix = f"_{mode}" if mode != "binary" else ""
    config.to_yaml(output_path / f"default_classification_config{suffix}.yaml")
    print(f"Created {output_path / f'default_classification_config{suffix}.yaml'}")

# Create both binary and ternary configs by default
def create_both_configs(output_dir: Union[str, Path] = "configs/classification/") -> None:
    """Create both binary and ternary default configuration files."""
    create_default_classification_config(output_dir, "binary")
    create_default_classification_config(output_dir, "ternary")