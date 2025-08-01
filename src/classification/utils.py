import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # For loading image formats like PNG

logger = logging.getLogger(__name__)

def load_classification_labels(csv_path: Path, classification_mode: str = "binary") -> Dict[str, int]:
    """
    Loads cell image filenames and their corresponding labels from a CSV file.
    
    Binary mode: 'positive' or 'negative' -> 1 or 0
    Ternary mode: 'positive', 'negative', or 'false-positive' -> 1, 0, or 2
    
    Args:
        csv_path (Path): Path to the CSV file.
        classification_mode (str): Either "binary" or "ternary".
        
    Returns:
        Dict[str, int]: A dictionary mapping filename (stem) to its label.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        if 'filename' not in df.columns or 'label' not in df.columns:
            raise ValueError(
                "CSV must contain 'filename' and 'label' columns."
                f" Found columns: {df.columns.tolist()}"
            )

        # Normalize label values
        df['label'] = df['label'].str.strip().str.lower()  # Handle casing and whitespace
        
        if classification_mode == "binary":
            label_mapping = {'positive': 1, 'negative': 0}
            expected_labels = set(label_mapping.keys())
        else:  # ternary
            label_mapping = {'positive': 1, 'negative': 0, 'false-positive': 2}
            expected_labels = set(label_mapping.keys())
        
        # Check for valid labels
        actual_labels = set(df['label'].unique())
        if not actual_labels.issubset(expected_labels):
            invalid_labels = actual_labels - expected_labels
            raise ValueError(
                f"Invalid labels found for {classification_mode} mode: {invalid_labels}. "
                f"Expected labels: {expected_labels}"
            )
        
        # Map labels to integers
        df['label'] = df['label'].map(label_mapping)
        
        # Check for unmapped labels (shouldn't happen after validation, but safety check)
        if df['label'].isnull().any():
            unmapped = df[df['label'].isnull()]['label'].unique()
            raise ValueError(f"Failed to map some labels: {unmapped}")
        
        # Ensure filename is just the stem (without extension) for easier matching
        df['filename'] = df['filename'].apply(lambda x: Path(x).stem)
        
        label_map = dict(zip(df['filename'], df['label']))
        
        # Log class distribution
        class_counts = df['label'].value_counts().sort_index()
        logger.info(f"Loaded {len(label_map)} classification labels from {csv_path} ({classification_mode} mode)")
        for label_idx, count in class_counts.items():
            if classification_mode == "binary":
                label_name = "positive" if label_idx == 1 else "negative"
            else:
                label_names = {0: "negative", 1: "positive", 2: "false-positive"}
                label_name = label_names[label_idx]
            logger.info(f"  {label_name}: {count} samples")
        
        return label_map
        
    except Exception as e:
        logger.error(f"Error loading labels from {csv_path}: {e}")
        raise

def get_image_paths_and_labels(
    data_dir: Path, 
    label_map: Dict[str, int],
    image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.npy')
) -> List[Tuple[Path, int]]:
    """
    Collects valid image paths and their corresponding labels from the data directory.
    
    Args:
        data_dir (Path): Directory containing the cell images.
        label_map (Dict[str, int]): Map of filename stems to labels.
        image_extensions (Tuple[str, ...]): Supported image file extensions.
        
    Returns:
        List[Tuple[Path, int]]: A list of (image_path, label) tuples.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    all_image_files = [
        f for f in data_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    data_samples = []
    for img_path in sorted(all_image_files): # Sort for reproducibility
        filename_stem = img_path.stem
        if filename_stem in label_map:
            data_samples.append((img_path, label_map[filename_stem]))
        else:
            logger.warning(f"No label found for image: {img_path.name}. Skipping.")
            
    if not data_samples:
        raise ValueError(f"No valid image files with corresponding labels found in {data_dir}")
            
    logger.info(f"Found {len(data_samples)} images with labels in {data_dir}")
    return data_samples

def split_data(
    data_samples: List[Tuple[Path, int]], 
    train_split: float, 
    val_split: float, 
    test_split: float, 
    random_seed: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Splits the data samples into training, validation, and test sets.
    
    Args:
        data_samples (List[Tuple[Path, int]]): List of (image_path, label) tuples.
        train_split (float): Proportion of data for training.
        val_split (float): Proportion of data for validation.
        test_split (float): Proportion of data for testing.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[List, List, List]: (train_data, val_data, test_data) lists.
    """
    if not (0 <= train_split <= 1 and 0 <= val_split <= 1 and 0 <= test_split <= 1 and 
            abs(train_split + val_split + test_split - 1.0) < 1e-6):
        raise ValueError("Splits must sum to 1.0 and be between 0 and 1.")
        
    labels = [sample[1] for sample in data_samples]
    
    # Handle cases where all labels are the same (stratify would fail)
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class found in data. Skipping stratified split.")
        train_data, temp_data = train_test_split(
            data_samples, test_size=(val_split + test_split), random_state=random_seed
        )
        if val_split > 0 and test_split > 0:
            val_data, test_data = train_test_split(
                temp_data, test_size=(test_split / (val_split + test_split)), random_state=random_seed
            )
        elif val_split > 0:
            val_data = temp_data
            test_data = []
        elif test_split > 0:
            test_data = temp_data
            val_data = []
        else:
            val_data = []
            test_data = []

    else:
        # First split: train vs (val + test)
        train_val_test_split_ratio = val_split + test_split
        
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data_samples, labels, 
            test_size=train_val_test_split_ratio, 
            random_state=random_seed,
            stratify=labels
        )
        
        # Second split: val vs test from temp_data
        if val_split > 0 and test_split > 0:
            val_test_ratio = test_split / (val_split + test_split)
            val_data, test_data, _, _ = train_test_split(
                temp_data, temp_labels, 
                test_size=val_test_ratio, 
                random_state=random_seed,
                stratify=temp_labels
            )
        elif val_split > 0:
            val_data = temp_data
            test_data = []
        elif test_split > 0:
            test_data = temp_data
            val_data = []
        else: # Only train_split = 1.0
            val_data = []
            test_data = []

    logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def calculate_pos_weight(data_samples: List[Tuple[Path, int]]) -> float:
    """
    Calculates the positive weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = num_negative_samples / num_positive_samples
    
    Note: This is only applicable for binary classification.
    
    Args:
        data_samples (List[Tuple[Path, int]]): List of (image_path, label) tuples.
        
    Returns:
        float: The calculated positive weight. Returns 1.0 if no positive samples.
    """
    labels = [sample[1] for sample in data_samples]
    num_positive = sum(1 for label in labels if label == 1)
    num_negative = sum(1 for label in labels if label == 0)
    
    if num_positive == 0:
        logger.warning("No positive samples found. Cannot calculate pos_weight, returning 1.0.")
        return 1.0
    
    pos_weight = num_negative / num_positive
    logger.info(f"Calculated pos_weight: {pos_weight:.2f} (Negative: {num_negative}, Positive: {num_positive})")
    return pos_weight

def calculate_class_weights(data_samples: List[Tuple[Path, int]], num_classes: int) -> np.ndarray:
    """
    Calculates class weights for multi-class classification to handle class imbalance.
    
    Args:
        data_samples (List[Tuple[Path, int]]): List of (image_path, label) tuples.
        num_classes (int): Number of classes (e.g., 3 for ternary).
        
    Returns:
        np.ndarray: Array of class weights.
    """
    labels = [sample[1] for sample in data_samples]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) != num_classes:
        logger.warning(f"Expected {num_classes} classes but found {len(unique_labels)}: {unique_labels}")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Ensure we have weights for all classes (fill missing with 1.0)
    full_weights = np.ones(num_classes)
    for i, label in enumerate(unique_labels):
        full_weights[label] = class_weights[i]
    
    logger.info(f"Calculated class weights: {full_weights}")
    for i, weight in enumerate(full_weights):
        count = sum(1 for label in labels if label == i)
        logger.info(f"  Class {i}: weight={weight:.3f}, count={count}")
    
    return full_weights

def visualize_cell(
    image_input: Union[Path, torch.Tensor], # Changed type to accept Path or Tensor
    true_label: Union[int, torch.Tensor],    # Changed type to accept int or Tensor
    filename_stem: Optional[str] = None,     # Optional filename for title
    predicted_label: Optional[str] = None,
    probability: Optional[Union[float, List[float]]] = None,  # Can be single prob or list of probs
    classification_mode: str = "binary",     # NEW: Support for different modes
    figsize: Tuple[int, int] = (4, 4)
) -> None:
    """
    Visualizes a single cell image along with its true label and optionally
    its predicted label and probability.

    Args:
        image_input (Union[Path, torch.Tensor]): Path to the cell image file OR a torch.Tensor
                                                 representing the image (C, H, W).
        true_label (Union[int, torch.Tensor]): The ground truth label OR a torch.Tensor
                                                containing the label.
        filename_stem (Optional[str]): Optional filename stem to display in the title if
                                       image_input is a tensor.
        predicted_label (Optional[str]): The predicted label.
        probability (Optional[Union[float, List[float]]]): The predicted probability(ies).
        classification_mode (str): Either "binary" or "ternary".
        figsize (Tuple[int, int]): Figure size for the plot.
    """
    img_data = None # Will store numpy array for imshow
    
    # Extract true_label value
    if isinstance(true_label, torch.Tensor):
        true_label_val = true_label.item() if true_label.ndim == 0 else true_label.squeeze().item()
    else:
        true_label_val = true_label

    try:
        if isinstance(image_input, torch.Tensor):
            # Handle PyTorch Tensor input
            img_tensor = image_input.cpu().detach() # Move to CPU and detach from graph

            # Normalize to 0-1 range if not already (e.g., if it came from transforms.Normalize)
            # This is a heuristic; ideally, you'd know your tensor's original range
            # For ImageNet normalized tensors, we need to denormalize first
            if img_tensor.dtype == torch.float32 and img_tensor.max() > 1.0 + 1e-5:
                # Assuming standard ImageNet normalization was applied [mean, std]
                # If your images are just scaled to 0-1, you might remove this or adjust
                logger.debug("Attempting to denormalize tensor for visualization.")
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # Example ImageNet mean
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # Example ImageNet std
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1) # Clamp to [0, 1] range
                img_np = (img_tensor * 255).to(torch.uint8).numpy()
            elif img_tensor.dtype == torch.float32: # Assume [0,1] float range
                img_np = (img_tensor * 255).to(torch.uint8).numpy()
            else: # Assume already uint8 or other integer type
                img_np = img_tensor.numpy()

            # Convert from C, H, W to H, W, C for matplotlib
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]: # C, H, W
                if img_np.shape[0] == 1: # Grayscale (1, H, W) to (H, W, 3)
                    img_np = np.repeat(img_np.transpose(1, 2, 0), 3, axis=2)
                elif img_np.shape[0] == 4: # RGBA (4, H, W) to (H, W, 3)
                    img_np = img_np[:3, :, :].transpose(1, 2, 0)
                else: # RGB (3, H, W) to (H, W, 3)
                    img_np = img_np.transpose(1, 2, 0)
            
            img_data = img_np
            # Set filename_stem for title if not provided
            if filename_stem is None:
                filename_stem = "Tensor_Input"

        elif isinstance(image_input, Path):
            # Existing Path handling logic
            if not image_input.exists():
                logger.error(f"Image file not found: {image_input}")
                return

            if image_input.suffix.lower() == '.npy':
                img_np = np.load(image_input)
                # Ensure numpy array is uint8 for display (0-255 range)
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0 and img_np.min() >= 0.0: # Assume float [0,1]
                        img_np = (img_np * 255).astype(np.uint8)
                    else: # Assume some other range, scale to 0-255 using min/max
                        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)

                if img_np.ndim == 2: # Grayscale, make it RGB for consistent display
                    img_data = np.stack([img_np]*3, axis=-1)
                elif img_np.ndim == 3:
                    if img_np.shape[2] == 4: # RGBA, drop alpha
                        img_data = img_np[:, :, :3]
                    elif img_np.shape[2] == 1: # Grayscale (H, W, 1), repeat to RGB
                        img_data = np.repeat(img_np, 3, axis=2)
                    elif img_np.shape[2] == 3: # Already RGB
                        img_data = img_np
                    else:
                        logger.warning(f"Unexpected channel count ({img_np.shape[2]}) for {image_input.name}. Displaying as is, may be malformed.")
                        img_data = img_np
                else:
                    logger.error(f"Unsupported number of dimensions for .npy image: {img_np.ndim} at {image_input.name}")
                    return

            else: # For .png, .jpg, .tiff, etc.
                img = Image.open(image_input)
                # Ensure it's RGB to be displayed correctly by matplotlib
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_data = np.array(img) # Convert PIL Image to NumPy array
            
            if filename_stem is None:
                filename_stem = image_input.stem

        else:
            logger.error(f"Unsupported image_input type: {type(image_input)}. Must be Path or torch.Tensor.")
            return

        if img_data is None or img_data.size == 0:
            logger.error(f"Failed to load or process image data for {filename_stem if filename_stem else 'unknown image'}")
            return

        plt.figure(figsize=figsize)
        
        # Decide interpolation based on image size. For very small images, 'nearest' is better.
        interpolation_style = 'bilinear'
        if img_data.shape[0] < 50 or img_data.shape[1] < 50: # Check height/width
            interpolation_style = 'nearest'
            logger.debug(f"Image is small ({img_data.shape[1]}x{img_data.shape[0]}), using 'nearest' interpolation.")

        plt.imshow(img_data, interpolation=interpolation_style) # Use numpy array
        plt.axis('off')

        # Handle true label display based on classification mode
        if classification_mode == "binary":
            true_label_str = "Cancerous" if true_label_val == 1 else "Non-Cancerous"
        else:  # ternary
            label_names = {0: "Non-Cancerous", 1: "Cancerous", 2: "False-Positive"}
            true_label_str = label_names.get(true_label_val, f"Unknown({true_label_val})")

        title_text = f"File: {filename_stem}\nTrue: {true_label_str}" # Include filename_stem

        if predicted_label is not None and probability is not None:
            if isinstance(probability, list):
                # For ternary classification, show all probabilities
                prob_str = ", ".join([f"{p:.3f}" for p in probability])
                title_text += f"\nPred: {predicted_label}\nProbs: [{prob_str}]"
            else:
                # For binary classification, show single probability
                title_text += f"\nPred: {predicted_label} (Prob: {probability:.4f})"
        elif predicted_label is not None:
            title_text += f"\nPred: {predicted_label}"

        plt.title(title_text, fontsize=10) # Reduced font size to accommodate more text
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing image ({filename_stem if filename_stem else 'unknown'}): {e}")

def visualize_class_distribution(data_samples: List[Tuple[Path, int]], classification_mode: str = "binary", 
                               title: str = "Class Distribution", figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualizes the distribution of classes in the dataset.
    
    Args:
        data_samples (List[Tuple[Path, int]]): List of (image_path, label) tuples.
        classification_mode (str): Either "binary" or "ternary".
        title (str): Title for the plot.
        figsize (Tuple[int, int]): Figure size for the plot.
    """
    labels = [sample[1] for sample in data_samples]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Define class names based on mode
    if classification_mode == "binary":
        class_names = {0: "Non-Cancerous", 1: "Cancerous"}
    else:  # ternary
        class_names = {0: "Non-Cancerous", 1: "Cancerous", 2: "False-Positive"}
    
    # Create labels for plotting
    plot_labels = [class_names.get(label, f"Class {label}") for label in unique_labels]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(plot_labels, counts, alpha=0.7, 
                   color=['lightblue', 'lightcoral', 'lightgreen'][:len(unique_labels)])
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage annotations
    total_samples = sum(counts)
    for i, (label, count) in enumerate(zip(plot_labels, counts)):
        percentage = (count / total_samples) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white' if count > max(counts)*0.3 else 'black')
    
    plt.tight_layout()
    plt.show()
    
    # Log the distribution
    logger.info(f"Class distribution ({classification_mode} mode):")
    for label, count in zip(unique_labels, counts):
        class_name = class_names.get(label, f"Class {label}")
        percentage = (count / total_samples) * 100
        logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")