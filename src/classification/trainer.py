import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional, Union, List
import matplotlib.pyplot as plt
import seaborn as sns

from classification.config import ClassificationConfig

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Trainer class for cell classification models.
    Supports both binary and ternary classification modes.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ClassificationConfig,
        pos_weight: Optional[float] = None, # For BCEWithLogitsLoss to handle imbalance
        class_weights: Optional[Union[torch.Tensor, np.ndarray]] = None  # For CrossEntropyLoss
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer setup
        if config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Loss function setup
        if config.loss_function == "BCEWithLogitsLoss":
            if config.use_class_weights and pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.device, dtype=torch.float32))
                logger.info(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight:.2f}")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                logger.info("Using BCEWithLogitsLoss without class weights.")
        elif config.loss_function == "CrossEntropyLoss":
            if config.use_class_weights and class_weights is not None:
                if isinstance(class_weights, np.ndarray):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
                logger.info(f"Using CrossEntropyLoss with class weights: {class_weights}")
            else:
                self.criterion = nn.CrossEntropyLoss()
                logger.info("Using CrossEntropyLoss without class weights.")
        else:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")
            
        # Early stopping parameters
        self.patience = config.patience
        self.early_stopping_metric = config.early_stopping_metric

        # Initialize best score based on metric type
        if self.early_stopping_metric == "val_loss":
            self.best_early_stopping_score = float('inf') # Lower is better
            self.is_better = lambda current, best: current < best
        else: # For accuracy, precision, recall, f1, etc. (higher is better)
            self.best_early_stopping_score = -float('inf') # Higher is better
            self.is_better = lambda current, best: current > best
        
        self.epochs_no_improve = 0 # Reset epochs no improve counter

        # Initialize metrics history based on classification mode
        if config.classification_mode == "binary":
            self.metrics_history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [], 
                'val_precision': [], 
                'val_recall': [], 
                'val_f1': [], 
                'val_auc': [],
                'val_undecided_percentage': []
            }
        else:  # ternary
            self.metrics_history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [], 
                'val_precision_macro': [],
                'val_recall_macro': [],
                'val_f1_macro': [],
                'val_precision_weighted': [],
                'val_recall_weighted': [],
                'val_f1_weighted': []
            }
        
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, labels, _ in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        if self.config.classification_mode == "binary":
            undecided_count = 0

        with torch.no_grad():
            for inputs, labels, _ in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                if self.config.classification_mode == "binary":
                    probs = torch.sigmoid(outputs)
                    preds_binary = (probs > 0.5).long()
                    
                    all_preds.extend(preds_binary.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    all_probs.extend(probs.cpu().numpy().flatten())

                    # Count undecided samples for binary classification
                    for prob in probs.cpu().numpy().flatten():
                        if self.config.confidence_threshold_low < prob < self.config.confidence_threshold_high:
                            undecided_count += 1
                            
                else:  # ternary
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics based on classification mode
        if self.config.classification_mode == "binary":
            return self._calculate_binary_metrics(avg_loss, all_labels, all_preds, all_probs, undecided_count)
        else:
            return self._calculate_ternary_metrics(avg_loss, all_labels, all_preds, all_probs)

    def _calculate_binary_metrics(self, avg_loss: float, all_labels: List, all_preds: List, 
                                all_probs: List, undecided_count: int) -> Dict[str, float]:
        """Calculate metrics for binary classification."""
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        
        if len(all_labels) == 0:
            recall = 0.0
            logger.warning("Recall not calculated: No true labels found in validation set.")
        else:
            recall = recall_score(all_labels, all_preds, zero_division=0)

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        auc = 0.0
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            logger.warning("AUC not calculated: Validation set contains only one class.")

        total_val_samples = len(all_labels)
        undecided_percentage = (undecided_count / total_val_samples) if total_val_samples > 0 else 0.0

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc,
            'val_undecided_percentage': undecided_percentage
        }

    def _calculate_ternary_metrics(self, avg_loss: float, all_labels: List, all_preds: List, 
                                 all_probs: List) -> Dict[str, float]:
        """Calculate metrics for ternary classification."""
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate macro and weighted averages
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision_macro': precision_macro,
            'val_recall_macro': recall_macro,
            'val_f1_macro': f1_macro,
            'val_precision_weighted': precision_weighted,
            'val_recall_weighted': recall_weighted,
            'val_f1_weighted': f1_weighted
        }

    def train(self, output_dir: Path):
        """
        Main training loop.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {self.config.epochs} epochs on {self.device}")
        logger.info(f"Classification mode: {self.config.classification_mode}")
        logger.info(f"Early stopping monitor: {self.early_stopping_metric} with patience {self.patience}")

        model_save_path = output_dir / f"best_{self.config.classification_mode}_classification_model.pth"
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch()

            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            for key, value in val_metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)

            # Log epoch results
            if self.config.classification_mode == "binary":
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                    f"Val F1: {val_metrics['val_f1']:.4f}, "
                    f"Val AUC: {val_metrics['val_auc']:.4f}, "
                    f"Val Undecided: {val_metrics['val_undecided_percentage']:.2%}"
                )
            else:  # ternary
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                    f"Val F1 (macro): {val_metrics['val_f1_macro']:.4f}, "
                    f"Val F1 (weighted): {val_metrics['val_f1_weighted']:.4f}"
                )

            # Early stopping logic
            current_early_stopping_score = val_metrics[self.early_stopping_metric]

            if self.is_better(current_early_stopping_score, self.best_early_stopping_score):
                self.best_early_stopping_score = current_early_stopping_score
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"Saved best model to {model_save_path} ({self.early_stopping_metric}: {self.best_early_stopping_score:.4f})")
            else:
                self.epochs_no_improve += 1
                logger.info(f"No improvement for {self.early_stopping_metric}. Patience: {self.epochs_no_improve}/{self.patience}")

            # Save training history
            history_path = output_dir / f"{self.config.classification_mode}_training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement in {self.early_stopping_metric}.")
                break
        
        logger.info("Training complete.")
    
    def plot_metrics_history(self, save_path: Optional[Path] = None) -> None:
        """
        Plots the training and validation metrics over epochs.
        
        Args:
            save_path (Optional[Path]): If provided, saves the plot to this path.
        """
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        if self.config.classification_mode == "binary":
            self._plot_binary_metrics(epochs, save_path)
        else:
            self._plot_ternary_metrics(epochs, save_path)