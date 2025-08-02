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
    UPDATED: Now supports train-only mode when val_loader is None.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],  # Can be None for train-only mode
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
        
        # Check if we're in train-only mode
        self.is_train_only = (val_loader is None)
        if self.is_train_only:
            logger.info("🔥 Trainer initialized in TRAIN-ONLY MODE")
            logger.info("⚠️  Early stopping will be disabled")
            logger.info("⚠️  Model will be saved after final epoch")

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
            
        # Early stopping parameters (disabled in train-only mode)
        if not self.is_train_only:
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
        else:
            # Disable early stopping for train-only mode
            self.patience = float('inf')
            self.early_stopping_metric = None
            self.best_early_stopping_score = None
            self.epochs_no_improve = 0

        # Initialize metrics history based on classification mode
        if not self.is_train_only:
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
        else:
            # For train-only mode, only track training loss
            self.metrics_history = {
                'train_loss': []
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
        """Validation epoch - only called when val_loader is available."""
        if self.is_train_only:
            logger.warning("_validate_epoch called in train-only mode")
            return {}
            
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
        UPDATED: Handles train-only mode without validation.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {self.config.epochs} epochs on {self.device}")
        logger.info(f"Classification mode: {self.config.classification_mode}")
        
        if not self.is_train_only:
            logger.info(f"Early stopping monitor: {self.early_stopping_metric} with patience {self.patience}")
        else:
            logger.info("Train-only mode: No validation, model will be saved after final epoch")

        model_save_path = output_dir / f"best_{self.config.classification_mode}_classification_model.pth"
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            
            if not self.is_train_only:
                # Run validation if available
                val_metrics = self._validate_epoch()
                
                # Update validation metrics history
                for key, value in val_metrics.items():
                    if key in self.metrics_history:
                        self.metrics_history[key].append(value)

                # Log epoch results with validation
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

                # Check early stopping
                if self.epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement in {self.early_stopping_metric}.")
                    break
                    
            else:
                # Train-only mode: just log training loss
                logger.info(f"Epoch {epoch}/{self.config.epochs} - Train Loss: {train_loss:.4f}")
                
                # Save model periodically in train-only mode (every 10 epochs and final epoch)
                if epoch % 10 == 0 or epoch == self.config.epochs:
                    epoch_model_path = output_dir / f"{self.config.classification_mode}_model_epoch_{epoch}.pth"
                    torch.save(self.model.state_dict(), epoch_model_path)
                    logger.info(f"Saved model checkpoint to {epoch_model_path}")
                    
                    # Also save as "best" model (since we have no validation to determine best)
                    torch.save(self.model.state_dict(), model_save_path)

            # Save training history
            history_path = output_dir / f"{self.config.classification_mode}_training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        
        # Final model save for train-only mode
        if self.is_train_only:
            torch.save(self.model.state_dict(), model_save_path)
            logger.info(f"Final model saved to {model_save_path}")
        
        logger.info("Training complete.")
    
    def evaluate_test_set(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test set.
        UPDATED: Added check to ensure test_loader is available.
        """
        if test_loader is None:
            logger.error("Cannot evaluate: test_loader is None")
            return {}
            
        logger.info("--- Evaluating on Test Set ---")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(test_loader, desc="Test Evaluation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                if self.config.classification_mode == "binary":
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    all_probs.extend(probs.cpu().numpy().flatten())
                else:  # ternary
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
        avg_test_loss = total_loss / len(test_loader)
        
        # Calculate metrics
        test_metrics = {'test_loss': avg_test_loss}
        
        if self.config.classification_mode == "binary":
            test_metrics.update({
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds, zero_division=0),
                'recall': recall_score(all_labels, all_preds, zero_division=0),
                'f1_score': f1_score(all_labels, all_preds, zero_division=0)
            })
            
            if len(np.unique(all_labels)) > 1:
                test_metrics['auc'] = roc_auc_score(all_labels, all_probs)
            else:
                test_metrics['auc'] = 0.0
                
        else:  # ternary
            test_metrics.update({
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
                'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
                'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
                'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
                'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
                'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            })
        
        # Log test results
        logger.info("Test Set Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def plot_confusion_matrix(self, test_loader: DataLoader, save_path: Optional[Path] = None) -> None:
        """
        Plot confusion matrix for test set.
        UPDATED: Added check for test_loader availability.
        """
        if test_loader is None:
            logger.warning("Cannot plot confusion matrix: test_loader is None")
            return
            
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(test_loader, desc="Computing Confusion Matrix"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                if self.config.classification_mode == "binary":
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                else:  # ternary
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Set up class names
        if self.config.classification_mode == "binary":
            class_names = ['Non-Cancerous', 'Cancerous']
        else:
            class_names = ['Non-Cancerous', 'Cancerous', 'False-Positive']
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.config.classification_mode.title()} Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_history(self, save_path: Optional[Path] = None) -> None:
        """
        Plots the training and validation metrics over epochs.
        UPDATED: Handles train-only mode with limited metrics.
        
        Args:
            save_path (Optional[Path]): If provided, saves the plot to this path.
        """
        if not self.metrics_history or not self.metrics_history.get('train_loss'):
            logger.warning("No metrics history available to plot")
            return
            
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        if self.is_train_only:
            self._plot_train_only_metrics(epochs, save_path)
        elif self.config.classification_mode == "binary":
            self._plot_binary_metrics(epochs, save_path)
        else:
            self._plot_ternary_metrics(epochs, save_path)
    
    def _plot_train_only_metrics(self, epochs: range, save_path: Optional[Path] = None) -> None:
        """Plot metrics for train-only mode (only training loss)."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Training Metrics - Train-Only Mode ({self.config.classification_mode.title()})', fontsize=16)
        
        # Training Loss
        ax.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Train-only metrics plot saved to {save_path}")
        
        plt.show()
    
    def _plot_binary_metrics(self, epochs: range, save_path: Optional[Path] = None) -> None:
        """Plot metrics for binary classification."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Binary Classification Training Metrics', fontsize=16)
        
        # Loss
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 'g-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[0, 2].plot(epochs, self.metrics_history['val_precision'], 'm-', label='Val Precision')
        axes[0, 2].set_title('Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Recall
        axes[1, 0].plot(epochs, self.metrics_history['val_recall'], 'c-', label='Val Recall')
        axes[1, 0].set_title('Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1 Score
        axes[1, 1].plot(epochs, self.metrics_history['val_f1'], 'orange', label='Val F1')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # AUC and Undecided Percentage
        ax_auc = axes[1, 2]
        ax_undecided = ax_auc.twinx()
        
        line1 = ax_auc.plot(epochs, self.metrics_history['val_auc'], 'purple', label='Val AUC')
        line2 = ax_undecided.plot(epochs, [x * 100 for x in self.metrics_history['val_undecided_percentage']], 
                                 'brown', linestyle='--', label='Undecided %')
        
        ax_auc.set_title('AUC and Undecided Percentage')
        ax_auc.set_xlabel('Epoch')
        ax_auc.set_ylabel('AUC', color='purple')
        ax_undecided.set_ylabel('Undecided %', color='brown')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_auc.legend(lines, labels, loc='center right')
        
        ax_auc.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Binary metrics plot saved to {save_path}")
        
        plt.show()
    
    def _plot_ternary_metrics(self, epochs: range, save_path: Optional[Path] = None) -> None:
        """Plot metrics for ternary classification."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Ternary Classification Training Metrics', fontsize=16)
        
        # Loss
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 'g-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision (Macro vs Weighted)
        axes[0, 2].plot(epochs, self.metrics_history['val_precision_macro'], 'm-', label='Precision (Macro)')
        axes[0, 2].plot(epochs, self.metrics_history['val_precision_weighted'], 'm--', label='Precision (Weighted)')
        axes[0, 2].set_title('Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Recall (Macro vs Weighted)
        axes[1, 0].plot(epochs, self.metrics_history['val_recall_macro'], 'c-', label='Recall (Macro)')
        axes[1, 0].plot(epochs, self.metrics_history['val_recall_weighted'], 'c--', label='Recall (Weighted)')
        axes[1, 0].set_title('Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1 Score (Macro vs Weighted)
        axes[1, 1].plot(epochs, self.metrics_history['val_f1_macro'], 'orange', label='F1 (Macro)')
        axes[1, 1].plot(epochs, self.metrics_history['val_f1_weighted'], 'red', label='F1 (Weighted)')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # All metrics combined
        axes[1, 2].plot(epochs, self.metrics_history['val_accuracy'], 'g-', label='Accuracy')
        axes[1, 2].plot(epochs, self.metrics_history['val_f1_macro'], 'orange', label='F1 (Macro)')
        axes[1, 2].plot(epochs, self.metrics_history['val_f1_weighted'], 'red', label='F1 (Weighted)')
        axes[1, 2].set_title('Combined Metrics')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ternary metrics plot saved to {save_path}")
        
        plt.show()