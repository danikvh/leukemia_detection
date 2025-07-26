import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional
import matplotlib.pyplot as plt

from classification.config import ClassificationConfig

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Trainer class for cell classification models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ClassificationConfig,
        pos_weight: Optional[float] = None # For BCEWithLogitsLoss to handle imbalance
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
        else:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")
            
        # Early stopping parameters
        self.patience = config.patience
        self.early_stopping_metric = config.early_stopping_metric

        # Initialize best score based on metric type
        if self.early_stopping_metric == "val_loss":
            self.best_early_stopping_score = float('inf') # Lower is better
            self.is_better = lambda current, best: current < best
        else: # For accuracy, precision, recall, f1, auc, undecided_percentage (higher is better for these)
            self.best_early_stopping_score = -float('inf') # Higher is better
            self.is_better = lambda current, best: current > best
        
        self.epochs_no_improve = 0 # Reset epochs no improve counter

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
        all_preds_binary = [] 
        all_labels = []
        all_probs = [] 
        
        undecided_count = 0 

        with torch.no_grad():
            for inputs, labels, _ in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds_binary = (probs > 0.5).long() 

                all_preds_binary.extend(preds_binary.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

                for prob in probs.cpu().numpy().flatten():
                    if self.config.confidence_threshold_low < prob < self.config.confidence_threshold_high:
                        undecided_count += 1

        avg_loss = total_loss / len(self.val_loader)
        
        accuracy = accuracy_score(all_labels, all_preds_binary)
        precision = precision_score(all_labels, all_preds_binary, zero_division=0)
        
        if len(all_labels) == 0:
            recall = 0.0
            logger.warning("Recall not calculated: No true labels found in validation set.")
        else:
            recall = recall_score(all_labels, all_preds_binary, zero_division=0)

        f1 = f1_score(all_labels, all_preds_binary, zero_division=0)
        
        auc = 0.0
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            logger.warning("AUC not calculated: Validation set contains only one class.")

        total_val_samples = len(all_labels)
        undecided_percentage = (undecided_count / total_val_samples) if total_val_samples > 0 else 0.0

        # CORRECTED: Prefix keys with 'val_' to match self.metrics_history and config.early_stopping_metric
        return {
            'val_loss': avg_loss, # Changed from 'loss'
            'val_accuracy': accuracy, # Changed from 'accuracy'
            'val_precision': precision, # Changed from 'precision'
            'val_recall': recall, # Changed from 'recall'
            'val_f1': f1, # Changed from 'f1_score'
            'val_auc': auc, # Changed from 'auc'
            'val_undecided_percentage': undecided_percentage
        }

    def train(self, output_dir: Path):
        """
        Main training loop.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {self.config.epochs} epochs on {self.device}")
        logger.info(f"Early stopping monitor: {self.early_stopping_metric} with patience {self.patience}")

        model_save_path = output_dir / "best_classification_model.pth"
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate_epoch() # This now returns keys with 'val_' prefix

            # These keys now correctly match the names returned by _validate_epoch
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['val_accuracy'].append(val_metrics['val_accuracy'])
            self.metrics_history['val_precision'].append(val_metrics['val_precision'])
            self.metrics_history['val_recall'].append(val_metrics['val_recall'])
            self.metrics_history['val_f1'].append(val_metrics['val_f1'])
            self.metrics_history['val_auc'].append(val_metrics['val_auc'])
            self.metrics_history['val_undecided_percentage'].append(val_metrics['val_undecided_percentage'])

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, " # Corrected key
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, " # Corrected key
                f"Val F1: {val_metrics['val_f1']:.4f}, " # Corrected key
                f"Val AUC: {val_metrics['val_auc']:.4f}, " # Corrected key
                f"Val Undecided: {val_metrics['val_undecided_percentage']:.2%}" # Corrected key
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

            history_path = output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement in {self.early_stopping_metric}.")
                break
        
        logger.info("Training complete.")
    
    def plot_metrics_history(self, save_path: Optional[Path] = None) -> None:
        """
        Plots the training and validation loss, and validation metrics over epochs.
        
        Args:
            save_path (Optional[Path]): If provided, saves the plot to this path.
        """
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(15, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.metrics_history['train_loss'], label='Train Loss', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_loss'], label='Validation Loss', marker='o', markersize=4)
        plt.title('Loss over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True)

        # Plot Metrics
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.metrics_history['val_accuracy'], label='Val Accuracy', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_precision'], label='Val Precision', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_recall'], label='Val Recall', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_f1'], label='Val F1-Score', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_auc'], label='Val AUC', marker='o', markersize=4)
        plt.plot(epochs, self.metrics_history['val_undecided_percentage'], label='Val Undecided %', marker='x', linestyle='--', markersize=4) 
        
        plt.title('Validation Metrics over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score / Percentage', fontsize=12) 
        plt.ylim(0, 1.05) 
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved metrics plot to {save_path}")
        plt.show()