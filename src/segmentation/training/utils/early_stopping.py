"""
Early stopping implementation for training.
"""
import numpy as np
from typing import Optional, Callable, Any


class EarlyStopping:
    """Early stopping utility to stop training when a metric stops improving."""
    
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.inf
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")
    
    def __call__(self,
                 current_score: float,
                 model: Optional[Any] = None,
                 epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_score: Current metric value
            model: Model object (for weight restoration)
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.wait = 0
            
            # Save best weights if model is provided
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_state(model)
            
            if self.verbose:
                print(f"  -> New best score: {current_score:.6f} (improved by {abs(current_score - self.best_score):.6f})")
        else:
            self.wait += 1
            if self.verbose:
                print(f"  -> No improvement for {self.wait} epochs (best: {self.best_score:.6f})")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch if epoch is not None else self.wait
            
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                print(f"Best score was: {self.best_score:.6f}")
            
            # Restore best weights if requested
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                self._restore_model_state(model, self.best_weights)
                if self.verbose:
                    print("Restored best model weights")
            
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
    
    def _get_model_state(self, model):
        """Get model state for restoration."""
        if hasattr(model, 'state_dict'):
            # PyTorch model
            import torch
            return {k: v.clone() for k, v in model.state_dict().items()}
        elif hasattr(model, 'get_weights'):
            # Keras model
            return model.get_weights()
        else:
            # Generic case - try to copy the model
            import copy
            return copy.deepcopy(model)
    
    def _restore_model_state(self, model, weights):
        """Restore model state."""
        if hasattr(model, 'load_state_dict'):
            # PyTorch model
            model.load_state_dict(weights)
        elif hasattr(model, 'set_weights'):
            # Keras model
            model.set_weights(weights)
        else:
            # This won't work for all cases, but it's a fallback
            print("Warning: Could not restore model weights - unsupported model type")
    
    def get_info(self):
        """Get information about early stopping state."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'wait': self.wait,
            'best_score': self.best_score,
            'stopped_epoch': self.stopped_epoch,
            'should_stop': self.wait >= self.patience
        }


class AdaptiveEarlyStopping(EarlyStopping):
    """
    Early stopping with adaptive patience based on training progress.
    """
    
    def __init__(self,
                 base_patience: int = 15,
                 max_patience: int = 30,
                 patience_factor: float = 1.5,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize adaptive early stopping.
        
        Args:
            base_patience: Base patience value
            max_patience: Maximum patience value
            patience_factor: Factor to increase patience when improvement is slow
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best weights
            verbose: Whether to print messages
        """
        super().__init__(base_patience, min_delta, mode, restore_best_weights, verbose)
        
        self.base_patience = base_patience
        self.max_patience = max_patience
        self.patience_factor = patience_factor
        self.improvement_history = []
        
    def __call__(self,
                 current_score: float,
                 model: Optional[Any] = None,
                 epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped with adaptive patience.
        
        Args:
            current_score: Current metric value
            model: Model object
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        # Track improvement
        if len(self.improvement_history) > 0:
            improvement = abs(current_score - self.improvement_history[-1])
            self.improvement_history.append(current_score)
        else:
            self.improvement_history.append(current_score)
            improvement = float('inf')
        
        # Adapt patience based on recent improvements
        if len(self.improvement_history) >= 5:
            recent_improvements = [
                abs(self.improvement_history[i] - self.improvement_history[i-1])
                for i in range(-4, 0)
            ]
            avg_improvement = np.mean(recent_improvements)
            
            # If improvements are getting smaller, increase patience
            if avg_improvement < self.min_delta * 2:
                new_patience = min(int(self.base_patience * self.patience_factor), self.max_patience)
                if new_patience != self.patience:
                    self.patience = new_patience
                    if self.verbose:
                        print(f"  -> Adapted patience to {self.patience} due to slow improvement")
        
        # Use parent class logic
        return super().__call__(current_score, model, epoch)
    
    def reset(self):
        """Reset adaptive early stopping state."""
        super().reset()
        self.patience = self.base_patience
        self.improvement_history = []


class MultiMetricEarlyStopping:
    """
    Early stopping that considers multiple metrics.
    """
    
    def __init__(self,
                 metrics_config: dict,
                 patience: int = 15,
                 require_all: bool = False,
                 verbose: bool = True):
        """
        Initialize multi-metric early stopping.
        
        Args:
            metrics_config: Dict mapping metric names to {'mode': 'min'/'max', 'weight': float}
            patience: Number of epochs to wait
            require_all: If True, all metrics must stop improving; if False, any metric can trigger
            verbose: Whether to print messages
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.require_all = require_all
        self.verbose = verbose
        
        # Create individual early stopping for each metric
        self.stoppers = {}
        for metric_name, config in metrics_config.items():
            self.stoppers[metric_name] = EarlyStopping(
                patience=patience,
                mode=config['mode'],
                verbose=False  # We'll handle verbose output here
            )
        
        self.best_combined_score = float('inf') if not require_all else float('-inf')
        self.wait = 0
        
    def __call__(self,
                 metrics: dict,
                 model: Optional[Any] = None,
                 epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped based on multiple metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            model: Model object
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        # Check each metric
        should_stop_per_metric = {}
        for metric_name, stopper in self.stoppers.items():
            if metric_name in metrics:
                should_stop_per_metric[metric_name] = stopper(
                    metrics[metric_name], model, epoch
                )
            else:
                should_stop_per_metric[metric_name] = False
        
        # Calculate combined score
        combined_score = 0
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_config:
                weight = self.metrics_config[metric_name].get('weight', 1.0)
                if self.metrics_config[metric_name]['mode'] == 'max':
                    combined_score += value * weight
                else:
                    combined_score -= value * weight
        
        # Check if combined score improved
        improved = False
        if self.require_all:
            # For 'require_all', we want the combined score to increase
            if combined_score > self.best_combined_score:
                self.best_combined_score = combined_score
                improved = True
        else:
            # For 'any', we want at least one metric to improve
            improved = any(not should_stop for should_stop in should_stop_per_metric.values())
        
        if improved:
            self.wait = 0
        else:
            self.wait += 1
        
        # Determine if we should stop
        if self.require_all:
            # Stop only if ALL metrics have stopped improving
            should_stop = all(should_stop_per_metric.values())
        else:
            # Stop if ANY metric has stopped improving for too long
            should_stop = self.wait >= self.patience
        
        if self.verbose:
            print(f"  -> Multi-metric early stopping: wait={self.wait}/{self.patience}")
            for metric_name, value in metrics.items():
                if metric_name in should_stop_per_metric:
                    status = "✓" if not should_stop_per_metric[metric_name] else "✗"
                    print(f"    {metric_name}: {value:.6f} {status}")
        
        return should_stop
    
    def reset(self):
        """Reset all early stopping states."""
        for stopper in self.stoppers.values():
            stopper.reset()
        self.wait = 0
        self.best_combined_score = float('inf') if not self.require_all else float('-inf')