"""
Early stopping implementation for training.
"""
import numpy as np
from typing import Optional, Any
import logging


class EarlyStopping:
    """Early stopping utility to stop training when a metric stops improving."""
    
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.verbose = verbose
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.should_stop_flag = False
        
        # Set up comparison function based on mode
        if mode == 'min':
            self.is_better = self._is_better_min
        elif mode == 'max':
            self.is_better = self._is_better_max
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _is_better_min(self, current: float, best: float) -> bool:
        """Check if current score is better for minimization."""
        return current < (best - self.min_delta)
    
    def _is_better_max(self, current: float, best: float) -> bool:
        """Check if current score is better for maximization."""
        return current > (best + self.min_delta)
    
    def update(self, current_score: float, epoch: Optional[int] = None) -> None:
        """
        Update early stopping state with current score.
        
        Args:
            current_score: Current metric value
            epoch: Current epoch number (for logging)
        """
        if self.best_score is None:
            # First epoch
            self.best_score = current_score
            self.wait = 0
            if self.verbose:
                self.logger.info(f"Initial best score: {current_score:.6f}")
        elif self.is_better(current_score, self.best_score):
            # Improvement found
            improvement = abs(current_score - self.best_score)
            self.best_score = current_score
            self.wait = 0
            if self.verbose:
                self.logger.info(f"New best score: {current_score:.6f} (improved by {improvement:.6f})")
        else:
            # No improvement
            self.wait += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement for {self.wait}/{self.patience} epochs "
                    f"(current: {current_score:.6f}, best: {self.best_score:.6f})"
                )
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.should_stop_flag = True
            self.stopped_epoch = epoch if epoch is not None else self.wait
            if self.verbose:
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs without improvement. "
                    f"Best score: {self.best_score:.6f}"
                )
    
    def should_stop(self, current_score: Optional[float] = None, epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_score: Current metric value (if provided, will update state)
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if current_score is not None:
            self.update(current_score, epoch)
        
        return self.should_stop_flag
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.should_stop_flag = False
        
        if self.verbose:
            self.logger.info("Early stopping state reset")
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score seen so far."""
        return self.best_score
    
    def get_wait_count(self) -> int:
        """Get current wait count."""
        return self.wait
    
    def get_info(self) -> dict:
        """Get information about early stopping state."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'wait': self.wait,
            'best_score': self.best_score,
            'stopped_epoch': self.stopped_epoch,
            'should_stop': self.should_stop_flag
        }


class AdaptiveEarlyStopping(EarlyStopping):
    """
    Early stopping with adaptive patience based on training progress.
    Increases patience when improvements are getting smaller.
    """
    
    def __init__(self,
                 base_patience: int = 15,
                 max_patience: int = 50,
                 patience_factor: float = 1.5,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Initialize adaptive early stopping.
        
        Args:
            base_patience: Base patience value
            max_patience: Maximum patience value
            patience_factor: Factor to increase patience when improvement is slow
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        super().__init__(base_patience, min_delta, mode, verbose)
        
        self.base_patience = base_patience
        self.max_patience = max_patience
        self.patience_factor = patience_factor
        self.score_history = []
        self.patience_adapted = False
        
    def update(self, current_score: float, epoch: Optional[int] = None) -> None:
        """
        Update adaptive early stopping state.
        
        Args:
            current_score: Current metric value
            epoch: Current epoch number
        """
        # Store score history
        self.score_history.append(current_score)
        
        # Adapt patience if we have enough history and haven't adapted yet
        if len(self.score_history) >= 10 and not self.patience_adapted and self.wait > 0:
            self._adapt_patience()
        
        # Call parent update
        super().update(current_score, epoch)
    
    def _adapt_patience(self) -> None:
        """Adapt patience based on recent score improvements."""
        if len(self.score_history) < 10:
            return
        
        # Calculate recent improvements
        recent_scores = self.score_history[-10:]
        improvements = []
        
        for i in range(1, len(recent_scores)):
            if self.mode == 'min':
                improvement = recent_scores[i-1] - recent_scores[i]  # Positive if score decreased
            else:
                improvement = recent_scores[i] - recent_scores[i-1]  # Positive if score increased
            improvements.append(max(0, improvement))  # Only consider positive improvements
        
        avg_improvement = np.mean(improvements)
        
        # If improvements are getting very small, increase patience
        if avg_improvement < self.min_delta * 0.5:
            old_patience = self.patience
            new_patience = min(int(self.patience * self.patience_factor), self.max_patience)
            
            if new_patience > old_patience:
                self.patience = new_patience
                self.patience_adapted = True
                if self.verbose:
                    self.logger.info(
                        f"Adapted patience from {old_patience} to {new_patience} "
                        f"due to slow improvement (avg: {avg_improvement:.8f})"
                    )
    
    def reset(self) -> None:
        """Reset adaptive early stopping state."""
        super().reset()
        self.patience = self.base_patience
        self.score_history = []
        self.patience_adapted = False


class MultiMetricEarlyStopping:
    """
    Early stopping that considers multiple metrics.
    Useful when you want to monitor multiple validation metrics simultaneously.
    """
    
    def __init__(self,
                 metrics_config: dict,
                 patience: int = 15,
                 combination_mode: str = 'any',
                 verbose: bool = True):
        """
        Initialize multi-metric early stopping.
        
        Args:
            metrics_config: Dict mapping metric names to {'mode': 'min'/'max', 'weight': float}
                          Example: {'loss': {'mode': 'min', 'weight': 1.0}, 
                                   'accuracy': {'mode': 'max', 'weight': 0.5}}
            patience: Number of epochs to wait
            combination_mode: 'any' (stop if any metric stops improving) or 
                            'all' (stop only if all metrics stop improving) or
                            'weighted' (use weighted combination of metrics)
            verbose: Whether to print messages
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.combination_mode = combination_mode
        self.verbose = verbose
        
        # Create individual early stopping for each metric
        self.stoppers = {}
        for metric_name, config in metrics_config.items():
            self.stoppers[metric_name] = EarlyStopping(
                patience=patience,
                mode=config['mode'],
                verbose=False  # We handle verbose output centrally
            )
        
        # For weighted combination
        if combination_mode == 'weighted':
            self.combined_stopper = EarlyStopping(
                patience=patience,
                mode='max',  # We want to maximize the weighted score
                verbose=False
            )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def should_stop(self, metrics: dict, epoch: Optional[int] = None) -> bool:
        """
        Check if training should be stopped based on multiple metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        # Update each metric's early stopping
        should_stop_per_metric = {}
        for metric_name, stopper in self.stoppers.items():
            if metric_name in metrics:
                should_stop_per_metric[metric_name] = stopper.should_stop(
                    metrics[metric_name], epoch
                )
            else:
                # Missing metric - treat as no improvement
                should_stop_per_metric[metric_name] = stopper.should_stop(None, epoch)
        
        # Determine overall stopping decision
        if self.combination_mode == 'any':
            should_stop = any(should_stop_per_metric.values())
        elif self.combination_mode == 'all':
            should_stop = all(should_stop_per_metric.values())
        elif self.combination_mode == 'weighted':
            # Calculate weighted score
            weighted_score = 0
            total_weight = 0
            for metric_name, value in metrics.items():
                if metric_name in self.metrics_config:
                    config = self.metrics_config[metric_name]
                    weight = config.get('weight', 1.0)
                    
                    # Normalize score based on mode
                    if config['mode'] == 'min':
                        # For minimization metrics, use negative value
                        normalized_value = -value
                    else:
                        # For maximization metrics, use positive value
                        normalized_value = value
                    
                    weighted_score += normalized_value * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            should_stop = self.combined_stopper.should_stop(weighted_score, epoch)
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")
        
        # Logging
        if self.verbose:
            status_strings = []
            for metric_name, value in metrics.items():
                if metric_name in should_stop_per_metric:
                    status = "STOP" if should_stop_per_metric[metric_name] else "OK"
                    wait = self.stoppers[metric_name].get_wait_count()
                    status_strings.append(f"{metric_name}: {value:.6f} ({wait}/{self.patience}) [{status}]")
            
            self.logger.info(f"Multi-metric early stopping ({self.combination_mode}): {', '.join(status_strings)}")
            
            if should_stop:
                self.logger.info("Multi-metric early stopping triggered!")
        
        return should_stop
    
    def reset(self) -> None:
        """Reset all early stopping states."""
        for stopper in self.stoppers.values():
            stopper.reset()
        
        if hasattr(self, 'combined_stopper'):
            self.combined_stopper.reset()
        
        if self.verbose:
            self.logger.info("Multi-metric early stopping state reset")
    
    def get_best_scores(self) -> dict:
        """Get best scores for all metrics."""
        return {
            name: stopper.get_best_score() 
            for name, stopper in self.stoppers.items()
        }
    
    def get_info(self) -> dict:
        """Get information about all early stopping states."""
        info = {
            'combination_mode': self.combination_mode,
            'patience': self.patience,
            'metrics': {}
        }
        
        for name, stopper in self.stoppers.items():
            info['metrics'][name] = stopper.get_info()
        
        return info