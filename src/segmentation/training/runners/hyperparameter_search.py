"""Hyperparameter optimization for training experiments."""

import json
import logging
import itertools
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .experiment_runner import ExperimentRunner, ExperimentConfig
from ..config.stage1_config import Stage1Config
from ..config.stage2_config import Stage2Config


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    
    # Stage 1 parameters
    stage1_learning_rates: List[float] = None
    stage1_total_epochs: List[int] = None
    stage1_types: List[str] = None
    ce_loss_weights: List[float] = None
    bbox_loss_weights: List[float] = None
    giou_loss_weights: List[float] = None
    
    # Stage 2 parameters
    stage2_learning_rates: List[float] = None
    stage2_total_epochs: List[int] = None
    focal_loss_weights: List[float] = None
    focal_gammas: List[float] = None
    dice_loss_weights: List[float] = None
    boundary_loss_weights: List[float] = None
    
    # Transform parameters
    gammas: List[float] = None
    eosin_values: List[float] = None
    dab_values: List[float] = None
    
    # Training parameters
    batch_sizes: List[int] = None
    augmentation_flags: List[bool] = None
    complex_augmentation_flags: List[bool] = None
    
    def __post_init__(self):
        """Set default values for None parameters."""
        if self.stage1_learning_rates is None:
            self.stage1_learning_rates = [1e-4, 1e-5, 5e-5]
        if self.stage1_total_epochs is None:
            self.stage1_total_epochs = [300, 500, 700]
        if self.stage1_types is None:
            self.stage1_types = ['cellfinder', 'image_encoder', 'all_backbones']
        if self.ce_loss_weights is None:
            self.ce_loss_weights = [1, 2]
        if self.bbox_loss_weights is None:
            self.bbox_loss_weights = [5, 10]
        if self.giou_loss_weights is None:
            self.giou_loss_weights = [2, 5]
            
        if self.stage2_learning_rates is None:
            self.stage2_learning_rates = [1e-4, 1e-5, 5e-5]
        if self.stage2_total_epochs is None:
            self.stage2_total_epochs = [300, 500, 700]
        if self.focal_loss_weights is None:
            self.focal_loss_weights = [1.0, 2.0]
        if self.focal_gammas is None:
            self.focal_gammas = [2.0, 3.0]
        if self.dice_loss_weights is None:
            self.dice_loss_weights = [1.0, 2.0]
        if self.boundary_loss_weights is None:
            self.boundary_loss_weights = [0.0, 1.0]
            
        if self.gammas is None:
            self.gammas = [1.8, 2.1, 2.4]
        if self.eosin_values is None:
            self.eosin_values = [0.0, 0.1]
        if self.dab_values is None:
            self.dab_values = [0.0, 0.1]
            
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2]
        if self.augmentation_flags is None:
            self.augmentation_flags = [False, True]
        if self.complex_augmentation_flags is None:
            self.complex_augmentation_flags = [False, True]


@dataclass
class HyperparameterConfig:
    """Single hyperparameter configuration."""
    
    # Stage 1 parameters
    stage1_lr: float
    stage1_epochs: int
    stage1_type: str
    ce_weight: float
    bbox_weight: float
    giou_weight: float
    
    # Stage 2 parameters
    stage2_lr: float
    stage2_epochs: int
    focal_weight: float
    focal_gamma: float
    dice_weight: float
    boundary_weight: float
    
    # Transform parameters
    gamma: float
    eosin: float
    dab: float
    
    # Training parameters
    batch_size: int
    augmentation: bool
    complex_augmentation: bool
    
    def to_experiment_config(self, base_config: ExperimentConfig) -> ExperimentConfig:
        """Convert to ExperimentConfig."""
        # Create stage configs
        stage1_config = Stage1Config(
            stage1_type=self.stage1_type,
            total_epochs=self.stage1_epochs,
            learning_rate=self.stage1_lr,
            ce_loss_weight=self.ce_weight,
            bbox_loss_weight=self.bbox_weight,
            giou_loss_weight=self.giou_weight
        )
        
        stage2_config = Stage2Config(
            total_epochs=self.stage2_epochs,
            learning_rate=self.stage2_lr,
            focal_loss_weight=self.focal_weight,
            focal_gamma=self.focal_gamma,
            dice_loss_weight=self.dice_weight,
            boundary_loss_weight=self.boundary_weight
        )
        
        # Update transform parameters
        transform_params = base_config.transform_params.copy()
        transform_params.update({
            'gamma': self.gamma,
            'eosin': self.eosin,
            'dab': self.dab,
            'augmentation': self.augmentation,
            'complex_augmentation': self.complex_augmentation
        })
        
        # Create new experiment config
        config = ExperimentConfig(
            img_path=base_config.img_path,
            mask_path=base_config.mask_path,
            output_name=f"{base_config.output_name}_hp_{id(self)}",
            strategy=base_config.strategy,
            k_folds=base_config.k_folds,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            batch_size=self.batch_size,
            debug=base_config.debug,
            preeval=base_config.preeval,
            test=base_config.test,
            multiple_datasets=base_config.multiple_datasets,
            additional_datasets=base_config.additional_datasets
        )
        
        config.transform_params = transform_params
        return config


class HyperparameterOptimizer:
    """Hyperparameter optimization orchestrator."""
    
    def __init__(self, 
                 base_config: ExperimentConfig,
                 param_space: HyperparameterSpace,
                 output_dir: str,
                 max_parallel: int = 1,
                 metric_name: str = "best_f1",
                 metric_key: str = "f1"):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            base_config: Base experiment configuration.
            param_space: Hyperparameter search space.
            output_dir: Directory to save optimization results.
            max_parallel: Maximum parallel experiments.
            metric_name: Name of the metric category to optimize.
            metric_key: Specific metric key to optimize.
        """
        self.base_config = base_config
        self.param_space = param_space
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel
        self.metric_name = metric_name
        self.metric_key = metric_key
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.results = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hyperparameter search."""
        logger = logging.getLogger("hyperparameter_optimizer")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.output_dir / "hyperparameter_search.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def generate_grid_search_configs(self) -> List[HyperparameterConfig]:
        """Generate all combinations for grid search."""
        # Get all parameter combinations
        param_names = []
        param_values = []
        
        for field_name, field_value in asdict(self.param_space).items():
            if field_value is not None:
                param_names.append(field_name)
                param_values.append(field_value)
        
        # Generate all combinations
        configs = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # Map parameter names to HyperparameterConfig fields
            config = HyperparameterConfig(
                stage1_lr=param_dict.get('stage1_learning_rates', 1e-4),
                stage1_epochs=param_dict.get('stage1_total_epochs', 500),
                stage1_type=param_dict.get('stage1_types', 'cellfinder'),
                ce_weight=param_dict.get('ce_loss_weights', 1),
                bbox_weight=param_dict.get('bbox_loss_weights', 5),
                giou_weight=param_dict.get('giou_loss_weights', 2),
                
                stage2_lr=param_dict.get('stage2_learning_rates', 1e-4),
                stage2_epochs=param_dict.get('stage2_total_epochs', 500),
                focal_weight=param_dict.get('focal_loss_weights', 1.0),
                focal_gamma=param_dict.get('focal_gammas', 2.0),
                dice_weight=param_dict.get('dice_loss_weights', 1.0),
                boundary_weight=param_dict.get('boundary_loss_weights', 1.0),
                
                gamma=param_dict.get('gammas', 2.1),
                eosin=param_dict.get('eosin_values', 0.0),
                dab=param_dict.get('dab_values', 0.0),
                
                batch_size=param_dict.get('batch_sizes', 1),
                augmentation=param_dict.get('augmentation_flags', False),
                complex_augmentation=param_dict.get('complex_augmentation_flags', False)
            )
            
            configs.append(config)
        
        self.logger.info(f"Generated {len(configs)} configurations for grid search")
        return configs
    
    def generate_random_search_configs(self, n_trials: int) -> List[HyperparameterConfig]:
        """Generate random configurations for random search."""
        configs = []
        
        for _ in range(n_trials):
            config = HyperparameterConfig(
                stage1_lr=random.choice(self.param_space.stage1_learning_rates),
                stage1_epochs=random.choice(self.param_space.stage1_total_epochs),
                stage1_type=random.choice(self.param_space.stage1_types),
                ce_weight=random.choice(self.param_space.ce_loss_weights),
                bbox_weight=random.choice(self.param_space.bbox_loss_weights),
                giou_weight=random.choice(self.param_space.giou_loss_weights),
                
                stage2_lr=random.choice(self.param_space.stage2_learning_rates),
                stage2_epochs=random.choice(self.param_space.stage2_total_epochs),
                focal_weight=random.choice(self.param_space.focal_loss_weights),
                focal_gamma=random.choice(self.param_space.focal_gammas),
                dice_weight=random.choice(self.param_space.dice_loss_weights),
                boundary_weight=random.choice(self.param_space.boundary_loss_weights),
                
                gamma=random.choice(self.param_space.gammas),
                eosin=random.choice(self.param_space.eosin_values),
                dab=random.choice(self.param_space.dab_values),
                
                batch_size=random.choice(self.param_space.batch_sizes),
                augmentation=random.choice(self.param_space.augmentation_flags),
                complex_augmentation=random.choice(self.param_space.complex_augmentation_flags)
            )
            
            configs.append(config)
        
        self.logger.info(f"Generated {len(configs)} configurations for random search")
        return configs
    
    def _run_single_experiment(self, hp_config: HyperparameterConfig, trial_id: int) -> Dict:
        """Run a single experiment with given hyperparameters."""
        try:
            # Convert to experiment config
            exp_config = hp_config.to_experiment_config(self.base_config)
            exp_config.output_name = f"{self.base_config.output_name}_trial_{trial_id}"
            
            # Run experiment
            runner = ExperimentRunner(exp_config)
            results = runner.run()
            
            # Extract metric value for optimization
            metric_value = self._extract_metric_value(results)
            
            return {
                'trial_id': trial_id,
                'hyperparameters': asdict(hp_config),
                'results': results,
                'metric_value': metric_value,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Trial {trial_id} failed: {str(e)}")
            return {
                'trial_id': trial_id,
                'hyperparameters': asdict(hp_config),
                'results': None,
                'metric_value': float('-inf'),
                'success': False,
                'error': str(e)
            }
    
    def _extract_metric_value(self, results: List[Dict]) -> float:
        """Extract the metric value to optimize."""
        if not results:
            return float('-inf')
        
        # For k-fold results, average across folds
        if len(results) > 1:
            values = []
            for result in results:
                if self.metric_name in result and self.metric_key in result[self.metric_name]:
                    metric_data = result[self.metric_name][self.metric_key]
                    if isinstance(metric_data, list):
                        values.append(np.mean(metric_data))
                    else:
                        values.append(metric_data)
            
            return np.mean(values) if values else float('-inf')
        else:
            # Single result
            result = results[0]
            if self.metric_name in result and self.metric_key in result[self.metric_name]:
                metric_data = result[self.metric_name][self.metric_key]
                if isinstance(metric_data, list):
                    return np.mean(metric_data)
                else:
                    return metric_data
            
            return float('-inf')
    
    def run_grid_search(self) -> Dict:
        """Run grid search optimization."""
        self.logger.info("Starting grid search optimization")
        
        configs = self.generate_grid_search_configs()
        return self._run_optimization(configs)
    
    def run_random_search(self, n_trials: int) -> Dict:
        """Run random search optimization."""
        self.logger.info(f"Starting random search optimization with {n_trials} trials")
        
        configs = self.generate_random_search_configs(n_trials)
        return self._run_optimization(configs)
    
    def _run_optimization(self, configs: List[HyperparameterConfig]) -> Dict:
        """Run optimization with given configurations."""
        self.results = []
        
        if self.max_parallel > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
                futures = {
                    executor.submit(self._run_single_experiment, config, i): i 
                    for i, config in enumerate(configs)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    trial_id = result['trial_id']
                    
                    if result['success']:
                        self.logger.info(f"Trial {trial_id} completed - "
                                       f"Metric: {result['metric_value']:.4f}")
                    else:
                        self.logger.error(f"Trial {trial_id} failed")
        else:
            # Sequential execution
            for i, config in enumerate(configs):
                self.logger.info(f"Running trial {i+1}/{len(configs)}")
                result = self._run_single_experiment(config, i)
                self.results.append(result)
                
                if result['success']:
                    self.logger.info(f"Trial {i} completed - "
                                   f"Metric: {result['metric_value']:.4f}")
        
        # Find best configuration
        best_result = max(self.results, key=lambda x: x['metric_value'])
        
        optimization_results = {
            'best_trial': best_result,
            'all_trials': self.results,
            'n_trials': len(configs),
            'metric_optimized': f"{self.metric_name}.{self.metric_key}",
            'best_metric_value': best_result['metric_value']
        }
        
        # Save results
        self._save_optimization_results(optimization_results)
        
        self.logger.info(f"Optimization completed. Best metric value: "
                        f"{best_result['metric_value']:.4f}")
        
        return optimization_results
    
    def _save_optimization_results(self, results: Dict) -> None:
        """Save optimization results."""
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best configuration separately
        best_config_file = self.output_dir / "best_configuration.json"
        with open(best_config_file, 'w') as f:
            json.dump(results['best_trial']['hyperparameters'], f, indent=2)
        
        self.logger.info(f"Optimization results saved to {results_file}")
        self.logger.info(f"Best configuration saved to {best_config_file}")


# Convenience functions
def run_grid_search(base_config: ExperimentConfig,
                   param_space: HyperparameterSpace,
                   output_dir: str,
                   max_parallel: int = 1) -> Dict:
    """Run grid search hyperparameter optimization."""
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        param_space=param_space,
        output_dir=output_dir,
        max_parallel=max_parallel
    )
    
    return optimizer.run_grid_search()


def run_random_search(base_config: ExperimentConfig,
                     param_space: HyperparameterSpace,
                     output_dir: str,
                     n_trials: int,
                     max_parallel: int = 1) -> Dict:
    """Run random search hyperparameter optimization."""
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        param_space=param_space,
        output_dir=output_dir,
        max_parallel=max_parallel
    )
    
    return optimizer.run_random_search(n_trials)


def create_default_param_space(focused: bool = False) -> HyperparameterSpace:
    """
    Create a default hyperparameter space.
    
    Args:
        focused: If True, creates a smaller, more focused search space.
        
    Returns:
        HyperparameterSpace instance.
    """
    if focused:
        return HyperparameterSpace(
            stage1_learning_rates=[1e-4, 1e-5],
            stage2_learning_rates=[1e-4, 1e-5],
            focal_gammas=[2.0, 3.0],
            gammas=[2.1, 2.4],
            augmentation_flags=[False, True]
        )
    else:
        return HyperparameterSpace()  # Uses all defaults


# Example usage function
def optimize_hyperparameters_example():
    """Example of how to run hyperparameter optimization."""
    from .experiment_runner import ExperimentConfig
    
    # Create base configuration
    base_config = ExperimentConfig(
        img_path="path/to/images",
        mask_path="path/to/masks",
        output_name="hp_optimization",
        strategy="k_fold",
        k_folds=3  # Smaller for faster optimization
    )
    
    # Create parameter space
    param_space = create_default_param_space(focused=True)
    
    # Run optimization
    results = run_random_search(
        base_config=base_config,
        param_space=param_space,
        output_dir="../output/hyperparameter_search",
        n_trials=10,
        max_parallel=2
    )
    
    return results