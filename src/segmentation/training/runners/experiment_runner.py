"""Main experiment orchestrator for training runs."""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from ..config.stage1_config import Stage1Config
from ..config.stage2_config import Stage2Config
from ..core.multi_stage_trainer import MultiStageTrainer
from ..strategies.training_strategy import TrainingStrategy
from ..strategies.full_dataset import FullDatasetStrategy
from ..strategies.train_val_split import TrainValSplitStrategy
from ..strategies.k_fold import KFoldStrategy
from ..utils.metrics_tracker import FoldMetricsAggregator
from ..utils.gpu_monitor import GPUMonitor
from ..utils.visualization import create_training_visualizer

from ...datasets.dataset_factory import get_datasets
from ...transforms.composed_transforms import FullTransform


class ExperimentConfig:
    """Configuration for a complete experiment."""
    
    def __init__(self,
                 # Data configuration
                 img_path: str,
                 mask_path: str,
                 output_name: str,
                 
                 # Training strategy
                 strategy: str = "k_fold",  # "full_dataset", "train_val_split", "k_fold"
                 k_folds: int = 5,
                 
                 # Stage configurations
                 stage1_config: Optional[Stage1Config] = None,
                 stage2_config: Optional[Stage2Config] = None,
                 
                 # General training parameters
                 batch_size: int = 1,
                 debug: bool = False,
                 preeval: bool = False,
                 test: bool = False,
                 
                 # Transform parameters
                 normalize: bool = False,
                 rgb_transform: bool = False,
                 stain_transform: bool = True,
                 eosin: float = 0.0,
                 dab: float = 0.0,
                 inversion: bool = True,
                 only_nuclei: bool = True,
                 gamma: float = 2.1,
                 normalize_inf: bool = True,
                 augmentation: bool = False,
                 complex_augmentation: bool = False,
                 
                 # Multi-dataset training
                 multiple_datasets: bool = False,
                 additional_datasets: Optional[List[Dict[str, str]]] = None):
        
        self.img_path = img_path
        self.mask_path = mask_path
        self.output_name = output_name
        self.strategy = strategy
        self.k_folds = k_folds
        
        # Use provided configs or create defaults
        self.stage1_config = stage1_config or Stage1Config()
        self.stage2_config = stage2_config or Stage2Config()
        
        self.batch_size = batch_size
        self.debug = debug
        self.preeval = preeval
        self.test = test
        
        # Transform parameters
        self.transform_params = {
            'normalize': normalize,
            'rgb_transform': rgb_transform,
            'stain_transform': stain_transform,
            'eosin': eosin,
            'dab': dab,
            'inversion': inversion,
            'only_nuclei': only_nuclei,
            'gamma': gamma,
            'normalize_inf': normalize_inf,
            'augmentation': augmentation,
            'complex_augmentation': complex_augmentation
        }
        
        self.multiple_datasets = multiple_datasets
        self.additional_datasets = additional_datasets or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'img_path': self.img_path,
            'mask_path': self.mask_path,
            'output_name': self.output_name,
            'strategy': self.strategy,
            'k_folds': self.k_folds,
            'stage1_config': asdict(self.stage1_config),
            'stage2_config': asdict(self.stage2_config),
            'batch_size': self.batch_size,
            'debug': self.debug,
            'preeval': self.preeval,
            'test': self.test,
            'transform_params': self.transform_params,
            'multiple_datasets': self.multiple_datasets,
            'additional_datasets': self.additional_datasets
        }


class ExperimentRunner:
    """Main experiment runner that orchestrates training."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Setup output directories
        self.output_dir = Path("../output") / config.output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU monitoring
        self.gpu_monitor = GPUMonitor()
        
        # Results aggregator for k-fold
        self.results_aggregator = FoldMetricsAggregator(
            save_dir=str(self.output_dir / "results")
        )
        
        self.logger.info(f"Initialized experiment: {config.output_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        log_dir = Path("../output") / self.config.output_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(f"experiment_{self.config.output_name}")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_datasets(self) -> tuple:
        """Create training and validation datasets."""
        # Create transform
        transform = FullTransform(**self.config.transform_params)
        
        # Create main dataset
        train_dataset = get_datasets(
            img_folder_path=self.config.img_path,
            mask_folder_path=self.config.mask_path,
            transform=transform,
            do_augmentation=self.config.transform_params['augmentation'],
            complex_augmentation=self.config.transform_params['complex_augmentation']
        )
        
        # Create validation dataset (without augmentation)
        val_transform = FullTransform(**{
            **self.config.transform_params,
            'augmentation': False,
            'complex_augmentation': False
        })
        
        val_dataset = get_datasets(
            img_folder_path=self.config.img_path,
            mask_folder_path=self.config.mask_path,
            transform=val_transform
        )
        
        self.logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _create_training_strategy(self, train_dataset, val_dataset) -> TrainingStrategy:
        """Create the appropriate training strategy."""
        strategy_kwargs = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'batch_size': self.config.batch_size,
            'output_dir': str(self.output_dir),
            'config': self.config
        }
        
        if self.config.strategy == "full_dataset":
            return FullDatasetStrategy(**strategy_kwargs)
        elif self.config.strategy == "train_val_split":
            return TrainValSplitStrategy(**strategy_kwargs)
        elif self.config.strategy == "k_fold":
            return KFoldStrategy(
                k_folds=self.config.k_folds,
                **strategy_kwargs
            )
        else:
            raise ValueError(f"Unknown training strategy: {self.config.strategy}")
    
    def _run_preeval(self, val_dataset) -> Optional[Dict]:
        """Run pre-evaluation if requested."""
        if not self.config.preeval:
            return None
            
        self.logger.info("Running pre-evaluation...")
        
        # Create a simple trainer for pre-evaluation
        trainer = MultiStageTrainer(
            stage1_config=self.config.stage1_config,
            stage2_config=self.config.stage2_config,
            output_dir=str(self.output_dir),
            fold=0,
            debug=self.config.debug
        )
        
        # Run inference on validation dataset
        results = trainer.run_inference_validation(
            val_dataset, 
            normalize_inf=self.config.transform_params['normalize_inf'],
            stain_transform=self.config.transform_params['stain_transform'],
            total_epochs=self.config.stage2_config.total_epochs
        )
        
        # Save pre-evaluation results
        preeval_dir = self.output_dir / "results"
        preeval_dir.mkdir(parents=True, exist_ok=True)
        
        with open(preeval_dir / "preeval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Pre-evaluation completed")
        return results
    
    def _save_experiment_config(self) -> None:
        """Save experiment configuration."""
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Experiment config saved to {config_path}")
    
    def _save_final_results(self, results: List[Dict]) -> None:
        """Save final experiment results."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        with open(results_dir / "individual_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # If k-fold, aggregate results
        if self.config.strategy == "k_fold" and len(results) > 1:
            for i, result in enumerate(results):
                self.results_aggregator.add_fold_results(i, result)
            
            # Save aggregated results
            self.results_aggregator.save_aggregated_results()
            self.results_aggregator.print_summary()
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def run(self) -> List[Dict]:
        """
        Run the complete experiment.
        
        Returns:
            List of results from each fold/run.
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING EXPERIMENT: {self.config.output_name}")
        self.logger.info("=" * 60)
        
        # Save experiment configuration
        self._save_experiment_config()
        
        # Monitor initial GPU state
        self.gpu_monitor.print_memory_summary("Experiment Start")
        
        try:
            # Create datasets
            train_dataset, val_dataset = self._create_datasets()
            
            # Run pre-evaluation if requested
            preeval_results = self._run_preeval(val_dataset)
            
            # Create training strategy
            training_strategy = self._create_training_strategy(train_dataset, val_dataset)
            
            self.logger.info(f"Using training strategy: {type(training_strategy).__name__}")
            
            # Execute training
            results = training_strategy.execute()
            
            # Save results
            self._save_final_results(results)
            
            # Final GPU state
            self.gpu_monitor.print_memory_summary("Experiment End")
            
            total_time = time.time() - start_time
            self.logger.info(f"Experiment completed in {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed with error: {str(e)}", exc_info=True)
            raise
        
        finally:
            # Clean up GPU memory
            self.gpu_monitor.clear_gpu_cache()


def create_experiment_runner(
    img_path: str,
    mask_path: str,
    output_name: str,
    strategy: str = "k_fold",
    **kwargs
) -> ExperimentRunner:
    """
    Factory function to create an experiment runner.
    
    Args:
        img_path: Path to images.
        mask_path: Path to masks.
        output_name: Name for the experiment output.
        strategy: Training strategy to use.
        **kwargs: Additional configuration parameters.
        
    Returns:
        ExperimentRunner instance.
    """
    config = ExperimentConfig(
        img_path=img_path,
        mask_path=mask_path,
        output_name=output_name,
        strategy=strategy,
        **kwargs
    )
    
    return ExperimentRunner(config)


def run_experiment_from_config(config_path: str) -> List[Dict]:
    """
    Run experiment from a configuration file.
    
    Args:
        config_path: Path to configuration JSON file.
        
    Returns:
        List of results from the experiment.
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create stage configs
    stage1_config = Stage1Config(**config_dict['stage1_config'])
    stage2_config = Stage2Config(**config_dict['stage2_config'])
    
    # Create experiment config
    config = ExperimentConfig(
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        **{k: v for k, v in config_dict.items() 
           if k not in ['stage1_config', 'stage2_config']}
    )
    
    # Run experiment
    runner = ExperimentRunner(config)
    return runner.run()


# Convenience function for backward compatibility
def finetune(img_path: str, mask_path: str, output_name: str, **kwargs) -> List[Dict]:
    """
    Legacy finetune function for backward compatibility.
    
    This function maintains the same interface as the original finetune function
    but uses the new structured approach internally.
    """
    # Map old parameters to new structure
    stage1_kwargs = {
        'stage1_type': kwargs.get('stage1', 'cellfinder'),
        'total_epochs': kwargs.get('total_epochs_s1', 500),
        'learning_rate': kwargs.get('lr_stage1', 1e-4),
        'ce_loss_weight': kwargs.get('ce_loss_weight', 1),
        'bbox_loss_weight': kwargs.get('bbox_loss_weight', 5),
        'giou_loss_weight': kwargs.get('giou_loss_weight', 2)
    }
    
    stage2_kwargs = {
        'total_epochs': kwargs.get('total_epochs_s2', 500),
        'learning_rate': kwargs.get('lr_stage2', 1e-4),
        'focal_loss_weight': kwargs.get('focal_loss_weight', 1.0),
        'focal_gamma': kwargs.get('focal_gamma', 2.0),
        'dice_loss_weight': kwargs.get('dice_loss_weight', 1.0),
        'boundary_loss_weight': kwargs.get('boundary_loss_weight', 1.0),
        'online_hard_negative_mining': kwargs.get('online_hard_negative_mining', False),
        'online_hard_negative_mining_weighted': kwargs.get('online_hard_negative_mining_weighted', False)
    }
    
    # Determine strategy
    if kwargs.get('use_full_dataset', False):
        strategy = 'full_dataset'
    elif kwargs.get('train_val_training', False):
        strategy = 'train_val_split'
    elif kwargs.get('k_fold_training', False):
        strategy = 'k_fold'
    else:
        strategy = 'k_fold'  # Default
    
    # Create configs
    stage1_config = Stage1Config(**stage1_kwargs)
    stage2_config = Stage2Config(**stage2_kwargs)
    
    # Create experiment config
    config = ExperimentConfig(
        img_path=img_path,
        mask_path=mask_path,
        output_name=output_name,
        strategy=strategy,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        batch_size=kwargs.get('batch_size', 1),
        debug=kwargs.get('debug', False),
        preeval=kwargs.get('preeval', False),
        **{k: v for k, v in kwargs.items() 
           if k in ['normalize', 'rgb_transform', 'stain_transform', 'eosin', 'dab',
                   'inversion', 'only_nuclei', 'gamma', 'normalize_inf', 'augmentation',
                   'complex_augmentation', 'multiple_datasets']}
    )
    
    # Run experiment
    runner = ExperimentRunner(config)
    return runner.run()