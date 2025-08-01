"""
Simple experiment runner script.
"""

import argparse
import os
from pathlib import Path
from segmentation.training.config.stage1_config import Stage1Config
from segmentation.training.config.stage2_config import Stage2Config
from segmentation.datasets.config import DatasetConfig
from segmentation.datasets.dataset_factory import DatasetFactory
from segmentation.transforms.composed_transforms import FullTransform
from segmentation.training.strategies.full_dataset import FullDatasetStrategy
from segmentation.training.strategies.train_val_split import TrainValSplitStrategy
from segmentation.training.strategies.k_fold import KFoldStrategy
from segmentation.training.core.multistage_trainer import MultiStageTrainer
from segmentation.utils.model_utils import load_cellsam_model


def run_experiment(args,
                  stage1_config_path: str,
                  stage2_config_path: str,
                  dataset_config_path: str,
                  **overrides):
    """
    Run experiment with given configs and optional parameter overrides.
    
    Args:
        stage1_config_path: Path to Stage 1 YAML config
        stage2_config_path: Path to Stage 2 YAML config  
        dataset_config_path: Path to dataset YAML config
        output_name: Experiment output name
        **overrides: Parameter overrides for hyperparameter search
    """
    # Load configs
    stage1_config = Stage1Config()
    stage1_config.load_from_file(config_file=stage1_config_path)
    stage2_config = Stage2Config()
    stage2_config.load_from_file(config_file=stage2_config_path)
    dataset_config = DatasetConfig()
    dataset_config.load_from_file(config_file=dataset_config_path)
    
    # Apply any parameter overrides
    for key, value in overrides.items():
        if hasattr(stage1_config, key):
            setattr(stage1_config, key, value)
        if hasattr(stage2_config, key):  
            setattr(stage2_config, key, value)
        if hasattr(dataset_config, key):
            setattr(dataset_config, key, value)

    # Create datasets
    transform = FullTransform(
            normalize=dataset_config.normalize, 
            rgb_transform=dataset_config.rgb_transform, 
            stain_transform=dataset_config.stain_transform,
            eosin=dataset_config.eosin,
            dab=dataset_config.dab, 
            inversion=dataset_config.inversion, 
            only_nuclei=dataset_config.only_nuclei, 
            gamma=dataset_config.gamma, 
            debug=dataset_config.debug
    )

    # Create training strategy
    output_path = os.path.join(stage2_config.output_dir, stage2_config.output_name)
    if stage2_config.training_strategy == "full_dataset":
        strat = FullDatasetStrategy(output_name=output_path)
    elif stage2_config.training_strategy == "train_val_split":
        strat = TrainValSplitStrategy(output_name=output_path)
    elif stage2_config.training_strategy == "k_fold":
        strat = KFoldStrategy(
            k=stage2_config.k_folds,
            output_name=output_path
        )
    else:
        raise ValueError(f"Unknown training strategy: {stage2_config.training_strategy}")

    # Load model
    cellsam, device = load_cellsam_model(args.model_path)

    # Create datasets
    dataset = DatasetFactory().create_dataset(
        (dataset_config.img_path, dataset_config.mask_path),
        transform = transform,
        do_augmentation = dataset_config.do_augmentation,
        complex_augmentation = dataset_config.complex_augmentation
    )
    splits = strat.prepare_data_splits(dataset, dataset)

    # Trainer
    trainer = MultiStageTrainer(cellsam, stage1_config, stage2_config, device, output_path)

    # Run
    if not args.only_eval:
        strat.execute_training(trainer, splits)

    # Metrics evaluation during infernce
    evaluation_results = trainer.finalize_with_evaluation(
        test_dataset=splits[0][1],
        metrics_to_optimize=['AP50'],
        is_deep_model=False  # Set based on your training epochs
    )

    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Run segmentation experiment")
    
    # Required arguments
    parser.add_argument("--stage1_config", required=True, help="Stage 1 config YAML file")
    parser.add_argument("--stage2_config", required=True, help="Stage 2 config YAML file")
    parser.add_argument("--dataset_config", required=True, help="Dataset config YAML file")
    
    # Optional parameter overrides for hyperparameter search
    parser.add_argument("--img_path", help="Path to images")
    parser.add_argument("--mask_path", help="Path to masks")
    parser.add_argument("--output_dir", help="Experiment output dir")
    parser.add_argument("--output_name", help="Experiment output name")
    parser.add_argument("--only_eval", action="store_true", help="Perform evaluation of a model")
    parser.add_argument("--model_path", default=None, help="Experiment output name")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--focal_loss_weight", type=float, help="Override focal loss weight")
    parser.add_argument("--gamma", type=float, help="Override gamma")
    parser.add_argument("--training_strategy", type=str, help="Training strategy used")
    parser.add_argument("--bbox_loss_weight", type=float, help="Stage 1 loss bbox weight")
    parser.add_argument("--giou_loss_weight", type=float, help="Stage 1 loss giou weight")
    parser.add_argument("--epochs_s1", type=int, help="Stage 1 epochs")
    parser.add_argument("--epochs_s2", type=int, help="Stage 2 epochs")
    
    # Add more parameters as needed for your hyperparameter search
    
    args = parser.parse_args()
    
    # Extract overrides (remove None values)
    overrides = {k: v for k, v in vars(args).items() 
                 if v is not None and k not in ['stage1_config', 'stage2_config', 'dataset_config']}
    
    # Run experiment
    results = run_experiment(
        args,
        stage1_config_path=args.stage1_config,
        stage2_config_path=args.stage2_config,
        dataset_config_path=args.dataset_config,
        **overrides
    )
    
    print("Experiment completed!")
    return results


if __name__ == "__main__":
    main()