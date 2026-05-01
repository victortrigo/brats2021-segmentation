"""
Training Script for BraTS 2021 Brain Tumor Segmentation.

This module provides the main training pipeline for all segmentation models
(U-Net, DeepLabV3+, DeepLabV3+SAM, CLCUNet). It handles configuration management,
dataset loading, model instantiation, and the complete training loop with early
stopping and TensorBoard logging.

Key Features:
    - Multi-model support (U-Net, DeepLabV3+, DeepLabV3+SAM, CLCUNet)
    - Configuration merging (base config + mode override)
    - Flexible dataset subsampling for quick testing
    - Early stopping with configurable patience
    - Learning rate scheduling (StepLR)
    - TensorBoard integration for monitoring
    - Automatic model checkpointing on improvement
    
Main Functions:
    get_model: Instantiates model from configuration
    main: Complete training pipeline
    
Usage:
    # Full dataset training
    python training.py --config configs/unet_config.yaml
    
    # Subset training (quick testing)
    python training.py --config configs/unet_config.yaml --mode configs/config_test.yaml
    
Configuration Structure:
    model:
        name: UNet|DeepLabV3+|DeepLabV3+SAM|CLCUNet
        in_channels: 4
        num_classes: 4
        base_channels: 64
    training:
        epochs: 500
        batch_size: 2
        learning_rate: 0.001
        loss: DiceLoss|JaccardLoss
        patience: 50
        subset_train_size: 0  # 0 = use full dataset
        subset_valid_size: 0
        subset_test_size: 0
    paths:
        data_dir: ./data/processed
        models_dir: ./models
"""


import os
from datetime import datetime
from typing import Dict, Any, Tuple

import random
import numpy as np
import torch
import yaml

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset, create_subset
import clcunet
import deeplabv3
import deeplabv3sam
import unet

from metrics import Accuracy, DiceLoss, Fscore, IoU, JaccardLoss, Precision, Recall
from train import TrainEpoch, ValidEpoch

def set_seed(seed: int = 42) -> None:
    """
    Freezes all sources of randomness to ensure full code reproducibility.
    
    Args:
        seed (int): The seed value to use for all random number generators. Default is 42.
    """
    # 1. Basic Python and environment variables
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. NumPy (Affects train_test_split and other random selections)
    np.random.seed(seed)

    # 3. PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    # 4. CuDNN (NVIDIA backend for convolutions)
    # IMPORTANT: Deterministic mode ensures identical results across runs,
    # but it may slightly slow down the training process.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 Global seed set to: {seed}")


# ============================================================================
# Model Factory
# ============================================================================

def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Instantiate a segmentation model from configuration.
    
    This factory function creates the appropriate model based on the 'name' field
    in the configuration. Each model has specific parameter requirements that are
    automatically extracted from the config.
    
    Supported Models:
        - UNet: Standard U-Net with configurable base channels
        - DeepLabV3+: Xception backbone with ASPP
        - DeepLabV3+SAM: DeepLabV3+ with Segmented Attention Module
        - CLCUNet: Cross-Level Connected U-Net with attention
        
    Args:
        config (Dict[str, Any]): Configuration dictionary with structure:
            {
                'model': {
                    'name': str,  # Model name
                    'in_channels': int,  # Input channels (default: 4)
                    'num_classes': int,  # Output classes (default: 4)
                    'base_channels': int  # Base feature channels (default: 64)
                }
            }
            
    Returns:
        torch.nn.Module: Instantiated model ready for training
        
    Raises:
        ValueError: If model name is not supported
        
    Example:
        >>> config = {
        ...     'model': {
        ...         'name': 'UNet',
        ...         'in_channels': 4,
        ...         'num_classes': 4,
        ...         'base_channels': 64
        ...     }
        ... }
        >>> model = get_model(config)
        >>> print(type(model))
        <class 'unet.UNet'>
        
    Note:
        - UNet and CLCUNet use 'base_channels' parameter
        - DeepLabV3+ models only use 'in_channels' and 'num_classes'
        - Missing parameters default to standard BraTS values
    """
    model_name = config.get('model', {}).get('name')
    params = config.get('model', {})

    # Extract universal parameters with BraTS-specific defaults
    in_channels = params.get('in_channels', 4)  # FLAIR, T1, T1CE, T2
    num_classes = params.get('num_classes', 4)  # Background, NCR, ED, ET
    base_channels = params.get('base_channels', 64)
    
    # Instantiate model based on name
    if model_name == "UNet":
        return unet.UNet(
            in_channels=in_channels, 
            num_classes=num_classes, 
            base_channels=base_channels
        )
    
    elif model_name == "DeepLabV3+":
        return deeplabv3.DeepLabV3Plus(
            in_channels=in_channels, 
            num_classes=num_classes
        )
    
    elif model_name == "DeepLabV3+SAM":
        return deeplabv3sam.DeepLabV3PlusSAM(
            in_channels=in_channels, 
            num_classes=num_classes
        )
    elif model_name == "CLCUNet":
        return clcunet.CLCUNet(
            in_channels=in_channels, 
            base_channels=base_channels
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        

# ============================================================================
# Training Pipeline for Full Training Experiment
# ============================================================================

def run_experiment(
    model_name: str, 
    seed: int, 
    best_params: Dict[str, Any], 
    run_dir: str, 
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader, 
    device: str = 'cuda:0'
) -> Tuple[float, float]:    
    """
    Dynamic training engine adapted for the Factorial Design (Phase 2).
    
    This function runs a SINGLE training trial for a specific model and a SINGLE seed.
    The outer loop (iterating through all models and all seeds) is handled externally 
    by `run_phase2_factorial.py`.
    
    Args:
        model_name (str): Name of the model architecture (e.g., 'UNet', 'CLCUNet').
        seed (int): The specific random seed for this training trial.
        best_params (Dict[str, Any]): Hyperparameters found by Optuna in Phase 1.
        run_dir (str): Directory where the model weights and logs will be saved.
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        device (str): Device to run the training on (default: 'cuda:0').
        
    Returns:
        Tuple[float, float]: The best Dice Score and the best IoU achieved during training.
    """
    print(f"\n--- Starting Training Engine for {model_name} | Seed: {seed} ---")

    # 1. Set the EXACT statistical seed for this specific trial
    # (Ensure set_seed is defined in your utils or imported into this file)
    set_seed(seed)

    # 2. Instantiate the model
    # Using the existing get_model() factory from training.py
    model = get_model({"model": {"name": model_name}}).to(device)

    # 3. Configure the Optimizer with the "recipe" from the YAML
    lr = float(best_params.get('lr', 1e-3))
    wd = float(best_params.get('weight_decay', 1e-5))
    opt_name = best_params.get('optimizer', 'Adam')

    if opt_name == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    # 4. Configure Scheduler and Loss Function
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Special handling for CLCUNet (requires sigmoid activation)
    if model_name == "CLCUNet":
        criterion = DiceLoss(activation="sigmoid")
        print("  Using DiceLoss with 'sigmoid' activation for CLCUNet.")
    else:
        criterion = DiceLoss()
        print("  Using DiceLoss by default.")
        
    criterion.to(device)

    # 5. Metrics and Epoch Handlers
    # Make sure DiceScore is imported if you are using it
    metrics = [
        IoU(threshold=0.5),
        Accuracy(threshold=0.5),
        Fscore(threshold=0.5),
        Recall(threshold=0.5),
        Precision(threshold=0.5),
    ]

    train_epoch = TrainEpoch(
        model, loss=criterion, metrics=metrics, optimizer=optimizer, device=device, verbose=True
    )
    valid_epoch = ValidEpoch(
        model, loss=criterion, metrics=metrics, device=device, verbose=True
    )

    # 6. TensorBoard Setup (Saving logs inside the specific run directory)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, 'logs'))

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    EPOCHS = 200 # Maximum number of epochs
    PATIENCE = 50
    
    max_iou = 0.0 
    best_fscore = 0.0  # Dice score
    patience_counter = 0 

    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch: {epoch}/{EPOCHS}')

        # Run training and validation phases
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        scheduler.step()

        # Log metrics to TensorBoard
        if 'dice_loss' in train_logs:
            writer.add_scalar('Loss/train', train_logs['dice_loss'], epoch)
        if 'dice_loss' in valid_logs:
            writer.add_scalar('Loss/validation', valid_logs['dice_loss'], epoch)
        if 'iou_score' in train_logs:
            writer.add_scalar('IoU/train', train_logs['iou_score'], epoch)
        if 'iou_score' in valid_logs:
            writer.add_scalar('IoU/validation', valid_logs['iou_score'], epoch)
        if 'fscore' in train_logs:
            writer.add_scalar('Fscore/train', train_logs['fscore'], epoch)
        if 'fscore' in valid_logs:
            writer.add_scalar('Fscore/validation', valid_logs['fscore'], epoch)

        valid_iou = float(valid_logs.get('iou_score', 0.0))
        valid_fscore = float(valid_logs.get('fscore', 0.0)) 

        # Early Stopping and Checkpointing Logic
        if max_iou < valid_iou:
            max_iou = valid_iou
            best_fscore = valid_fscore
            patience_counter = 0
            
            # Save the .pth file inside the run_dir
            safe_name = model_name.replace("+", "plus").lower()
            model_path = os.path.join(run_dir, f"{safe_name}_seed{seed}.pth")
            torch.save(model.state_dict(), model_path)
            
            print(f'🔥 Improvement detected! Model saved to: {model_path}')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'🛑 Early stopping at epoch {epoch}. No improvements in {PATIENCE} consecutive epochs.')
                break

    writer.close()
    print(f"✅ Training completed for {model_name} (Seed {seed}). Best IoU: {max_iou:.4f}")
    
    # Return the best scores to the main orchestrator script
    return best_fscore, max_iou


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main(config_path: str, mode_config_path: str = None):
    """
    Main training pipeline with configuration merging and early stopping.
    
    This function orchestrates the complete training process:
    1. Load and merge configurations (base + optional mode override)
    2. Setup device (CPU/CUDA)
    3. Load and prepare datasets (with optional subsampling)
    4. Instantiate model, optimizer, scheduler, and loss function
    5. Initialize TensorBoard logging
    6. Run training loop with validation and early stopping
    7. Save best model checkpoints
    
    Configuration Merging:
        - Base config (e.g., unet_config.yaml): Full training setup
        - Mode config (e.g., config_test.yaml): Override for quick testing
        - Mode config values override base config values
        - Useful for switching between full training and quick experiments
        
    Early Stopping:
        - Monitors validation IoU score
        - Stops training if no improvement for 'patience' epochs
        - Saves model only when validation IoU improves
        - Prevents overfitting and saves computation time
        
    Args:
        config_path (str): Path to base configuration YAML file
            Example: 'configs/unet_config.yaml'
        mode_config_path (str, optional): Path to mode override YAML
            Example: 'configs/config_test.yaml'
            If provided, values here override base config
            
    Configuration Example:
        # unet_config.yaml (base)
        model:
            name: UNet
            in_channels: 4
            num_classes: 4
            base_channels: 64
        training:
            epochs: 500
            batch_size: 2
            learning_rate: 0.001
            loss: DiceLoss
            patience: 50
            subset_train_size: 0  # 0 = full dataset (999 samples)
            subset_valid_size: 0
        paths:
            data_dir: ./data/processed
            models_dir: ./models
            
        # config_test.yaml (mode override)
        training:
            epochs: 10  # Override: fewer epochs
            subset_train_size: 50  # Override: use only 50 samples
            subset_valid_size: 5   # Override: use only 5 samples
            patience: 3  # Override: less patience
            
    Training Loop:
        for epoch in range(epochs):
            1. Train on training set
            2. Validate on validation set
            3. Log metrics to TensorBoard
            4. Check for improvement:
                - If IoU improves: Save model, reset patience counter
                - If IoU doesn't improve: Increment patience counter
            5. Check early stopping:
                - If patience_counter >= patience: Stop training
            6. Update learning rate scheduler
            
    Outputs:
        - Trained model: saved to {models_dir}/{model_name}.pt
        - TensorBoard logs: saved to runs/{model_name}_{timestamp}/
        - Console output: Progress bars and epoch summaries
        
    Example:
        >>> # Full training with 999 samples
        >>> main('configs/unet_config.yaml')
        Usando dispositivo: cuda:0
        Modo Entrenamiento Completo. Train size: 999, Valid size: 125
        Iniciando el entrenamiento del modelo: UNet
        
        Epoch: 0
        train: 100%|██████| dice_loss - 0.234, iou_score - 0.856
        valid: 100%|██████| dice_loss - 0.189, iou_score - 0.891
        Modelo guardado en ./models/UNet.pt
        
        >>> # Quick testing with 50 samples
        >>> main('configs/unet_config.yaml', 'configs/config_test.yaml')
        --- MODO PRUEBAS RÁPIDAS (SUBSET) ---
        Train size: 50, Valid size: 5, Test size: 5
        
    Note:
        - Always run validation after each training epoch
        - Model is saved ONLY when validation IoU improves
        - Early stopping prevents unnecessary computation
        - TensorBoard logs are timestamped to avoid conflicts
        - Learning rate is reduced by 10× at epoch 50 (StepLR)
        
    See Also:
        - get_model(): Model instantiation
        - TrainEpoch: Training loop implementation
        - ValidEpoch: Validation loop implementation
        - Dataset: Data loading and preprocessing
    """
    
    # ========================================================================
    # Configuration Loading and Merging
    # ========================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load base configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f) or {}

    # Load optional mode configuration (for overrides)
    if mode_config_path and os.path.exists(mode_config_path):
        with open(mode_config_path, 'r', encoding='utf-8') as f_mode:
            mode_cfg = yaml.safe_load(f_mode) or {}
    else:
        mode_cfg = {}
        
    # Merge configurations: mode_cfg overrides base_cfg
    config: Dict[str, Any] = {}
    config['model'] = {**base_cfg.get('model', {}), **mode_cfg.get('model', {})}
    config['paths'] = {**base_cfg.get('paths', {}), **mode_cfg.get('paths', {})}
    config['training'] = {**base_cfg.get('training', {}), **mode_cfg.get('training', {})}

    # Extract training hyperparameters
    training_cfg = config.get('training', {})
    subset_train_size = training_cfg.get('subset_train_size', 0)
    subset_valid_size = training_cfg.get('subset_valid_size', 0)
    subset_test_size = training_cfg.get('subset_test_size', 0)
    BATCH_SIZE = training_cfg.get('batch_size', 1)
    EPOCHS = training_cfg.get('epochs', 1)
    LEARNING_RATE = training_cfg.get('learning_rate', 1e-3)
    PATIENCE = training_cfg.get('patience', 10)
    LOSS_NAME = training_cfg.get('loss', 'DiceLoss')

    print(f"Using device: {device}")

    # ========================================================================
    # Dataset Loading and Preparation
    # ========================================================================

    DATA_DIR = config.get('paths', {}).get('data_dir', './')
    CLASSES = ['background', 'NCR', 'ED', 'ET']

    # Define dataset paths
    x_train_dir = os.path.join(DATA_DIR, 'X_train')
    y_train_dir = os.path.join(DATA_DIR, 'y_train')
    x_valid_dir = os.path.join(DATA_DIR, 'X_val')
    y_valid_dir = os.path.join(DATA_DIR, 'y_val')
    x_test_dir = os.path.join(DATA_DIR, 'X_test')
    y_test_dir = os.path.join(DATA_DIR, 'y_test')

    # Create full datasets
    train_dataset_full = Dataset(x_train_dir, y_train_dir, CLASSES)
    valid_dataset_full = Dataset(x_valid_dir, y_valid_dir, CLASSES)
    test_dataset_full = Dataset(x_test_dir, y_test_dir, CLASSES)

    # Create DataLoaders (with optional subsampling)
    subset_train, train_loader = create_subset(
        train_dataset_full, subset_train_size, BATCH_SIZE, shuffle=True
    )
    subset_valid, valid_loader = create_subset(
        valid_dataset_full, subset_valid_size, BATCH_SIZE, shuffle=False
    )
    subset_test, test_loader = create_subset(
        test_dataset_full, subset_test_size, BATCH_SIZE, shuffle=False
    )

    # Display training mode
    if subset_train_size > 0 or subset_valid_size > 0:
        print("--- QUICK TEST MODE (SUBSET) ---")
        print(f"Train size: {len(subset_train)}, Valid size: {len(subset_valid)}, Test size: {len(subset_test)}")
    else:
        print(f"Full Training Mode. Train size: {len(subset_train)}, Valid size: {len(subset_valid)}")

    # ========================================================================
    # Model, Optimizer, and Loss Setup
    # ========================================================================

    # Define evaluation metrics
    metrics = [
        IoU(threshold=0.5),
        Accuracy(threshold=0.5),
        Fscore(threshold=0.5),
        Recall(threshold=0.5),
        Precision(threshold=0.5),
    ]

    # Instantiate model
    model = get_model(config)
    model_name = config.get('model', {}).get('name', '')

    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Setup learning rate scheduler
    # Reduces LR by 10× at epoch 50
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Setup loss function
    if LOSS_NAME == "DiceLoss":
        # Special case: CLCUNet requires sigmoid activation
        if model_name == "CLCUNet":
            criterion = DiceLoss(activation="sigmoid")
            print("Using DiceLoss with 'sigmoid' activation for CLCUNet.")
        else:
            criterion = DiceLoss()
            print("Using DiceLoss by default.")
    elif LOSS_NAME == "JaccardLoss":
        criterion = JaccardLoss()
        print("Using JaccardLoss.")
    else:
        raise ValueError(f"Unsupported loss: {LOSS_NAME}")

    criterion.to(device)

    print(f"Starting model training: {model_name}")

    # ========================================================================
    # TensorBoard Initialization
    # ========================================================================

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}_{now}"
    writer = SummaryWriter(os.path.join('runs', run_name))

    # ========================================================================
    # Training and Validation Epoch Setup
    # ========================================================================
    
    train_epoch = TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True
    )

    valid_epoch = ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True
    )

    # ========================================================================
    # Main Training Loop with Early Stopping
    # ========================================================================
    
    max_score = 0 # Best validation IoU score
    patience_counter = 0  # Epochs without improvement

    for i in range(0, EPOCHS):
        print('\nEpoch: {}'.format(i))

        # Training phase
        train_logs = train_epoch.run(train_loader)

        # Validation phase
        valid_logs = valid_epoch.run(valid_loader)

        # Update learning rate
        # optimizer.step()
        scheduler.step()

        # Log metrics to TensorBoard
        if 'dice_loss' in train_logs:
            writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
        if 'iou_score' in train_logs:
            writer.add_scalar('IoU/train', train_logs.get('iou_score', 0.0), i)
        if 'dice_loss' in valid_logs:
            writer.add_scalar('Loss/validation', valid_logs['dice_loss'], i)
        if 'iou_score' in valid_logs:
            writer.add_scalar('IoU/validation', valid_logs.get('iou_score', 0.0), i)

        # ====================================================================
        # Checkpointing and Early Stopping Logic
        # ====================================================================

        valid_iou = valid_logs.get('iou_score', 0.0)

        if max_score < valid_iou:
            # Validation IoU improved - save model
            max_score = valid_iou
            models_dir = config.get('paths', {}).get('models_dir', './')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'Model saved in {model_path}')

            # Reset patience counter
            patience_counter = 0
        else:
            # No improvement - increment patience counter
            patience_counter += 1

            # Check if we should stop early
            if patience_counter >= PATIENCE:
                print(f'Early stopping in epoch {i} after {PATIENCE} epochs without improvement.')
                break

    # ========================================================================
    # Cleanup
    # ========================================================================

    writer.close()
    print(f"\nTraining completed. Best IoU: {max_score:.4f}")


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train BraTS 2021 segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training with UNet (999 samples)
    python training.py --config configs/unet_config.yaml
    
    # Quick testing with subset (50 samples)
    python training.py --config configs/unet_config.yaml --mode configs/config_test.yaml
    
    # Train DeepLabV3+ with SAM
    python training.py --config configs/deeplabv3sam_config.yaml
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to base configuration YAML file (e.g., configs/unet_config.yaml)."
    )

    parser.add_argument(
        "--mode", 
        type=str, 
        required=False, 
        help="Path to mode override YAML (e.g., configs/config_test.yaml). "
             "Values here override base config for quick testing."
    )

    args = parser.parse_args()
    main(args.config, args.mode)