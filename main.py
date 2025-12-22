"""
Main Training Script for BraTS 2021 - Execute All 4 Models.

This script orchestrates the training of all 4 segmentation models:
- UNet
- CLCUNet
- DeepLabV3+
- DeepLabV3+SAM

It provides options to run in FULL mode (complete dataset) or TEST mode (small subset).
Each model's training is logged and results are saved independently.

Usage:
    # Run all models in TEST mode (quick validation)
    python main.py --mode test
    
    # Run all models in FULL mode (complete training)
    python main.py --mode full
    
    # Run specific models only
    python main.py --mode test --models unet clcunet
    
    # Run with custom test configuration
    python main.py --mode test --test-config configs/custom_test.yaml
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import List, Dict, Any

# Add src to path to import training module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training import main as train_model


# ============================================================================
# Model Configuration Mapping
# ============================================================================

MODEL_CONFIGS = {
    'unet': 'configs/unet_config.yaml',
    'clcunet': 'configs/clcunet_config.yaml',
    'deeplabv3': 'configs/deeplabv3_config.yaml',
    'deeplabv3sam': 'configs/deeplabv3sam_config.yaml'
}

TEST_CONFIG = 'configs/config_test.yaml'


# ============================================================================
# Training Execution Functions
# ============================================================================

def print_banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'─' * 80}")
    print(f"► {text}")
    print(f"{'─' * 80}\n")


def verify_config_exists(config_path: str) -> bool:
    """
    Verify that a configuration file exists.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Configuration file not found: {config_path}")
        return False
    return True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dict containing configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_training_info(model_name: str, mode: str) -> str:
    """
    Get formatted training information string.
    
    Args:
        model_name: Name of the model
        mode: Training mode (test or full)
        
    Returns:
        Formatted info string
    """
    config_path = MODEL_CONFIGS[model_name]
    config = load_config(config_path)
    
    epochs = config.get('training', {}).get('epochs', '?')
    batch_size = config.get('training', {}).get('batch_size', '?')
    lr = config.get('training', {}).get('learning_rate', '?')
    
    info = f"Model: {model_name.upper()}\n"
    info += f"Config: {config_path}\n"
    info += f"Mode: {mode.upper()}\n"
    info += f"Epochs: {epochs}\n"
    info += f"Batch Size: {batch_size}\n"
    info += f"Learning Rate: {lr}"
    
    return info


def train_single_model(model_name: str, 
                       mode: str = 'test', 
                       test_config: str = None) -> bool:
    """
    Train a single model with specified configuration.
    
    Args:
        model_name: Name of model to train (unet, clcunet, etc.)
        mode: Training mode - 'test' or 'full'
        test_config: Optional custom test configuration path
        
    Returns:
        bool: True if training succeeded, False if failed
    """
    print_section(f"Starting Training: {model_name.upper()}")
    
    # Get configuration path
    config_path = MODEL_CONFIGS.get(model_name.lower())
    if not config_path:
        print(f"❌ ERROR: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return False
    
    # Verify config exists
    if not verify_config_exists(config_path):
        return False
    
    # Determine mode config
    mode_config = None
    if mode.lower() == 'test':
        mode_config = test_config or TEST_CONFIG
        if not verify_config_exists(mode_config):
            return False
    
    # Display training info
    print(get_training_info(model_name, mode))
    print()
    
    # Record start time
    start_time = datetime.now()
    print(f"⏱️  Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Execute training
        train_model(config_path, mode_config)
        
        # Record end time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ SUCCESS: {model_name.upper()} training completed!")
        print(f"⏱️  Duration: {duration}")
        print(f"⏱️  Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {model_name.upper()} training failed!")
        print(f"Error message: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


def train_all_models(mode: str = 'test', 
                     models: List[str] = None,
                     test_config: str = None):
    """
    Train all specified models sequentially.
    
    Args:
        mode: Training mode - 'test' or 'full'
        models: List of model names to train. If None, trains all models.
        test_config: Optional custom test configuration path
    """
    # Determine which models to train
    if models is None:
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        models_to_train = [m.lower() for m in models]
        # Validate model names
        invalid = [m for m in models_to_train if m not in MODEL_CONFIGS]
        if invalid:
            print(f"❌ ERROR: Invalid model names: {', '.join(invalid)}")
            print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return
    
    # Display execution plan
    print_banner("BraTS 2021 Training - Execution Plan")
    print(f"Mode: {mode.upper()}")
    print(f"Models to train: {', '.join([m.upper() for m in models_to_train])}")
    print(f"Total models: {len(models_to_train)}")
    if mode.lower() == 'test':
        test_cfg = test_config or TEST_CONFIG
        print(f"Test config: {test_cfg}")
    print()
    
    # Confirm execution
    response = input("Proceed with training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled by user.")
        return
    
    # Record overall start time
    overall_start = datetime.now()
    print_banner(f"Starting Training Session - {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track results
    results = {}
    
    # Train each model
    for i, model_name in enumerate(models_to_train, 1):
        print_banner(f"Model {i}/{len(models_to_train)}: {model_name.upper()}", char="═")
        
        success = train_single_model(model_name, mode, test_config)
        results[model_name] = success
        
        # Separator between models
        if i < len(models_to_train):
            print("\n" + "▓" * 80 + "\n")
    
    # Record overall end time
    overall_end = datetime.now()
    total_duration = overall_end - overall_start
    
    # Display summary
    print_banner("Training Session Summary", char="═")
    print(f"Started:  {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_duration}")
    print()
    
    # Display results table
    print("Results:")
    print("─" * 80)
    print(f"{'Model':<20} {'Status':<15} {'Config'}")
    print("─" * 80)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        config = MODEL_CONFIGS[model_name]
        print(f"{model_name.upper():<20} {status:<15} {config}")
    
    print("─" * 80)
    
    # Summary statistics
    total = len(results)
    succeeded = sum(1 for s in results.values() if s)
    failed = total - succeeded
    
    print(f"\nTotal: {total} | Succeeded: {succeeded} | Failed: {failed}")
    
    if failed > 0:
        print("\n⚠️  Some models failed. Check logs above for details.")
    else:
        print("\n🎉 All models trained successfully!")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train all BraTS 2021 segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all models in TEST mode (subset training - quick validation)
    python main.py --mode test
    
    # Run all models in FULL mode (complete dataset)
    python main.py --mode full
    
    # Run specific models only
    python main.py --mode test --models unet clcunet
    python main.py --mode full --models deeplabv3 deeplabv3sam
    
    # Run with custom test configuration
    python main.py --mode test --test-config configs/my_test.yaml
    
Available Models:
    - unet          : Standard U-Net
    - clcunet       : Cross-Level Connected U-Net with SAM
    - deeplabv3     : DeepLabV3+ with Xception backbone
    - deeplabv3sam  : DeepLabV3+ with SAM attention
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'full'],
        default='test',
        help='Training mode: "test" (subset) or "full" (complete dataset). Default: test'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='List of models to train. If not specified, trains all models. '
             'Options: unet, clcunet, deeplabv3, deeplabv3sam'
    )
    
    parser.add_argument(
        '--test-config',
        type=str,
        default=None,
        help='Custom test configuration file (only used in test mode). '
             'Default: configs/config_test.yaml'
    )
    
    args = parser.parse_args()
    
    # Display welcome banner
    print_banner("BraTS 2021 Brain Tumor Segmentation - Training Pipeline", char="█")
    print("This script will train the following models:")
    for model_name, config_path in MODEL_CONFIGS.items():
        print(f"  • {model_name.upper():<15} - {config_path}")
    
    # Execute training
    try:
        train_all_models(
            mode=args.mode,
            models=args.models,
            test_config=args.test_config
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Some models may have completed training.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    