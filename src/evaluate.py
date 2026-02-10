"""
Comprehensive Evaluation Script for BraTS 2021 Brain Tumor Segmentation.

This script evaluates trained models on the test dataset and computes multiple
performance metrics including Dice, Hausdorff Distance (HD95), Sensitivity, 
and Specificity for each tumor subregion.

Key Features:
    - Multi-metric evaluation (Dice, HD95, Sensitivity, Specificity, IoU, Precision, Recall)
    - Per-class and overall metrics
    - Visualization of predictions vs ground truth
    - RGB color-coded segmentation maps
    - Detailed results saved to CSV and JSON
    - Statistical analysis (mean, std, median)
    
Metrics Computed:
    - Dice Score: Overlap measure (0-1, higher is better)
    - HD95: 95th percentile Hausdorff Distance (mm, lower is better)
    - Sensitivity (Recall): True Positive Rate
    - Specificity: True Negative Rate
    - Precision: Positive Predictive Value
    - IoU (Jaccard): Intersection over Union
    
Usage:
    # Evaluate single model
    python evaluate.py --config configs/unet_config.yaml --model models/UNet.pt
    
    # Evaluate all models
    python evaluate.py --evaluate-all
    
    # Evaluate with visualizations
    python evaluate.py --config configs/unet_config.yaml --model models/UNet.pt --visualize
    
Output Structure:
    reports/
    ├── {model_name}/
    │   ├── metrics.json          # Detailed per-sample metrics
    │   ├── summary.csv           # Aggregated statistics
    │   ├── confusion_matrices/   # Per-class confusion matrices
    │   └── visualizations/       # Sample predictions
    │       ├── sample_001_prediction.png
    │       ├── sample_001_comparison.png
    │       └── ...
"""

import os
import sys
import json
import argparse
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import Dataset
from training import get_model


# ============================================================================
# Color Mapping for BraTS Classes
# ============================================================================

# BraTS color scheme for visualization
# IMPORTANT: BraTS uses labels 0, 1, 2, 4 (NOT 0, 1, 2, 3)
# Label 3 does not exist in BraTS!
BRATS_COLORS = {
    0: (0, 0, 0),           # Background - Black
    1: (255, 0, 0),         # NCR (Necrotic Core) - Red
    2: (0, 255, 0),         # ED (Edema) - Green  
    4: (0, 0, 255),         # ET (Enhancing Tumor) - Blue
}

BRATS_CLASSES = {
    0: 'Background',
    1: 'NCR',  # Label 1: Necrotic and Non-Enhancing Tumor Core
    2: 'ED',   # Label 2: Peritumoral Edema
    4: 'ET',   # Label 4: GD-Enhancing Tumor (NOTE: label is 4, not 3!)
}

# For iteration over tumor classes (excluding background)
TUMOR_CLASSES = [1, 2, 4]


# ============================================================================
# Advanced Metrics
# ============================================================================

def compute_dice_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Dice Score for a specific class.
    
    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    
    Args:
        pred: Predicted segmentation (H, W, D)
        target: Ground truth segmentation (H, W, D)
        class_id: Class to compute Dice for
        
    Returns:
        float: Dice score [0, 1], higher is better
    """
    pred_mask = (pred == class_id).astype(np.float32)
    target_mask = (target == class_id).astype(np.float32)
    
    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    
    if union == 0:
        # Both masks empty - perfect match
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice


def compute_hausdorff_95(pred: np.ndarray, target: np.ndarray, class_id: int, 
                         spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95).
    
    HD measures the maximum distance from a point in one set to the closest
    point in the other set. HD95 uses the 95th percentile to reduce sensitivity
    to outliers.
    
    Args:
        pred: Predicted segmentation (H, W, D)
        target: Ground truth segmentation (H, W, D)
        class_id: Class to compute HD95 for
        spacing: Voxel spacing (mm) for each dimension
        
    Returns:
        float: HD95 distance in mm, lower is better. Returns inf if either mask is empty.
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)
    
    # Check if either mask is empty
    if not np.any(pred_mask) or not np.any(target_mask):
        return np.inf
    
    # Compute surface points (boundary voxels)
    pred_border = pred_mask ^ np.pad(pred_mask, ((1,1), (1,1), (1,1)), mode='constant')[1:-1, 1:-1, 1:-1]
    target_border = target_mask ^ np.pad(target_mask, ((1,1), (1,1), (1,1)), mode='constant')[1:-1, 1:-1, 1:-1]
    
    # Get coordinates of border voxels
    pred_coords = np.argwhere(pred_border)
    target_coords = np.argwhere(target_border)
    
    # Scale coordinates by spacing
    pred_coords_scaled = pred_coords * np.array(spacing)
    target_coords_scaled = target_coords * np.array(spacing)
    
    # Compute distances from pred to target
    distances_pred_to_target = np.min(
        np.linalg.norm(pred_coords_scaled[:, None] - target_coords_scaled[None, :], axis=2),
        axis=1
    )
    
    # Compute distances from target to pred
    distances_target_to_pred = np.min(
        np.linalg.norm(target_coords_scaled[:, None] - pred_coords_scaled[None, :], axis=2),
        axis=1
    )
    
    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    
    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)
    return hd95


def compute_sensitivity(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Sensitivity (Recall, True Positive Rate).
    
    Sensitivity = TP / (TP + FN)
    
    Measures the proportion of actual positives correctly identified.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_id: Class to compute sensitivity for
        
    Returns:
        float: Sensitivity [0, 1], higher is better
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)
    
    tp = np.sum(pred_mask & target_mask)  # True Positives
    fn = np.sum(~pred_mask & target_mask)  # False Negatives
    
    if tp + fn == 0:
        return 1.0  # No positive samples
    
    sensitivity = tp / (tp + fn)
    return sensitivity


def compute_specificity(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    Measures the proportion of actual negatives correctly identified.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_id: Class to compute specificity for
        
    Returns:
        float: Specificity [0, 1], higher is better
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)
    
    tn = np.sum(~pred_mask & ~target_mask)  # True Negatives
    fp = np.sum(pred_mask & ~target_mask)  # False Positives
    
    if tn + fp == 0:
        return 1.0  # No negative samples
    
    specificity = tn / (tn + fp)
    return specificity


def compute_precision(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Precision (Positive Predictive Value).
    
    Precision = TP / (TP + FP)
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_id: Class to compute precision for
        
    Returns:
        float: Precision [0, 1], higher is better
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)
    
    tp = np.sum(pred_mask & target_mask)
    fp = np.sum(pred_mask & ~target_mask)
    
    if tp + fp == 0:
        return 1.0  # No positive predictions
    
    precision = tp / (tp + fp)
    return precision


def compute_iou(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    IoU = |pred ∩ target| / |pred ∪ target|
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_id: Class to compute IoU for
        
    Returns:
        float: IoU [0, 1], higher is better
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)
    
    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask | target_mask)
    
    if union == 0:
        return 1.0  # Both masks empty
    
    iou = intersection / union
    return iou


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, 
                       class_id: int, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Compute all metrics for a specific class.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_id: Class to evaluate
        spacing: Voxel spacing in mm
        
    Returns:
        Dict with all metric values
    """
    metrics = {
        'dice': compute_dice_score(pred, target, class_id),
        'hd95': compute_hausdorff_95(pred, target, class_id, spacing),
        'sensitivity': compute_sensitivity(pred, target, class_id),
        'specificity': compute_specificity(pred, target, class_id),
        'precision': compute_precision(pred, target, class_id),
        'iou': compute_iou(pred, target, class_id),
    }
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def decode_segmentation_to_rgb(segmentation: np.ndarray) -> np.ndarray:
    """
    Convert 2D segmentation map to RGB image using BraTS colors.
    
    Args:
        segmentation: 2D array with class indices (H, W)
        
    Returns:
        RGB image (H, W, 3) as uint8
    """
    h, w = segmentation.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in BRATS_COLORS.items():
        mask = segmentation == class_id
        rgb[mask] = color
    
    return rgb


def visualize_slice(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, 
                    slice_idx: int, save_path: str = None):
    """
    Visualize a single slice with ground truth and prediction.
    
    Args:
        image: Input image (4, D, H, W) - multi-modal MRI
        gt_mask: Ground truth mask (D, H, W)
        pred_mask: Predicted mask (D, H, W)
        slice_idx: Which slice to visualize
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display individual modalities
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    for i in range(4):
        row = i // 3
        col = i % 3
        if i < 4:
            axes[row, col].imshow(image[i, slice_idx, :, :], cmap='gray')
            axes[row, col].set_title(modality_names[i])
            axes[row, col].axis('off')
    
    # Display ground truth
    gt_rgb = decode_segmentation_to_rgb(gt_mask[slice_idx, :, :])
    axes[1, 1].imshow(gt_rgb)
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].axis('off')
    
    # Display prediction
    pred_rgb = decode_segmentation_to_rgb(pred_mask[slice_idx, :, :])
    axes[1, 2].imshow(pred_rgb)
    axes[1, 2].set_title('Prediction')
    axes[1, 2].axis('off')
    
    # Create legend (only tumor classes, no background)
    legend_patches = [
        mpatches.Patch(color=np.array(BRATS_COLORS[i])/255, label=BRATS_CLASSES[i])
        for i in TUMOR_CLASSES  # [1, 2, 4]
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, frameon=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_figure(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray,
                             metrics: Dict[str, float], save_path: str = None):
    """
    Create comprehensive comparison figure with metrics.
    
    Shows axial, sagittal, and coronal views with metrics overlay.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)
    
    # Get middle slices for each view
    d, h, w = gt_mask.shape
    mid_axial = d // 2
    mid_sagittal = h // 2
    mid_coronal = w // 2
    
    # Axial view (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image[0, mid_axial, :, :], cmap='gray')
    ax1.set_title('Axial - FLAIR')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    gt_rgb = decode_segmentation_to_rgb(gt_mask[mid_axial, :, :])
    ax2.imshow(gt_rgb)
    ax2.set_title('Axial - Ground Truth')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    pred_rgb = decode_segmentation_to_rgb(pred_mask[mid_axial, :, :])
    ax3.imshow(pred_rgb)
    ax3.set_title('Axial - Prediction')
    ax3.axis('off')
    
    # Sagittal view (middle row)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(image[0, :, mid_sagittal, :], cmap='gray')
    ax4.set_title('Sagittal - FLAIR')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    gt_rgb_sag = decode_segmentation_to_rgb(gt_mask[:, mid_sagittal, :])
    ax5.imshow(gt_rgb_sag)
    ax5.set_title('Sagittal - Ground Truth')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    pred_rgb_sag = decode_segmentation_to_rgb(pred_mask[:, mid_sagittal, :])
    ax6.imshow(pred_rgb_sag)
    ax6.set_title('Sagittal - Prediction')
    ax6.axis('off')
    
    # Coronal view (bottom row)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(image[0, :, :, mid_coronal], cmap='gray')
    ax7.set_title('Coronal - FLAIR')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 1])
    gt_rgb_cor = decode_segmentation_to_rgb(gt_mask[:, :, mid_coronal])
    ax8.imshow(gt_rgb_cor)
    ax8.set_title('Coronal - Ground Truth')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[2, 2])
    pred_rgb_cor = decode_segmentation_to_rgb(pred_mask[:, :, mid_coronal])
    ax9.imshow(pred_rgb_cor)
    ax9.set_title('Coronal - Prediction')
    ax9.axis('off')
    
    # Metrics panel (right side)
    ax_metrics = fig.add_subplot(gs[:, 3])
    ax_metrics.axis('off')
    
    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    for class_id in TUMOR_CLASSES:  # [1, 2, 4]
        class_name = BRATS_CLASSES[class_id]
        metrics_text += f"{class_name}:\n"
        metrics_text += f"  Dice: {metrics[f'{class_name}_dice']:.4f}\n"
        metrics_text += f"  HD95: {metrics[f'{class_name}_hd95']:.2f} mm\n"
        metrics_text += f"  Sens: {metrics[f'{class_name}_sensitivity']:.4f}\n"
        metrics_text += f"  Spec: {metrics[f'{class_name}_specificity']:.4f}\n"
        metrics_text += f"  Prec: {metrics[f'{class_name}_precision']:.4f}\n"
        metrics_text += f"  IoU:  {metrics[f'{class_name}_iou']:.4f}\n"
        metrics_text += "\n"
    
    ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # Legend (only tumor classes)
    legend_patches = [
        mpatches.Patch(color=np.array(BRATS_COLORS[i])/255, label=BRATS_CLASSES[i])
        for i in TUMOR_CLASSES  # [1, 2, 4]
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, frameon=False)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, 
                   device: torch.device, model_name: str,
                   output_dir: str = 'reports', visualize: bool = False,
                   num_visualizations: int = 5) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on test dataset.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        model_name: Name of the model for saving results
        output_dir: Directory to save reports
        visualize: Whether to generate visualization figures
        num_visualizations: Number of samples to visualize
        
    Returns:
        Dictionary with all evaluation results
    """
    model.eval()
    model.to(device)
    
    # Create output directories
    report_dir = Path(output_dir) / model_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = report_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Storage for all results
    all_metrics = []
    class_metrics = {class_id: [] for class_id in TUMOR_CLASSES}  # {1: [], 2: [], 4: []}
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on Test Dataset")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move to device
            images = images.to(device).float()
            masks = masks.to(device).long()
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (argmax over class dimension)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert to numpy for metric computation
            pred_np = predictions.cpu().numpy()[0]  # (D, H, W)
            target_np = masks.cpu().numpy()[0, 0]  # (D, H, W)
            image_np = images.cpu().numpy()[0]  # (4, D, H, W)
            
            # Compute metrics for each class
            sample_metrics = {'sample_id': idx}
            
            for class_id in TUMOR_CLASSES:  # [1, 2, 4] - correct BraTS labels
                class_name = BRATS_CLASSES[class_id]
                metrics = compute_all_metrics(pred_np, target_np, class_id)
                
                # Store with class prefix
                for metric_name, value in metrics.items():
                    sample_metrics[f'{class_name}_{metric_name}'] = value
                    class_metrics[class_id].append(value)
            
            all_metrics.append(sample_metrics)
            
            # Generate visualizations for first N samples
            if visualize and idx < num_visualizations:
                # Find slice with most tumor
                tumor_slices = np.sum(target_np > 0, axis=(1, 2))
                best_slice = np.argmax(tumor_slices)
                
                # Save comparison figure
                save_path = vis_dir / f'sample_{idx:03d}_comparison.png'
                create_comparison_figure(image_np, target_np, pred_np, 
                                        sample_metrics, save_path)
    
    # ========================================================================
    # Compute Summary Statistics
    # ========================================================================
    
    print("\n" + "="*80)
    print("Computing Summary Statistics...")
    print("="*80 + "\n")
    
    summary = {}
    
    for class_id in TUMOR_CLASSES:  # [1, 2, 4]
        class_name = BRATS_CLASSES[class_id]
        
        for metric_name in ['dice', 'hd95', 'sensitivity', 'specificity', 'precision', 'iou']:
            values = [m[f'{class_name}_{metric_name}'] for m in all_metrics]
            
            # Filter out inf values for HD95
            if metric_name == 'hd95':
                values = [v for v in values if not np.isinf(v)]
            
            if values:
                summary[f'{class_name}_{metric_name}_mean'] = np.mean(values)
                summary[f'{class_name}_{metric_name}_std'] = np.std(values)
                summary[f'{class_name}_{metric_name}_median'] = np.median(values)
                summary[f'{class_name}_{metric_name}_min'] = np.min(values)
                summary[f'{class_name}_{metric_name}_max'] = np.max(values)
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    print("\nSaving results...")
    
    # Save detailed metrics to JSON
    with open(report_dir / 'metrics_detailed.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save summary to JSON
    with open(report_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary to CSV (more readable)
    import csv
    with open(report_dir / 'metrics_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max'])
        
        for class_id in TUMOR_CLASSES:  # [1, 2, 4]
            class_name = BRATS_CLASSES[class_id]
            writer.writerow([])  # Empty row for spacing
            writer.writerow([f'--- {class_name} ---', '', '', '', '', ''])
            
            for metric_name in ['dice', 'hd95', 'sensitivity', 'specificity', 'precision', 'iou']:
                writer.writerow([
                    metric_name.upper(),
                    f"{summary[f'{class_name}_{metric_name}_mean']:.4f}",
                    f"{summary[f'{class_name}_{metric_name}_std']:.4f}",
                    f"{summary[f'{class_name}_{metric_name}_median']:.4f}",
                    f"{summary[f'{class_name}_{metric_name}_min']:.4f}",
                    f"{summary[f'{class_name}_{metric_name}_max']:.4f}",
                ])
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"Evaluation Results for {model_name}")
    print("="*80 + "\n")
    
    for class_id in TUMOR_CLASSES:  # [1, 2, 4]
        class_name = BRATS_CLASSES[class_id]
        print(f"{class_name}:")
        print(f"  Dice:        {summary[f'{class_name}_dice_mean']:.4f} ± {summary[f'{class_name}_dice_std']:.4f}")
        print(f"  HD95:        {summary[f'{class_name}_hd95_mean']:.2f} ± {summary[f'{class_name}_hd95_std']:.2f} mm")
        print(f"  Sensitivity: {summary[f'{class_name}_sensitivity_mean']:.4f} ± {summary[f'{class_name}_sensitivity_std']:.4f}")
        print(f"  Specificity: {summary[f'{class_name}_specificity_mean']:.4f} ± {summary[f'{class_name}_specificity_std']:.4f}")
        print(f"  Precision:   {summary[f'{class_name}_precision_mean']:.4f} ± {summary[f'{class_name}_precision_std']:.4f}")
        print(f"  IoU:         {summary[f'{class_name}_iou_mean']:.4f} ± {summary[f'{class_name}_iou_std']:.4f}")
        print()
    
    print(f"Results saved to: {report_dir}")
    print("="*80 + "\n")
    
    return {
        'detailed_metrics': all_metrics,
        'summary': summary
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BraTS 2021 segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, help='Path to model config file')
    parser.add_argument('--model', type=str, help='Path to trained model weights (.pt file)')
    parser.add_argument('--data-dir', type=str, default='data/processed', 
                       help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory to save evaluation reports')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization figures')
    parser.add_argument('--num-vis', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluate all models in models/ directory')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test dataset
    CLASSES = ['background', 'NCR', 'ED', 'ET']
    x_test_dir = os.path.join(args.data_dir, 'X_test')
    y_test_dir = os.path.join(args.data_dir, 'y_test')
    
    test_dataset = Dataset(x_test_dir, y_test_dir, CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples\n")
    
    # Evaluate model(s)
    if args.evaluate_all:
        # Evaluate all models in models/ directory
        models_dir = Path('models')
        model_files = list(models_dir.glob('*.pt'))
        
        if not model_files:
            print("No model files found in models/ directory!")
            return
        
        print(f"Found {len(model_files)} models to evaluate\n")
        
        for model_path in model_files:
            model_name = model_path.stem
            
            # Find corresponding config
            config_path = Path('configs') / f'{model_name.lower()}_config.yaml'
            if not config_path.exists():
                print(f"Config not found for {model_name}, skipping...")
                continue
            
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load model
            model = get_model(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Evaluate
            evaluate_model(
                model, test_loader, device, model_name,
                args.output_dir, args.visualize, args.num_vis
            )
    
    else:
        # Evaluate single model
        if not args.config or not args.model:
            print("Please provide --config and --model, or use --evaluate-all")
            return
        
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        
        # Load model
        model = get_model(config)
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        # Evaluate
        evaluate_model(
            model, test_loader, device, model_name,
            args.output_dir, args.visualize, args.num_vis
        )


if __name__ == '__main__':
    main()