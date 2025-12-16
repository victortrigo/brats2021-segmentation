"""
Dataset Module for BraTS 2021 Brain Tumor Segmentation.

This module provides dataset loading and preprocessing utilities for the Brain Tumor
Segmentation (BraTS) challenge. It handles multi-modal MRI data (FLAIR, T1, T1CE, T2)
and corresponding segmentation masks with multiple tumor regions.

Key Features:
    - Multi-modal MRI data loading (4 modalities: FLAIR, T1, T1CE, T2)
    - Flexible normalization strategies (z-score, min-max, none)
    - Brain-masked normalization to focus on tissue regions
    - Automatic validation of data integrity
    - Subset creation for efficient training/validation

Classes:
    Dataset: Main dataset class for loading BraTS data
    
Functions:
    create_subset: Creates DataLoader from dataset subsets

Example:
    >>> from dataset import Dataset, create_subset
    >>> dataset = Dataset(
    ...     images_dir='data/X_train',
    ...     masks_dir='data/y_train',
    ...     classes=['background', 'NCR', 'ED', 'ET'],
    ...     normalize='z-score'
    ... )
    >>> subset, loader = create_subset(dataset, subset_size=10, batch_size=2)
    
References:
    BraTS 2021: https://www.med.upenn.edu/cbica/brats2021/
"""

import os
import torch
import nibabel as nib
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, Subset

# Tumor segmentation classes following BraTS convention
CLASSES = ['background', 'NCR', 'ED', 'ET']
"""
Segmentation classes for BraTS tumors:
    - background (0): Non-tumor tissue
    - NCR (1): Necrotic and Non-Enhancing Tumor core
    - ED (2): Peritumoral Edema
    - ET (4): GD-Enhancing Tumor
"""



class Dataset(BaseDataset):
    """
    PyTorch Dataset for BraTS 2021 Brain Tumor Segmentation Challenge.

    This class handles loading of multi-modal MRI volumes and their corresponding
    segmentation masks. It performs validation to ensure data integrity and applies
    configurable normalization strategies optimized for medical imaging.

    The dataset expects the following directory structure:
        images_dir/
            patient_001/
                patient_001_flair.nii.gz
                patient_001_t1.nii.gz
                patient_001_t1ce.nii.gz
                patient_001_t2.nii.gz
            patient_002/
                ...
        masks_dir/
            patient_001/
                patient_001_seg.nii.gz
            patient_002/
                ...

    Attributes:
        ids (List[str]): List of valid patient IDs found in both directories
        images_fps (List[str]): Full paths to patient image folders
        masks_fps (List[str]): Full paths to patient mask folders
        class_values (List[int]): Integer values for segmentation classes
        augmentation (callable): Optional data augmentation function
        preprocessing (callable): Optional preprocessing function
        normalize (str): Normalization strategy ('z-score', 'min-max', 'none')

    Args:
        images_dir (str): Root directory containing patient image folders
        masks_dir (str): Root directory containing patient mask folders
        classes (List[str], optional): List of class names to segment.
            Defaults to None (all classes).
        augmentation (callable, optional): Function for data augmentation.
            Should accept (image, mask) and return (augmented_image, augmented_mask).
        preprocessing (callable, optional): Function for preprocessing.
            Should accept (image, mask) and return (preprocessed_image, preprocessed_mask).
        normalize (str): Normalization method. Options:
            - 'z-score': Standardization (mean=0, std=1) on brain tissue only
            - 'min-max': Scale to [0, 1] range on brain tissue only
            - 'none': No normalization applied
    """
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        classes: List[str] = None,
        augmentation: Optional[callable] = None,
        preprocessing: Optional[callable] = None,
        normalize: str = 'z-score'  
    ) -> None:
        """Initialize the BraTS dataset with validation and filtering."""

        # Valid MRI modality filenames following BraTS convention
        valid_modalities: List[str] = [
            'flair.nii.gz', 
            't1.nii.gz', 
            't1ce.nii.gz', 
            't2.nii.gz'
        ]

        # Retrieve all patient IDs from both directories
        try:
            ids_x_temp = set([
                d for d in os.listdir(images_dir) 
                if os.path.isdir(os.path.join(images_dir, d))
            ])
            ids_y_temp = set([
                d for d in os.listdir(masks_dir) 
                if os.path.isdir(os.path.join(masks_dir, d))
            ])
        except FileNotFoundError as e:
            raise RuntimeError(f"Directory not found: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error listing directories: {str(e)}") from e

        # Find patient IDs that exist in BOTH directories
        common_ids = sorted(list(ids_x_temp.intersection(ids_y_temp)))
        if not common_ids:
            raise RuntimeError(
                "No common patient IDs found between image and mask directories. "
                "Please verify directory structure and contents."
            )

        # Filter to ensure each patient has all required files
        self.ids: List[str] = []
        for patient_id in common_ids:
            try:
                imgs = os.listdir(os.path.join(images_dir, patient_id))
                msks = os.listdir(os.path.join(masks_dir, patient_id))
            except Exception as e:
                print(f"Warning: Could not access folder {patient_id}: {e}")
                continue

            # Verify presence of all MRI modalities and segmentation mask
            has_all_modalities = any(
                f.lower().endswith(tuple(valid_modalities)) for f in imgs
            )
            has_segmentation = any(
                f.lower().endswith('seg.nii.gz') for f in msks
            )
            
            if has_all_modalities and has_segmentation:
                self.ids.append(patient_id)

        # Construct full paths for all valid patients
        self.images_fps: List[str] = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_fps: List[str] = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

        # Convert class names to integer indices
        self.class_values: List[int] =(
            [CLASSES.index(cls) for cls in classes] if classes else [] #()
        )

        # Store transformation functions
        self.augmentation: Optional[callable] = augmentation
        self.preprocessing: Optional[callable] = preprocessing
        self.normalize: str = normalize

    def _normalize_modality(self, modality: np.ndarray) -> np.ndarray:
        """
        Normalize a single MRI modality using brain-masked techniques.
        
        This method applies normalization only to brain tissue voxels (where intensity > 0),
        preserving the background at zero. This approach follows BraTS best practices and
        helps the model focus on meaningful tissue variations.
        
        Normalization Strategies:
            z-score: Standardization to mean=0, std=1
                - Most commonly used by BraTS challenge winners
                - Robust to intensity variations across scanners
                - Formula: (x - μ) / σ where μ, σ computed on brain voxels only
                
            min-max: Linear scaling to [0, 1] range
                - Preserves relative intensity relationships
                - More sensitive to outliers than z-score
                - Formula: (x - min) / (max - min) on brain voxels only
                
            none: No normalization
                - Returns raw intensities as float32
                - Useful for debugging or custom preprocessing pipelines
        Args:
            modality (np.ndarray): 3D array of MRI intensities, shape (D, H, W)
            
        Returns:
            np.ndarray: Normalized 3D array as float32, same shape as input
        
        """
        # Return early if no normalization requested
        if self.normalize == 'none':
            return modality.astype(np.float32)
        
        # Create binary mask for brain tissue (exclude background)
        brain_mask = modality > 0
        
        # Handle edge case: empty brain mask
        if not brain_mask.any():
            return modality.astype(np.float32)
        
        # Apply z-score normalization (standardization)
        if self.normalize == 'z-score':
            mean = modality[brain_mask].mean()
            std = modality[brain_mask].std()
            
            # Avoid division by zero for constant-valued scans
            if std > 1e-8:  
                modality = modality.astype(np.float32)
                modality[brain_mask] = (modality[brain_mask] - mean) / std
            else:
                modality = modality.astype(np.float32)
                modality[brain_mask] = 0
            
            # Ensure background remains zero
            modality[~brain_mask] = 0
            
        # Apply min-max normalization (scaling to [0, 1])
        elif self.normalize == 'min-max':
            min_val = modality[brain_mask].min()
            max_val = modality[brain_mask].max()
            
            # Avoid division by zero for constant-valued scans
            if max_val - min_val > 1e-8:
                modality = modality.astype(np.float32)
                modality[brain_mask] = (
                    (modality[brain_mask] - min_val) / (max_val - min_val)
                )
            else:
                modality = modality.astype(np.float32)
                modality[brain_mask] = 0
            
             # Ensure background remains zero
            modality[~brain_mask] = 0
            
        return modality.astype(np.float32)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of valid patient scans
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga y devuelve una muestra del dataset en el índice `i`.

        Args:
            i (int): Índice de la muestra a cargar.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Una tupla que contiene la imagen y la máscara.
        """
        # Get paths to patient-specific folders
        folder_image: str = self.images_fps[i]
        folder_mask: str = self.masks_fps[i]
        
        # List all files in patient folders
        files_image: List[str] = os.listdir(folder_image)
        files_mask: List[str] = os.listdir(folder_mask)

        # Initialize containers for multi-modal data
        image_data: List[np.ndarray] = []
        mask_data: List[np.ndarray] = []
        
        # Load all MRI modalities (FLAIR, T1, T1CE, T2)
        for file in files_image:
            file_path: str = os.path.join(folder_image, file)

            if (file.endswith('flair.nii.gz') or
                file.endswith('t1.nii.gz') or
                file.endswith('t1ce.nii.gz') or 
                file.endswith('t2.nii.gz')):
                
                # Load NIfTI file and convert to numpy array
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))

                # Apply brain-masked normalization
                img_normalized = self._normalize_modality(img_new)
                image_data.append(img_normalized)

        # Load segmentation mask
        for file in files_mask:
            file_path: str = os.path.join(folder_mask, file)

            # Load only the segmentation file
            if file.endswith('seg.nii.gz'):
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))
                mask_data.append(img_new)

        # Stack modalities into multi-channel arrays
        # Shape: (4, D, H, W) for images, (1, D, H, W) for masks
        image_stacked: np.ndarray = np.asarray(image_data)
        mask_stacked: np.ndarray = np.asarray(mask_data)

        # Convert to PyTorch tensors with appropriate dtypes
        image_tensor = torch.from_numpy(image_stacked).float()
        mask_tensor = torch.from_numpy(mask_stacked).long()
        
        return image_tensor, mask_tensor


def create_subset(
        data: Dataset, 
        subset_size: int, 
        batch_size: int, 
        shuffle: bool = True
    ) -> Tuple[Dataset, DataLoader]:
    """
    Create a DataLoader from a random subset of the dataset.
    
    Subset Creation Process:
        1. If subset_size <= 0 or > dataset size, use full dataset
        2. Otherwise, randomly sample subset_size indices without replacement
        3. Create a PyTorch Subset wrapper around the original dataset
        4. Wrap the subset in a DataLoader with specified batch size
    
    Args:
        data (Dataset): The complete BraTS Dataset instance
        subset_size (int): Number of samples to include in subset.
            If <= 0 or > len(data), the entire dataset is used.
            This allows seamless switching between subset and full training.
        batch_size (int): Batch size for the DataLoader.
            Typical values: 1-4 for 3D medical imaging (memory intensive).
        shuffle (bool): Whether to shuffle data during iteration.
            - True: Recommended for training (improves generalization)
            - False: Recommended for validation/testing (reproducibility)
            Default: True

    Returns:
        Tuple[Subset, DataLoader]: A tuple containing:
            - subset (Subset): PyTorch Subset with selected indices
            - loader (DataLoader): DataLoader ready for iteration
    """

    # Use full dataset if subset_size is invalid or <= 0
    if subset_size <= 0 or subset_size > len(data):
        subset_size = len(data)

    # Randomly select unique indices without replacement
    indices = np.random.choice(len(data), subset_size, replace=False)

    # Create PyTorch Subset wrapper
    subset = Subset(data, indices)
    
    # Create DataLoader with specified configuration
    loader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0
    )
    
    return subset, loader


if __name__ == "__main__":
    """
    Test script to validate dataset functionality and display statistics.
    """
    
    # Configuration
    data_base_path = 'data/processed'
    classes = ['background', 'NCR', 'ED', 'ET']
    subsets = ['train', 'val', 'test']
    sizes = {}

    # Check sizes of all data splits
    print("=" * 60)
    print("VALIDATING DATASET STRUCTURE")
    print("=" * 60)

    for subset in subsets:
        images_dir = os.path.join(data_base_path, f'X_{subset}')
        masks_dir = os.path.join(data_base_path, f'y_{subset}')

        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            dataset = Dataset(
                images_dir, 
                masks_dir, 
                classes,
                normalize='z-score'
            )
            sizes[subset] = len(dataset)
        else:
            sizes[subset] = 0

    print("\nDataset Split Sizes:")
    print("-" * 40)
    for subset in subsets:
        print(f"{subset.capitalize()}: {sizes[subset]} samples")
    print("-" * 40)

    # Test normalization methods on first training sample
    if sizes['train'] > 0:
        print("\n" + "=" * 60)
        print("TESTING NORMALIZATION METHODS")
        print("=" * 60)

        for norm_type in ['none', 'min-max', 'z-score']:
            print(f"\n{'='*50}")
            print(f"NORMALIZATION: {norm_type.upper()}")
            print('='*50)

            # Create dataset with specific normalization
            dataset = Dataset(
                os.path.join(data_base_path, 'X_train'),
                os.path.join(data_base_path, 'y_train'),
                classes,
                normalize=norm_type
            )

            # Load first sample
            img, msk = dataset[0]

            # Display image statistics
            print(f"\nImage Tensor:")
            print(f"  dtype: {img.dtype}")
            print(f"  shape: {tuple(img.shape)}")
            print(f"  min: {img.min().item():.4f}")
            print(f"  max: {img.max().item():.4f}")

            # Display per-modality statistics (only brain tissue)
            modalities = ['FLAIR', 'T1CE', 'T1', 'T2']
            print(f"\nPer-Modality Statistics (Brain Tissue Only):")
            print("-" * 40)
            for idx, mod in enumerate(modalities):
                channel = img[idx]
                brain_mask = channel != 0

                if brain_mask.sum() > 0:
                    brain_voxels = channel[brain_mask]
                    print(f"{mod}: mean={brain_voxels.mean():.4f}, "
                          f"std={brain_voxels.std():.4f}")
                else:
                    print(f"  {mod:6s}: No brain tissue found")
            
            # Display mask statistics
            print(f"\nMask Tensor:")
            print(f"  dtype: {msk.dtype}")
            print(f"  shape: {tuple(msk.shape)}")

            # Class distribution (BraTS uses labels: 0, 1, 2, 4)
            unique, counts = np.unique(msk.numpy(), return_counts=True)
            print(f"\nClass Distribution:")
            print("-" * 40)

            # Mapping BraTS labels to class names
            label_to_class = {
                0: 'background',
                1: 'NCR',
                2: 'ED', 
                4: 'ET'
            }

            for val, count in zip(unique, counts):
                class_name = label_to_class.get(int(val), 'unknown')
                # class_name = classes[int(val)] if int(val) < len(classes) else "unknown"
                percentage = (count / msk.numel()) * 100
                print(f"  Class {val} ({class_name:10s}): {count:8d} voxels "
                      f"({percentage:5.2f}%)")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)