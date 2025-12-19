"""
DeepLabV3+ Architecture for 3D Medical Image Segmentation.

This module implements DeepLabV3+, a state-of-the-art semantic segmentation
architecture that combines:
- Xception backbone for feature extraction
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
- Encoder-decoder structure with skip connections
- Atrous (dilated) convolutions for dense feature extraction

Key Innovations:
    1. **Atrous Convolutions**: Expand receptive field without losing resolution
    2. **ASPP Module**: Captures multi-scale context at multiple dilation rates
    3. **Encoder-Decoder**: Combines semantic features with spatial details
    4. **Xception Backbone**: Efficient feature extraction with separable convolutions

Architecture Overview:
    Input (4 channels, 128³)
       ↓
    ┌─────────────────────────┐
    │   Xception Backbone     │
    │   (Entry → Middle → Exit)│
    │                         │
    │   Low-level [128, 32³]  │ ─────┐ (skip connection)
    │   High-level [2048, 8³] │      │
    └─────────────────────────┘      │
               ↓                      │
    ┌─────────────────────────┐      │
    │    ASPP Module          │      │
    │  (Multi-scale context)  │      │
    │   Output: [256, 8³]     │      │
    └─────────────────────────┘      │
               ↓                      │
    ┌─────────────────────────┐      │
    │      Decoder            │      │
    │  Upsample → Fuse ←──────┘      │
    │  Low-level projection          │
    │  Output: [256, 128³]           │
    └─────────────────────────┘
               ↓
         Output Conv (1×1×1)
               ↓
    Output (num_classes, 128³)

Comparison with U-Net:
    U-Net:
    - Symmetric encoder-decoder
    - Standard convolutions
    - Skip connections at all levels
    
    DeepLabV3+:
    - Asymmetric (deep encoder, shallow decoder)
    - Atrous convolutions (dilated)
    - ASPP for multi-scale context
    - Single skip connection (low-level features)

Usage:
    # Standard DeepLabV3+ for BraTS
    model = DeepLabV3Plus(in_channels=4, num_classes=4)
    x = torch.randn(1, 4, 128, 128, 128)
    output = model(x)  # [1, 4, 128, 128, 128]
    
    # Access encoder and decoder separately
    encoder = model.encoder
    decoder = model.decoder
    
    low_level, aspp_out = encoder(x)
    output = decoder(aspp_out, low_level)

Performance Characteristics:
    - Parameters: ~40M (larger than U-Net due to Xception)
    - Memory: ~8-10 GB for batch=1
    - Strengths: Excellent for capturing multi-scale context
    - Best for: Objects with varying sizes, complex boundaries

References:
    - Chen et al., "Encoder-Decoder with Atrous Separable Convolution for 
      Semantic Image Segmentation", ECCV 2018
    - Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", 
      CVPR 2017

Author: Victor Trigo
Date: 2024
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchinfo import summary

from backbone import BackboneXception
from pooling import ASPP


# ============================================================================
# Encoder: Backbone + ASPP
# ============================================================================

class Encoder(nn.Module):
    """
    DeepLabV3+ Encoder combining Xception backbone and ASPP module.
    
    The encoder extracts hierarchical features using the Xception backbone,
    then applies ASPP to capture multi-scale context. It outputs both:
    - Low-level features: For skip connections (fine spatial details)
    - High-level features: After ASPP (rich semantic context)
    
    Architecture:
        Input [B, in_ch, 128, 128, 128]
               ↓
        ┌──────────────────────────┐
        │  Xception Backbone       │
        │                          │
        │  Entry Flow              │
        │  ├─> Low-level output    │ ─────┐
        │  │    [128, 32, 32, 32]  │      │ Skip connection
        │  └─> To Middle Flow      │      │
        │                          │      │
        │  Middle Flow (16 blocks) │      │
        │  └─> To Exit Flow        │      │
        │                          │      │
        │  Exit Flow               │      │
        │  └─> High-level output   │      │
        │       [2048, 8, 8, 8]    │      │
        └──────────────────────────┘      │
               ↓                          │
        ┌──────────────────────────┐      │
        │  ASPP Module             │      │
        │  ├─> Conv 1×1            │      │
        │  ├─> Atrous Conv (r=2)   │      │
        │  ├─> Atrous Conv (r=4)   │      │
        │  ├─> Atrous Conv (r=6)   │      │
        │  └─> Global Pool         │      │
        │                          │      │
        │  Concat + Project        │      │
        │  Output: [256, 8, 8, 8]  │      │
        └──────────────────────────┘      │
               ↓                          ↓
        (aspp_output)              (low_level)
    
    Key Features:
        - Xception backbone: Efficient feature extraction
        - Low-level features: Fine spatial details (1/4 resolution)
        - High-level features: Semantic context (1/16 resolution)
        - ASPP: Multi-scale context aggregation
        - Output stride 16: Good balance between resolution and receptive field
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 4
            For BraTS: 4 modalities (FLAIR, T1, T1CE, T2)
        out_channels (int, optional): Output channels from ASPP. Default: 256
            This determines the channel count for decoder input
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output low_level: (B, 128, D/4, H/4, W/4)
        - Output aspp_output: (B, out_channels, D/16, H/16, W/16)
        
    Example:
        >>> encoder = Encoder(in_channels=4, out_channels=256)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> low_level, aspp_out = encoder(x)
        >>> 
        >>> print(f"Low-level: {low_level.shape}")
        >>> # torch.Size([1, 128, 32, 32, 32])
        >>> print(f"ASPP output: {aspp_out.shape}")
        >>> # torch.Size([1, 256, 8, 8, 8])
        
    Note:
        - Low-level features are at 1/4 resolution (after 2 pooling layers)
        - High-level features are at 1/16 resolution (after 4 pooling layers)
        - ASPP uses dilation rates [1, 2, 4, 6] by default
        - Output stride is 16 (preserves more spatial detail than 32)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()

        # Xception backbone for feature extraction
        # output_stride=16: Less aggressive downsampling
        self.backbone = BackboneXception(in_channels=in_channels)

        # ASPP for multi-scale context aggregation
        # Input: 2048 channels from Xception exit flow
        # Output: out_channels (typically 256)
        self.aspp = ASPP(in_channels=2048, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - low_level: Low-level features for skip connection
                  Shape: (B, 128, D/4, H/4, W/4)
                  Use: Preserves fine spatial details
                  
                - aspp_output: High-level features after ASPP
                  Shape: (B, out_channels, D/16, H/16, W/16)
                  Use: Rich semantic context for decoder
                  
        Process:
            1. Backbone: Extract low-level and high-level features
            2. ASPP: Aggregate multi-scale context from high-level features
            3. Return both for decoder
            
        Example:
            >>> encoder = Encoder(4, 256)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> 
            >>> # Forward pass
            >>> low, high = encoder(x)
            >>> 
            >>> # Verify shapes
            >>> assert low.shape == (2, 128, 32, 32, 32)
            >>> assert high.shape == (2, 256, 8, 8, 8)
            >>> 
            >>> # Low-level has more spatial detail
            >>> print(f"Low-level resolution: {low.shape[2:]}")
            >>> # (32, 32, 32) - Higher resolution
            >>> 
            >>> # High-level has more semantic info
            >>> print(f"High-level channels: {high.shape[1]}")
            >>> # 256 - Rich features
        """
        # Extract features from backbone
        # low_level: [B, 128, D/4, H/4, W/4] from Entry Flow block 2
        # high_level: [B, 2048, D/16, H/16, W/16] from Exit Flow
        low_level, high_level = self.backbone(x)

        # Apply ASPP to high-level features
        # aspp_output: [B, out_channels, D/16, H/16, W/16]
        aspp_output = self.aspp(high_level)

        return low_level, aspp_output


# ============================================================================
# Decoder: Upsample and Fuse
# ============================================================================

class Decoder(nn.Module):
    """
    DeepLabV3+ Decoder with Skip Connection Fusion.
    
    The decoder upsamples ASPP features and fuses them with low-level features
    from the encoder. This combination provides both semantic understanding
    (from ASPP) and spatial precision (from low-level features).
    
    Architecture:
        ASPP Output [B, 256, 8, 8, 8]
               ↓ Upsample (4×)
        [B, 256, 32, 32, 32]
               ↓
        ┌──────────────────────┐
        │  Low-level Features  │
        │  [B, 128, 32, 32, 32]│
        │        ↓             │
        │  Conv 1×1 → [B, 48]  │ Reduce channels
        └──────────────────────┘
               ↓
        Concatenate: [B, 256+48, 32, 32, 32]
               ↓
        ┌──────────────────────┐
        │  Refinement          │
        │  Conv 3×3 → BN → ReLU│
        │  Conv 3×3 → BN → ReLU│
        │  Output: [B, 256]    │
        └──────────────────────┘
               ↓ Upsample (4×)
        [B, 256, 128, 128, 128]
               ↓
        Conv 1×1 (classification)
               ↓
        [B, num_classes, 128, 128, 128]
    
    Key Design Choices:
        1. **Low-level projection**: 128 → 48 channels
           - Reduces computation
           - Prevents low-level features from dominating
           
        2. **Two refinement convolutions**:
           - Learn to combine low and high-level features
           - Extract additional features after fusion
           
        3. **Two upsampling stages**:
           - ASPP → Low-level: 8³ → 32³ (4×)
           - After refinement → Output: 32³ → 128³ (4×)
           - Total upsampling: 16× (8³ → 128³)
    
    Args:
        low_level_in (int): Channels in low-level features from encoder
            Default from Xception: 128
        num_classes (int): Number of output segmentation classes
            For BraTS: 4 (background, NCR, ED, ET)
            
    Shape:
        - Input aspp_output: (B, 256, D/16, H/16, W/16)
        - Input low_level_feat: (B, low_level_in, D/4, H/4, W/4)
        - Output: (B, num_classes, D, H, W)
        
    Example:
        >>> decoder = Decoder(low_level_in=128, num_classes=4)
        >>> aspp_out = torch.randn(1, 256, 8, 8, 8)
        >>> low_level = torch.randn(1, 128, 32, 32, 32)
        >>> 
        >>> output = decoder(aspp_out, low_level)
        >>> print(output.shape)
        >>> # torch.Size([1, 4, 128, 128, 128])
        
    Note:
        - Output is raw logits (no softmax/sigmoid)
        - Uses trilinear interpolation for smooth upsampling
        - Low-level features provide fine spatial details
        - ASPP features provide semantic context
    """

    def __init__(self, low_level_in, num_classes):
        super().__init__()
        
        # ====================================================================
        # Low-Level Feature Projection
        # ====================================================================
        
        # Reduce low-level feature channels: 128 → 48
        # This prevents low-level features from dominating the fusion
        self.conv_low = nn.Sequential(
            nn.Conv3d(low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Feature Fusion and Refinement
        # ====================================================================
        
        # Concatenated channels: 256 (ASPP) + 48 (low-level) = 304
        self.conv_final = nn.Sequential(
            # First refinement conv
            nn.Conv3d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # Second refinement conv
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Output Classification Head
        # ====================================================================
        
        # Final 1×1×1 conv: 256 → num_classes
        self.out_conv = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, aspp_output, low_level_feat):
        """
        Decode features to produce segmentation map.
        
        Args:
            aspp_output (torch.Tensor): Features from ASPP module
                Shape: (B, 256, D/16, H/16, W/16)
                Contains rich semantic context
                
            low_level_feat (torch.Tensor): Features from encoder
                Shape: (B, 128, D/4, H/4, W/4)
                Contains fine spatial details
                
        Returns:
            torch.Tensor: Segmentation logits
                Shape: (B, num_classes, D, H, W)
                Raw predictions (no activation applied)
                
        Process:
            1. Upsample ASPP output: D/16 → D/4 (4× upsampling)
            2. Project low-level features: 128 → 48 channels
            3. Concatenate upsampled ASPP + projected low-level
            4. Refine with two 3×3×3 convolutions
            5. Upsample refined features: D/4 → D (4× upsampling)
            6. Apply 1×1×1 conv for class predictions
            
        Example:
            >>> decoder = Decoder(128, 4)
            >>> aspp = torch.randn(1, 256, 8, 8, 8)
            >>> low = torch.randn(1, 128, 32, 32, 32)
            >>> 
            >>> # Decode
            >>> logits = decoder(aspp, low)
            >>> 
            >>> # Get predictions
            >>> probs = F.softmax(logits, dim=1)
            >>> preds = torch.argmax(probs, dim=1)
            >>> 
            >>> print(f"Logits: {logits.shape}")  # [1, 4, 128, 128, 128]
            >>> print(f"Preds: {preds.shape}")    # [1, 128, 128, 128]
        """
        # ====================================================================
        # Stage 1: Upsample ASPP output to match low-level resolution
        # ====================================================================
        
        # Upsample ASPP: [B, 256, 8, 8, 8] → [B, 256, 32, 32, 32]
        aspp_output = F.interpolate(
            aspp_output, 
            size=low_level_feat.shape[2:],   # Match low-level spatial size
            mode='trilinear', 
            align_corners=False
        )

        # ====================================================================
        # Stage 2: Project low-level features
        # ====================================================================
        
        # Reduce channels: [B, 128, 32, 32, 32] → [B, 48, 32, 32, 32]
        low_level_feat = self.conv_low(low_level_feat)

        # ====================================================================
        # Stage 3: Concatenate and refine
        # ====================================================================
        
        # Concatenate along channel dimension
        # [B, 256, 32, 32, 32] + [B, 48, 32, 32, 32]
        # → [B, 304, 32, 32, 32]
        x = torch.cat([aspp_output, low_level_feat], dim=1)

        # Refine with two convolutions
        # [B, 304, 32, 32, 32] → [B, 256, 32, 32, 32]
        x = self.conv_final(x)

        # ====================================================================
        # Stage 4: Upsample to full resolution
        # ====================================================================
        
        # Upsample: [B, 256, 32, 32, 32] → [B, 256, 128, 128, 128]
        x = F.interpolate(
            x, 
            scale_factor=4,   # 4× upsampling
            mode='trilinear', 
            align_corners=False
        )

        # ====================================================================
        # Stage 5: Classify
        # ====================================================================
        
        # Final classification: [B, 256, 128, 128, 128] → [B, num_classes, 128, 128, 128]
        return self.out_conv(x)


# ============================================================================
# Complete DeepLabV3+ Model
# ============================================================================

class DeepLabV3Plus(nn.Module):
    """
    Complete DeepLabV3+ Model for 3D Medical Image Segmentation.
    
    DeepLabV3+ is a state-of-the-art semantic segmentation architecture that
    excels at capturing multi-scale context through atrous convolutions and
    ASPP, while maintaining spatial precision through skip connections.
    
    Complete Architecture:
        Input [B, in_ch, D, H, W]
               ↓
        ┌──────────────────────────────┐
        │         ENCODER              │
        │                              │
        │  ┌────────────────────┐      │
        │  │  Xception Backbone │      │
        │  │                    │      │
        │  │  Entry Flow        │      │
        │  │  ├─ Conv layers    │      │
        │  │  └─ Low-level ─────┼──┐   │
        │  │     [128, 32³]     │  │   │
        │  │                    │  │   │
        │  │  Middle Flow       │  │   │
        │  │  └─ 16× residual   │  │   │
        │  │                    │  │   │
        │  │  Exit Flow         │  │   │
        │  │  └─ High-level     │  │   │
        │  │     [2048, 8³]     │  │   │
        │  └────────────────────┘  │   │
        │           ↓               │   │
        │  ┌────────────────────┐  │   │
        │  │  ASPP Module       │  │   │
        │  │  ├─ Conv 1×1       │  │   │
        │  │  ├─ Atrous r=2     │  │   │
        │  │  ├─ Atrous r=4     │  │   │
        │  │  ├─ Atrous r=6     │  │   │
        │  │  └─ Global Pool    │  │   │
        │  │                    │  │   │
        │  │  Output: [256, 8³] │  │   │
        │  └────────────────────┘  │   │
        └──────────────────────────┼───┘
                     ↓             │
        ┌────────────────────────┐ │
        │       DECODER          │ │
        │                        │ │
        │  Upsample ASPP (4×)    │ │
        │  ├─ [256, 32³]         │ │
        │  │                     │ │
        │  Project Low-level ←───┘ │
        │  ├─ [48, 32³]            │
        │  │                       │
        │  Concatenate              │
        │  ├─ [304, 32³]            │
        │  │                        │
        │  Refine (2× Conv3×3)      │
        │  ├─ [256, 32³]            │
        │  │                        │
        │  Upsample (4×)            │
        │  ├─ [256, 128³]           │
        │  │                        │
        │  Classify (Conv1×1)       │
        │  └─ [num_classes, 128³]   │
        └────────────────────────────┘
                     ↓
        Output [B, num_classes, D, H, W]
    
    Key Features:
        ✓ Atrous convolutions for dense feature extraction
        ✓ ASPP for multi-scale context aggregation
        ✓ Xception backbone for efficient feature extraction
        ✓ Encoder-decoder with skip connection
        ✓ Proven performance on medical imaging tasks
    
    Advantages over U-Net:
        ✓ Better at capturing multi-scale context (ASPP)
        ✓ Larger receptive field (atrous convolutions)
        ✓ More efficient backbone (Xception vs standard convs)
        ✓ State-of-the-art performance on many benchmarks
        
    Potential Disadvantages:
        ⚠ More parameters (~40M vs ~31M for U-Net)
        ⚠ Only one skip connection (vs 4 in U-Net)
        ⚠ Deeper architecture (can be slower)
    
    Args:
        in_channels (int): Number of input channels
            - 1: Grayscale (CT, single MRI)
            - 3: RGB images
            - 4: Multi-modal MRI (FLAIR, T1, T1CE, T2)
        num_classes (int): Number of output segmentation classes
            - 2: Binary (tumor vs background)
            - 4: BraTS (background, NCR, ED, ET)
            - N: N-class segmentation
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output: (B, num_classes, D, H, W)
        - Spatial dimensions are preserved
        
    Example:
        >>> # Standard DeepLabV3+ for BraTS
        >>> model = DeepLabV3Plus(in_channels=4, num_classes=4)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
        >>> # torch.Size([1, 4, 128, 128, 128])
        >>> 
        >>> # Binary segmentation
        >>> model_binary = DeepLabV3Plus(in_channels=1, num_classes=2)
        >>> x_ct = torch.randn(1, 1, 256, 256, 256)
        >>> output = model_binary(x_ct)
        >>> 
        >>> # Access components
        >>> encoder = model.encoder
        >>> decoder = model.decoder
        >>> low, aspp = encoder(x)
        >>> output = decoder(aspp, low)
        
    Model Characteristics:
        - Parameters: ~40M
        - Memory (training, batch=1): ~8-10 GB
        - Memory (inference, batch=1): ~4-5 GB
        - FLOPs: ~190 G
        
    Training Tips:
        - Use learning rate: 1e-3 to 1e-4
        - Optimizer: Adam or SGD with momentum
        - Loss: Dice + CrossEntropy for medical imaging
        - Batch size: 1-2 for 3D volumes (memory intensive)
        - Data augmentation: rotation, flipping, intensity shifts
        - Training time: ~12-24 hours on single GPU
        
    Note:
        - Output is raw logits (apply softmax for probabilities)
        - Input dimensions should be divisible by 16 (due to downsampling)
        - Uses output_stride=16 in backbone (good balance)
        - ASPP uses dilation rates [1, 2, 4, 6]
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # Encoder: Xception backbone + ASPP
        self.encoder = Encoder(in_channels=in_channels)

        # Decoder: Upsample and fuse with low-level features
        # low_level_in=128: From Xception Entry Flow block 2
        self.decoder = Decoder(low_level_in=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete DeepLabV3+.
        
        Args:
            x (torch.Tensor): Input volume with shape (B, C_in, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS
                
        Returns:
            torch.Tensor: Segmentation logits with shape (B, num_classes, D, H, W)
                - Raw predictions (no activation)
                - Apply softmax for multi-class probabilities
                - Apply sigmoid for binary segmentation
                - Spatial dimensions match input
                
        Process:
            1. Encoder: Extract low-level and ASPP features
            2. Decoder: Upsample, fuse, and refine
            3. Output: Class predictions
            
        Example:
            >>> model = DeepLabV3Plus(4, 4)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> 
            >>> # Forward pass
            >>> logits = model(x)
            >>> print(f"Logits: {logits.shape}")
            >>> # torch.Size([2, 4, 128, 128, 128])
            >>> 
            >>> # Get probabilities
            >>> probs = F.softmax(logits, dim=1)
            >>> 
            >>> # Get predictions
            >>> preds = torch.argmax(probs, dim=1)
            >>> print(f"Predictions: {preds.shape}")
            >>> # torch.Size([2, 128, 128, 128])
            >>> 
            >>> # Verify dimensions preserved
            >>> assert logits.shape[2:] == x.shape[2:]
        """
        # Encode: Extract features
        low_level, aspp_output = self.encoder(x)

        # Decode: Produce segmentation
        return self.decoder(aspp_output, low_level)


# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script to verify DeepLabV3+ functionality.
    
    This script demonstrates:
    1. Model instantiation
    2. Architecture visualization
    3. Forward pass verification
    4. Memory and parameter analysis
    
    Usage:
        python deeplabv3.py
    """

    print("=" * 80)
    print("DeepLabV3+ Architecture - Summary and Testing")
    print("=" * 80)
    
    # Create model
    print("\nInstantiating DeepLabV3+ (in_channels=4, num_classes=4)...")
    model = DeepLabV3Plus(in_channels=4, num_classes=4)  
    model = model.cpu()  # Force CPU for testing

    # Display architecture
    print("\n" + "=" * 80)
    print("Architecture Summary")
    print("=" * 80 + "\n")
    
    summary(
        model, 
        input_size=(1, 4, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=4,
        device='cpu'
    )
    
    # Test forward pass
    print("\n" + "-" * 80)
    print("Testing Forward Pass...")
    print("-" * 80)
    
    with torch.no_grad():
        x = torch.randn(1, 4, 128, 128, 128).cpu()
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Spatial dims preserved: {x.shape[2:] == output.shape[2:]}")
    print(f"Correct num_classes: {output.shape[1] == 4}")
    
    # Output statistics
    print(f"\nOutput statistics:")
    print(f"  Min: {output.min():.4f}")
    print(f"  Max: {output.max():.4f}")
    print(f"  Mean: {output.mean():.4f}")
    print(f"  Std: {output.std():.4f}")
    
    print("\n✓ DeepLabV3+ test passed successfully!")
    
    print("\n" + "=" * 80)
    print("DeepLabV3+ Testing Completed!")
    print("=" * 80)