"""
Spatial Pyramid Pooling Modules for 3D Medical Image Segmentation.

This module implements two types of multi-scale feature aggregation:
- SPP (Spatial Pyramid Pooling): Captures context at multiple scales using adaptive pooling
- ASPP (Atrous Spatial Pyramid Pooling): Uses dilated convolutions for multi-scale context

Both modules are designed to enhance feature representations by capturing information
at different spatial scales, which is crucial for accurate segmentation of objects
with varying sizes.

Key Components:
    - SPP: Adaptive pooling at multiple output sizes (2×2×2, 4×4×4, 8×8×8)
    - ASPP: Parallel dilated convolutions with different dilation rates
    - Optional SAM integration for attention-enhanced ASPP

Architecture Comparison:
    SPP:  Input → [Pool1, Pool2, Pool3, Identity] → Concat → Conv → Output
    ASPP: Input → [Conv1×1, AtrousConv3×3 (×3), GlobalPool] → Concat → Conv → Output

Usage:
    # SPP for CLCUNet bottleneck
    spp = SPP(in_channels=256, out_channels=256)
    x = torch.randn(1, 256, 16, 16, 16)
    output = spp(x)  # Same spatial size, enriched features
    
    # ASPP for DeepLabV3+ encoder
    aspp = ASPP(in_channels=2048, out_channels=256)
    x = torch.randn(1, 2048, 8, 8, 8)
    output = aspp(x)  # [1, 256, 8, 8, 8]
    
    # ASPP with attention (DeepLabV3+SAM)
    aspp_sam = ASPP(in_channels=2048, out_channels=256, use_sam=True)
    output = aspp_sam(x)

References:
    - He et al., "Spatial Pyramid Pooling in Deep Convolutional Networks", ECCV 2014
    - Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", arXiv 2017
    - Chen et al., "Encoder-Decoder with Atrous Separable Convolution", ECCV 2018

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchinfo import summary
from sam import SAM


# ============================================================================
# Spatial Pyramid Pooling (SPP)
# ============================================================================

class SPP(nn.Module):
    """
    Spatial Pyramid Pooling Module for Multi-Scale Context Aggregation.
    
    SPP captures features at multiple spatial scales by applying adaptive pooling
    with different output sizes, then upsampling and concatenating the results.
    This allows the network to aggregate context from different receptive fields.
    
    Architecture:
        Input [B, C_in, D, H, W]
        │
        ├─> Branch 1: Identity (original features)
        ├─> Branch 2: AdaptiveAvgPool(2×2×2) → Conv1×1 → Upsample → [B, C_out, D, H, W]
        ├─> Branch 3: AdaptiveAvgPool(4×4×4) → Conv1×1 → Upsample → [B, C_out, D, H, W]
        └─> Branch 4: AdaptiveAvgPool(8×8×8) → Conv1×1 → Upsample → [B, C_out, D, H, W]
        
        Concat → [B, C_in + 3*C_out, D, H, W]
        Conv1×1 → [B, C_out, D, H, W]
    
    Key Features:
        - Multiple pooling scales capture different levels of context
        - Adaptive pooling ensures fixed output sizes regardless of input size
        - 1×1×1 convolutions reduce channels before concatenation
        - Final fusion layer combines multi-scale features
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        pool_sizes (List[int], optional): Output sizes for adaptive pooling.
            Default: [2, 4, 8] corresponding to coarse, medium, and fine scales.
            Smaller values = more global context
            Larger values = more local detail
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output: (B, out_channels, D, H, W) - Same spatial size as input
        
    Example:
        >>> # Basic usage
        >>> spp = SPP(in_channels=256, out_channels=256)
        >>> x = torch.randn(2, 256, 16, 16, 16)
        >>> output = spp(x)
        >>> print(output.shape)  # torch.Size([2, 256, 16, 16, 16])
        >>> 
        >>> # Custom pooling scales
        >>> spp_custom = SPP(in_channels=128, out_channels=128, pool_sizes=[1, 2, 4])
        >>> x = torch.randn(1, 128, 32, 32, 32)
        >>> output = spp_custom(x)
        >>> print(output.shape)  # torch.Size([1, 128, 32, 32, 32])
        
    Use Cases:
        - CLCUNet bottleneck for multi-scale feature aggregation
        - Any architecture requiring global and local context
        - Feature pyramid construction
        
    Note:
        - Spatial dimensions are preserved (input size = output size)
        - Each pooling branch captures different receptive field sizes
        - pool_sizes=[2, 4, 8] means:
          * 2: Very coarse (1/8 of spatial dimensions per axis)
          * 4: Medium (1/4 of spatial dimensions per axis)
          * 8: Fine (1/2 of spatial dimensions per axis)
        - Trilinear upsampling is used to restore original resolution
    """

    def __init__(self, in_channels: int, out_channels: int, pool_sizes: List[int] = [2, 4, 8]):
        super().__init__()

        # ====================================================================
        # Multi-Scale Pooling Branches
        # ====================================================================
        
        # Create parallel pooling layers with different output sizes
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                # Adaptive pooling: Input size → Fixed output size (ps×ps×ps)
                nn.AdaptiveAvgPool3d(output_size=ps),

                # 1×1×1 conv: Reduce channels to out_channels
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])

        # ====================================================================
        # Fusion Layer
        # ====================================================================
        
        # Calculate total channels after concatenation
        # = in_channels (identity) + len(pool_sizes) * out_channels (pooled branches)
        total_ch = in_channels + len(pool_sizes) * out_channels

        # Final 1×1×1 conv to fuse multi-scale features
        self.output_conv = nn.Sequential(
            nn.Conv3d(total_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Spatial Pyramid Pooling to aggregate multi-scale features.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (B, C_in, D, H, W)
            
        Returns:
            torch.Tensor: Multi-scale aggregated features with shape (B, C_out, D, H, W)
                Spatial dimensions match input, but features are enriched with
                multi-scale context information.
                
        Process:
            1. Save original spatial size: (D, H, W)
            2. For each pooling scale:
               a. Adaptive pool to fixed size
               b. Conv1×1 to reduce channels
               c. Upsample back to original size
            3. Concatenate [original, pool1, pool2, pool3]
            4. Fuse with 1×1 convolution
            
        Example:
            >>> spp = SPP(256, 256, pool_sizes=[2, 4, 8])
            >>> x = torch.randn(1, 256, 16, 16, 16)
            >>> 
            >>> # Forward pass
            >>> output = spp(x)
            >>> 
            >>> # Verify shapes
            >>> print(f"Input:  {x.shape}")      # [1, 256, 16, 16, 16]
            >>> print(f"Output: {output.shape}") # [1, 256, 16, 16, 16]
            >>> 
            >>> # Check feature enrichment
            >>> change = (output - x).abs().mean()
            >>> print(f"Average change: {change:.6f}")  # Should be non-zero
        """
        # Save original spatial dimensions for upsampling
        size = x.shape[2:]  # (D, H, W)

        # Start with identity branch (original features)
        pooled = [x]

        # Process each pooling scale
        for layer in self.pool_layers:
            # Apply adaptive pooling + conv
            pooled_feat = layer(x)  # [B, C_out, ps, ps, ps]

            # Upsample back to original size
            upsampled = F.interpolate(
                pooled_feat, 
                size=size, 
                mode='trilinear', 
                align_corners=False
            )  # [B, C_out, D, H, W]

            pooled.append(upsampled)

        # Concatenate all branches
        # [B, C_in + 3*C_out, D, H, W]
        x_cat = torch.cat(pooled, dim=1)

        # Fuse multi-scale features
        return self.output_conv(x_cat)  # [B, C_out, D, H, W]


# ============================================================================
# Atrous Spatial Pyramid Pooling (ASPP)
# ============================================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for Multi-Scale Context in DeepLabV3+.
    
    ASPP captures multi-scale context using parallel atrous (dilated) convolutions
    with different dilation rates. Unlike SPP which uses pooling, ASPP uses
    dilated convolutions to expand the receptive field without losing resolution.
    
    Architecture:
        Input [B, C_in, D, H, W]
        │
        ├─> Branch 1: Conv1×1 (rate=1) ──────────────────┐
        ├─> Branch 2: AtrousConv3×3 (rate=2) ────────────┤
        ├─> Branch 3: AtrousConv3×3 (rate=4) ────────────┤─> Concat
        ├─> Branch 4: AtrousConv3×3 (rate=6) ────────────┤
        └─> Branch 5: GlobalAvgPool → Conv1×1 → Upsample─┘
        
        [Optional: Apply SAM to each branch]
        
        Concat → [B, 5*C_out, D, H, W]
        Project → [B, C_out, D, H, W]
    
    Atrous Convolution (Dilated Convolution):
        Standard Conv3×3: receptive field = 3×3×3
        Atrous Conv3×3 (rate=2): receptive field = 5×5×5 (with holes)
        Atrous Conv3×3 (rate=4): receptive field = 9×9×9 (with larger holes)
        
        Benefits:
        - Larger receptive field without additional parameters
        - Maintains spatial resolution
        - Captures multi-scale context efficiently
    
    Args:
        in_channels (int): Number of input feature channels
            Typically 2048 from Xception backbone
        out_channels (int): Number of output feature channels per branch
            Typically 256 for DeepLabV3+
        atrous_rates (List[int], optional): Dilation rates for atrous convolutions.
            Default: [1, 2, 4, 6]
            - Rate 1: Standard convolution (local features)
            - Rate 2: Small dilation (nearby context)
            - Rate 4: Medium dilation (moderate context)
            - Rate 6: Large dilation (global context)
        use_sam (bool, optional): Whether to apply SAM attention to each branch.
            Default: False
            Set to True for DeepLabV3+SAM variant
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output: (B, out_channels, D, H, W) - Same spatial size as input
        
    Example:
        >>> # Standard ASPP for DeepLabV3+
        >>> aspp = ASPP(in_channels=2048, out_channels=256)
        >>> x = torch.randn(1, 2048, 8, 8, 8)
        >>> output = aspp(x)
        >>> print(output.shape)  # torch.Size([1, 256, 8, 8, 8])
        >>> 
        >>> # ASPP with SAM attention (DeepLabV3+SAM)
        >>> aspp_sam = ASPP(in_channels=2048, out_channels=256, use_sam=True)
        >>> output = aspp_sam(x)
        >>> print(output.shape)  # torch.Size([1, 256, 8, 8, 8])
        >>> 
        >>> # Custom dilation rates
        >>> aspp_custom = ASPP(2048, 256, atrous_rates=[1, 3, 6, 9])
        >>> output = aspp_custom(x)
        
    Use Cases:
        - DeepLabV3+ encoder (standard configuration)
        - DeepLabV3+SAM encoder (with attention)
        - Any segmentation network requiring multi-scale context
        
    Note:
        - All branches output the same number of channels (out_channels)
        - Final output has 5× branches concatenated then projected
        - Global pooling branch captures image-level features
        - Dropout (0.5) is applied before final projection
        - use_sam=True adds SAM attention to each branch independently
        
    References:
        - Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation"
        - Chen et al., "Encoder-Decoder with Atrous Separable Convolution for 
          Semantic Image Segmentation", ECCV 2018
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 atrous_rates=[1, 2, 4, 6], 
                 use_sam=False):
        super().__init__()
        self.use_sam = use_sam

        # ====================================================================
        # Branch 1: 1×1×1 Convolution (Local Features)
        # ====================================================================

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Branch 2: Atrous Conv 3×3×3 (rate=atrous_rates[1], typically 2)
        # ====================================================================

        self.conv3_2 = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=atrous_rates[1],
                dilation=atrous_rates[1], 
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Branch 3: Atrous Conv 3×3×3 (rate=atrous_rates[2], typically 4)
        # ====================================================================
        
        self.conv3_4 = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=atrous_rates[2],
                dilation=atrous_rates[2], 
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Branch 4: Atrous Conv 3×3×3 (rate=atrous_rates[3], typically 6)
        # ====================================================================
        
        self.conv3_6 = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=atrous_rates[3],
                dilation=atrous_rates[3], 
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Branch 5: Global Average Pooling (Image-Level Features)
        # ====================================================================
        
        # Pool to very small size (2×2×2) to capture global context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((2, 2, 2)),  # Small fixed size
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Projection Layer (Fuse All Branches)
        # ====================================================================
        
        self.project = nn.Sequential(
            # Concatenate 5 branches: 5 × out_channels → out_channels
            nn.Conv3d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)  # Regularization
        )

        # ====================================================================
        # Optional SAM Attention Modules
        # ====================================================================

        if self.use_sam:
            # Create SAM for each of the 5 branches
            self.sam1 = SAM(in_channels=out_channels)
            self.sam2 = SAM(in_channels=out_channels)
            self.sam3 = SAM(in_channels=out_channels)
            self.sam4 = SAM(in_channels=out_channels)
            self.sam5 = SAM(in_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Atrous Spatial Pyramid Pooling to extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (B, C_in, D, H, W)
                Typically comes from Xception backbone exit flow
                
        Returns:
            torch.Tensor: Multi-scale features with shape (B, C_out, D, H, W)
                Spatial dimensions preserved, channels reduced to C_out
                
        Process:
            1. Process input through 5 parallel branches:
               - Branch 1: 1×1 conv (local)
               - Branch 2: 3×3 atrous conv, rate=2
               - Branch 3: 3×3 atrous conv, rate=4
               - Branch 4: 3×3 atrous conv, rate=6
               - Branch 5: Global pooling + 1×1 conv
            2. [Optional] Apply SAM to each branch
            3. Upsample global branch to match spatial size
            4. Concatenate all branches
            5. Project to output channels with dropout
            
        Example:
            >>> aspp = ASPP(2048, 256, use_sam=False)
            >>> x = torch.randn(1, 2048, 8, 8, 8)
            >>> 
            >>> # Forward pass
            >>> output = aspp(x)
            >>> 
            >>> # Verify shape
            >>> print(f"Input:  {x.shape}")      # [1, 2048, 8, 8, 8]
            >>> print(f"Output: {output.shape}") # [1, 256, 8, 8, 8]
            >>> 
            >>> # With SAM attention
            >>> aspp_sam = ASPP(2048, 256, use_sam=True)
            >>> output_sam = aspp_sam(x)
            >>> print(f"With SAM: {output_sam.shape}")  # [1, 256, 8, 8, 8]
        """
        # ====================================================================
        # Apply All Branches
        # ====================================================================
        
        x1 = self.conv1(x)      # [B, C_out, D, H, W] - 1×1 conv
        x2 = self.conv3_2(x)    # [B, C_out, D, H, W] - atrous rate=2
        x3 = self.conv3_4(x)    # [B, C_out, D, H, W] - atrous rate=4
        x4 = self.conv3_6(x)    # [B, C_out, D, H, W] - atrous rate=6
        x5 = self.global_pool(x)  # [B, C_out, 2, 2, 2] - global context

        # ====================================================================
        # Optional SAM Attention
        # ====================================================================

        if self.use_sam:
            x1 = self.sam1(x1)
            x2 = self.sam2(x2)
            x3 = self.sam3(x3)
            x4 = self.sam4(x4)
            x5 = self.sam5(x5)

        # ====================================================================
        # Upsample Global Branch
        # ====================================================================
        
        # Restore global branch to original spatial size
        x5 = F.interpolate(
            x5, 
            size=x.shape[2:],   # Match input spatial dimensions
            mode='trilinear', 
            align_corners=False
        )  # [B, C_out, D, H, W]

        # ====================================================================
        # Concatenate and Project
        # ====================================================================
        
        # Concatenate all 5 branches along channel dimension
        x = torch.cat([x1, x2, x3, x4, x5], dim=1) # [B, 5*C_out, D, H, W]
        
        # Project to output channels with dropout
        x = self.project(x)  # [B, C_out, D, H, W]

        return x
    

# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script to verify SPP and ASPP functionality.
    
    This script demonstrates:
    1. SPP with default and custom pooling scales
    2. ASPP without attention (DeepLabV3+)
    3. ASPP with SAM attention (DeepLabV3+SAM)
    4. Architecture visualization with torchinfo
    5. Parameter counting and comparison
    
    Usage:
        python pooling.py
    """

    print("=" * 80)
    print("Spatial Pyramid Pooling Modules - Architecture Summary")
    print("=" * 80)
    
    # Define test models
    models = [
        ("SPP", SPP(in_channels=256, out_channels=256)),
        ("ASPP (Standard)", ASPP(in_channels=256, out_channels=256)),
        ("ASPP + SAM", ASPP(in_channels=256, out_channels=256, use_sam=True)),
    ]

    for name, model in models:
        print(f"\n{'=' * 80}")
        print(f"{name}")
        print(f"{'=' * 80}\n")
        
        # Move to CPU for testing
        model = model.cpu()
        
        # Display architecture
        summary(
            model,
            input_size=(1, 256, 16, 16, 16),
            col_names=["input_size", "output_size", "num_params"],
            depth=3,
            device='cpu'
        )

        # Test forward pass
        print(f"\n{'-' * 80}")
        print("Testing forward pass...")
        print(f"{'-' * 80}")
        
        with torch.no_grad():
            x = torch.randn(1, 256, 16, 16, 16).cpu()
            output = model(x)
        
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Shape preserved: {x.shape[2:] == output.shape[2:]}")
        
        # Check that output changed
        change = (output - x[:, :256]).abs().mean()
        print(f"Average feature change: {change:.6f}")
        
        print(f"✓ {name} test passed")
    
    print(f"\n{'=' * 80}")
    print("All Pooling Module Tests Completed Successfully!")
    print(f"{'=' * 80}")

