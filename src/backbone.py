"""
Xception Backbone for 3D Medical Image Segmentation.

This module implements a 3D adaptation of the Xception architecture, originally
designed for 2D image classification. Xception (Extreme Inception) replaces
standard convolutions with depthwise separable convolutions, reducing parameters
while maintaining or improving performance.

Architecture Overview:
    Input → EntryFlow → MiddleFlow → ExitFlow → Output
    
    - EntryFlow: Initial feature extraction with spatial downsampling
    - MiddleFlow: 16 residual blocks for deep feature learning
    - ExitFlow: High-level feature extraction for decoder
    
Key Features:
    - Depthwise separable convolutions (efficient computation)
    - Residual connections (gradient flow, feature reuse)
    - Configurable output stride (16 or 32)
    - Multi-scale features (low-level + high-level outputs)
    
Output Stride:
    - 16: Less downsampling, preserves more spatial detail (recommended)
    - 32: More downsampling, faster but lower resolution
    
Usage:
    # For DeepLabV3+ encoder
    backbone = BackboneXception(in_channels=4, output_stride=16)
    low_level, high_level = backbone(x)
    
    # Low-level: For skip connections (128 channels, 1/4 resolution)
    # High-level: For ASPP module (2048 channels, 1/16 or 1/32 resolution)
    
Typical Channel Progression:
    Input (4 channels)
    → Entry: 4 → 32 → 64 → 128 (low-level) → 256 → 728
    → Middle: 728 (maintained for 16 blocks)
    → Exit: 728 → 1024 → 1536 → 2048 (high-level)
    
References:
    - Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017
    - Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018
"""

import torch
import torch.nn as nn
from typing import Tuple
from torchinfo import summary

from convs import SeparableConv


# ============================================================================
# Entry Flow - Initial Feature Extraction
# ============================================================================

class EntryFlow(nn.Module):
    """
    Entry Flow of Xception 3D Backbone.
    
    This module performs initial feature extraction with progressive downsampling
    and channel expansion. It consists of:
    - 2 initial standard convolutions (quick spatial reduction)
    - 3 residual blocks with depthwise separable convolutions
    - Skip connection output at 1/4 resolution (low-level features)
    
    Architecture:
        Input [B, in_ch, D, H, W]
        ├─> Conv1: in_ch → 32 (stride=2, /2 resolution)
        ├─> Conv2: 32 → 64
        │
        ├─> Block2: 64 → 128 (stride=2, /4 resolution) ─┐
        │                                                 ├─> low_level output
        ├─> Block3: 128 → 256 (stride=2, /8 resolution)  │
        │                                                 │
        └─> Block4: 256 → 728 ───────────────────────────┘
        
    Spatial Reduction:
        - After Conv1: 1/2 resolution
        - After Block2: 1/4 resolution (low-level features extracted here)
        - After Block3: 1/8 resolution
        - After Block4: 1/8 resolution (no stride)
        
    Args:
        in_channels (int): Number of input channels (e.g., 4 for multi-modal MRI)
        
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output low_level: (B, 128, D/4, H/4, W/4)
        - Output x: (B, 728, D/8, H/8, W/8)
        
    Example:
        >>> entry = EntryFlow(in_channels=4)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> low_level, x = entry(x)
        >>> print(low_level.shape)  # torch.Size([1, 128, 32, 32, 32])
        >>> print(x.shape)          # torch.Size([1, 728, 16, 16, 16])
        
    Note:
        - Block2 output is captured as low_level for decoder skip connections
        - All blocks use residual connections (identity shortcut)
        - Residual shortcuts use 1×1×1 conv to match dimensions
    """
    def __init__(self, in_channels: int):
        super().__init__()

        # ====================================================================
        # Initial Convolution Blocks (Standard Conv, not Separable)
        # ====================================================================
        
        # Conv1: Initial spatial reduction (stride=2 → /2 resolution)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # Conv2: Feature refinement (no downsampling)
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # ====================================================================
        # Block 2: 64 → 128 channels (stride=2 → /4 resolution)
        # ====================================================================
        
        # Main branch: 3× SeparableConv
        self.block2 = nn.Sequential(
            SeparableConv(64, 128, stride=1),
            SeparableConv(128, 128, stride=1),
            SeparableConv(128, 128, stride=2) # Spatial downsampling here
        )

        # Residual branch: 1×1 conv to match dimensions
        self.residual2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(128)
        )

        # ====================================================================
        # Block 3: 128 → 256 channels (stride=2 → /8 resolution)
        # ====================================================================

        self.block3 = nn.Sequential(
            SeparableConv(128, 256, stride=1),
            SeparableConv(256, 256, stride=1),
            SeparableConv(256, 256, stride=2) # Spatial downsampling here
        )

        self.residual3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(256)
        )

        # ====================================================================
        # Block 4: 256 → 728 channels (no downsampling)
        # ====================================================================
        
        self.block4 = nn.Sequential(
            SeparableConv(256, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1) # No stride=1 (maintain resolution)
        )

        self.residual4 = nn.Sequential(
            nn.Conv3d(256, 728, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(728)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Entry Flow.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS dataset
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - low_level: Features at 1/4 resolution for skip connections
                  Shape: (B, 128, D/4, H/4, W/4)
                - x: Deep features at 1/8 resolution for middle flow
                  Shape: (B, 728, D/8, H/8, W/8)
                  
        Process:
            1. Conv1 (stride=2): [B, 4, 128, 128, 128] → [B, 32, 64, 64, 64]
            2. Conv2: [B, 32, 64, 64, 64] → [B, 64, 64, 64, 64]
            3. Block2+Residual (stride=2): [B, 64, 64, 64, 64] → [B, 128, 32, 32, 32]
               └─> Capture as low_level
            4. Block3+Residual (stride=2): [B, 128, 32, 32, 32] → [B, 256, 16, 16, 16]
            5. Block4+Residual: [B, 256, 16, 16, 16] → [B, 728, 16, 16, 16]
            
        Example:
            >>> entry = EntryFlow(in_channels=4)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> low, high = entry(x)
            >>> print(f"Low-level: {low.shape}")   # [2, 128, 32, 32, 32]
            >>> print(f"High-level: {high.shape}") # [2, 728, 16, 16, 16]
        """
        # Initial convolutions
        x = self.conv1(x)  # /2 resolution: [B, 32, D/2, H/2, W/2]
        x = self.conv2(x)  # Same resolution: [B, 64, D/2, H/2, W/2]

        # Block 2 with residual (outputs low-level features)
        residual = self.residual2(x)
        x = self.block2(x) + residual
        x = self.relu(x)
        low_level = x  # Capture for skip connection: [B, 128, D/4, H/4, W/4]

        # Block 3 with residual
        residual = self.residual3(x)
        x = self.block3(x) + residual
        x = self.relu(x)  # [B, 256, D/8, H/8, W/8]

        # Block 4 with residual (no spatial reduction)
        residual = self.residual4(x)
        x = self.block4(x) + residual
        x = self.relu(x)  # [B, 728, D/8, H/8, W/8]

        return low_level, x


# ============================================================================
# Middle Flow - Deep Feature Learning
# ============================================================================

class MiddleFlow(nn.Module):
    """
    Middle Flow of Xception 3D Backbone.
    
    This module consists of repeated residual blocks that maintain spatial
    resolution and channel count (728 channels). The repetition allows the
    network to learn increasingly abstract representations without changing
    feature map dimensions.
    
    Architecture:
        Input [B, 728, D, H, W]
        ├─> Block 1: 728 → 728 (residual)
        ├─> Block 2: 728 → 728 (residual)
        ├─> ...
        └─> Block N: 728 → 728 (residual)
        
    Each Block Structure:
        Input
        ├─> SeparableConv (3×3×3, 728 → 728)
        ├─> SeparableConv (3×3×3, 728 → 728)
        ├─> SeparableConv (3×3×3, 728 → 728)
        └─> Add with Input (residual)
        
    Args:
        num_blocks (int, optional): Number of residual blocks. Default: 16
            - Original Xception uses 16 blocks
            - Can be reduced for faster training (e.g., 8)
            - Can be increased for more capacity (e.g., 24)
            
    Shape:
        - Input: (B, 728, D, H, W)
        - Output: (B, 728, D, H, W) - Same shape as input
        
    Example:
        >>> middle = MiddleFlow(num_blocks=16)
        >>> x = torch.randn(1, 728, 16, 16, 16)
        >>> output = middle(x)
        >>> print(output.shape)  # torch.Size([1, 728, 16, 16, 16])
        >>> print(x.shape == output.shape)  # True
        
    Note:
        - All blocks maintain spatial resolution (no downsampling)
        - All blocks maintain channel count (728 channels)
        - Residual connections help gradient flow
        - Depthwise separable convs reduce parameters
    """

    def __init__(self, num_blocks: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([self._make_block() for _ in range(num_blocks)])

    def _make_block(self):
        """
        Create a single residual block with 3× SeparableConv layers.
        
        Block Structure:
            Input [B, 728, D, H, W]
            ├─> SeparableConv (stride=1)
            ├─> SeparableConv (stride=1)
            └─> SeparableConv (stride=1)
            Output [B, 728, D, H, W]
            
        Returns:
            nn.Sequential: Residual block module
            
        Note:
            - All convolutions use stride=1 (no downsampling)
            - Input and output dimensions are identical
            - Allows for clean residual addition
        """
        block = nn.Sequential(
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1)
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Middle Flow.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, 728, D, H, W)
                Typically comes from EntryFlow output
                
        Returns:
            torch.Tensor: Output tensor with shape (B, 728, D, H, W)
                Same shape as input, but with refined features
                
        Process:
            For each block:
                1. Save input as residual
                2. Pass through 3× SeparableConv
                3. Add residual to output
                4. Pass to next block
                
        Example:
            >>> middle = MiddleFlow(num_blocks=16)
            >>> x = torch.randn(1, 728, 16, 16, 16)
            >>> output = middle(x)
            >>> 
            >>> # Features are refined but shape is preserved
            >>> assert output.shape == x.shape
            >>> 
            >>> # Check that features changed
            >>> change = (output - x).abs().mean()
            >>> print(f"Average feature change: {change:.6f}")
        """
        for block in self.blocks:
            residual = x
            x = block(x) + residual # Residual addition
        return x


# ============================================================================
# Exit Flow - High-Level Feature Extraction
# ============================================================================

class ExitFlow(nn.Module):
    """
    Exit Flow of Xception 3D Backbone.
    
    This module extracts high-level features for the decoder or classification
    head. It progressively increases channel count while optionally reducing
    spatial resolution (depending on output_stride).
    
    Architecture:
        Input [B, 728, D, H, W]
        ├─> Block: 728 → 1024 (stride=2 if output_stride=32)
        ├─> SepConv1: 1024 → 1536
        ├─> SepConv2: 1536 → 1536 (stride=2 if output_stride=32)
        └─> SepConv3: 1536 → 2048
        Output [B, 2048, D/s, H/s, W/s] where s depends on output_stride
        
    Output Stride Behavior:
        - output_stride=16: Only initial block downsamples (stride=2)
          Final resolution: Input/2
        - output_stride=32: Both block and sepconv2 downsample (stride=2 each)
          Final resolution: Input/4
          
    Args:
        output_stride (int, optional): Controls spatial downsampling. Default: 16
            - 16: Less downsampling, better spatial detail (recommended)
            - 32: More downsampling, faster computation
            
    Shape:
        - Input: (B, 728, D, H, W)
        - Output (stride=16): (B, 2048, D/2, H/2, W/2)
        - Output (stride=32): (B, 2048, D/4, H/4, W/4)
        
    Example:
        >>> # With output_stride=16 (default)
        >>> exit16 = ExitFlow(output_stride=16)
        >>> x = torch.randn(1, 728, 16, 16, 16)
        >>> output = exit16(x)
        >>> print(output.shape)  # torch.Size([1, 2048, 8, 8, 8])
        >>> 
        >>> # With output_stride=32
        >>> exit32 = ExitFlow(output_stride=32)
        >>> output = exit32(x)
        >>> print(output.shape)  # torch.Size([1, 2048, 4, 4, 4])
        
    Note:
        - Higher output_stride = more downsampling = less detail
        - output_stride=16 is recommended for segmentation tasks
        - Final output has 2048 channels (rich semantic features)
    """

    def __init__(self, output_stride: int = 16):
        super().__init__()
        
        # Determine stride for sepconv2 based on output_stride
        sepconv2_stride = 2 if output_stride == 32 else 1

        # ====================================================================
        # Initial Block: 728 → 1024 channels (always stride=2)
        # ====================================================================
        
        self.block = nn.Sequential(
            SeparableConv(728, 1024, stride=1),
            SeparableConv(1024, 1024, stride=1),
            SeparableConv(1024, 1024, stride=2)  # Always downsample here
        )

        # Residual branch to match dimensions
        self.residual = nn.Sequential(
            nn.Conv3d(728, 1024, kernel_size=1, stride=2, bias=False), 
            nn.BatchNorm3d(1024)
        )

        # ====================================================================
        # Progressive Channel Expansion
        # ====================================================================
        
        # SepConv1: 1024 → 1536 (no downsampling)
        self.sepconv1 = SeparableConv(1024, 1536, stride=1)

        # SepConv2: 1536 → 1536 (conditional downsampling)
        self.sepconv2 = SeparableConv(1536, 1536, stride=sepconv2_stride) 

        # SepConv3: 1536 → 2048 (no downsampling)
        self.sepconv3 = SeparableConv(1536, 2048, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Exit Flow.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, 728, D, H, W)
                Typically comes from MiddleFlow output
                
        Returns:
            torch.Tensor: High-level features with shape (B, 2048, D/s, H/s, W/s)
                where s depends on output_stride configuration
                
        Process:
            1. Block (stride=2): [B, 728, D, H, W] → [B, 1024, D/2, H/2, W/2]
            2. SepConv1: [B, 1024, D/2, H/2, W/2] → [B, 1536, D/2, H/2, W/2]
            3. SepConv2: [B, 1536, ...] → [B, 1536, ...] (stride depends on output_stride)
            4. SepConv3: [B, 1536, ...] → [B, 2048, ...]
            
        Example:
            >>> exit_flow = ExitFlow(output_stride=16)
            >>> x = torch.randn(1, 728, 16, 16, 16)
            >>> output = exit_flow(x)
            >>> 
            >>> # Check output
            >>> print(f"Output shape: {output.shape}")
            >>> print(f"Channel count: {output.shape[1]}")  # Should be 2048
        """
        # Initial block with residual
        residual = self.residual(x)
        x = self.block(x) + residual
        x = self.relu(x)  # [B, 1024, D/2, H/2, W/2]

        # Progressive channel expansion
        x = self.sepconv1(x)  # [B, 1536, D/2, H/2, W/2]
        x = self.sepconv2(x)  # [B, 1536, ...] (depends on output_stride)
        x = self.sepconv3(x)  # [B, 2048, ...]

        return x


# ============================================================================
# Complete Backbone
# ============================================================================

class BackboneXception(nn.Module):
    """
    Complete Xception 3D Backbone for Medical Image Segmentation.
    
    This module implements a full Xception backbone adapted for 3D volumetric
    data. It combines EntryFlow, MiddleFlow, and ExitFlow to extract multi-scale
    features suitable for segmentation tasks.
    
    Architecture Flow:
        Input [B, in_ch, D, H, W]
        │
        ├─> EntryFlow
        │   ├─> low_level [B, 128, D/4, H/4, W/4]  (for skip connections)
        │   └─> x [B, 728, D/8, H/8, W/8]
        │
        ├─> MiddleFlow (16 blocks)
        │   └─> x [B, 728, D/8, H/8, W/8]  (refined features)
        │
        └─> ExitFlow
            └─> high_level [B, 2048, D/s, H/s, W/s]  (for decoder)
            
    Output Stride:
        Determines final spatial resolution:
        - 16: Preserves more detail (recommended for segmentation)
        - 32: More aggressive downsampling (faster, less detail)
        
    Use Cases:
        - DeepLabV3+ encoder
        - U-Net backbone replacement
        - Feature extraction for segmentation
        
    Args:
        in_channels (int, optional): Number of input channels. Default: 4
            Example: 4 for multi-modal MRI (FLAIR, T1, T1CE, T2)
        output_stride (int, optional): Spatial downsampling factor. Default: 16
            Must be either 16 or 32
        num_middle_blocks (int, optional): Number of middle flow blocks. Default: 16
            Can be adjusted for speed/accuracy tradeoff
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output low_level: (B, 128, D/4, H/4, W/4)
        - Output high_level (stride=16): (B, 2048, D/16, H/16, W/16)
        - Output high_level (stride=32): (B, 2048, D/32, H/32, W/32)
        
    Example:
        >>> # Standard usage for DeepLabV3+
        >>> backbone = BackboneXception(in_channels=4, output_stride=16)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> low_level, high_level = backbone(x)
        >>> 
        >>> print(f"Low-level: {low_level.shape}")   # [1, 128, 32, 32, 32]
        >>> print(f"High-level: {high_level.shape}") # [1, 2048, 8, 8, 8]
        >>> 
        >>> # Use in encoder
        >>> class MyEncoder(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.backbone = BackboneXception(4, 16)
        ...         self.aspp = ASPP(2048, 256)
        ...     
        ...     def forward(self, x):
        ...         low, high = self.backbone(x)
        ...         high = self.aspp(high)
        ...         return low, high
        
    Raises:
        ValueError: If output_stride is not 16 or 32
        
    Note:
        - Low-level features are for decoder skip connections
        - High-level features go to ASPP or classification head
        - Uses depthwise separable convolutions for efficiency
        - All blocks use residual connections
        
    References:
        - Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017
        - Chen et al., "Encoder-Decoder with Atrous Separable Convolution", ECCV 2018
    """

    def __init__(self, 
                 in_channels: int = 4, 
                 output_stride: int = 16,
                 num_middle_blocks: int = 16):
        super().__init__()

        # ====================================================================
        # Validate Output Stride
        # ====================================================================

        ALLOWED_STRIDES = {16, 32}
        if output_stride not in ALLOWED_STRIDES:
            raise ValueError(
                f"Output stride '{output_stride}' is not supported for Xception 3D backbone. "
                f"Only the following are allowed: {ALLOWED_STRIDES}."
            )
        
        # ====================================================================
        # Build Backbone Components
        # ====================================================================

        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow(num_blocks=num_middle_blocks)
        self.exit = ExitFlow(output_stride=output_stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete Xception backbone.
        
        Args:
            x (torch.Tensor): Input volume with shape (B, C, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS dataset
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - low_level: Features for skip connections
                  Shape: (B, 128, D/4, H/4, W/4)
                  Use: Decoder skip connections in DeepLabV3+
                  
                - high_level: Deep semantic features
                  Shape: (B, 2048, D/s, H/s, W/s) where s = output_stride
                  Use: Input to ASPP or decoder
                  
        Spatial Resolution Progression:
            Input:      [B, 4, 128, 128, 128]
            Entry out:  [B, 728, 16, 16, 16]  (1/8 resolution)
            Middle out: [B, 728, 16, 16, 16]  (maintained)
            Exit out:   [B, 2048, 8, 8, 8]    (1/16 if stride=16)
            
        Example:
            >>> backbone = BackboneXception(in_channels=4, output_stride=16)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> 
            >>> # Forward pass
            >>> low, high = backbone(x)
            >>> 
            >>> # Verify shapes
            >>> print(f"Input: {x.shape}")
            >>> print(f"Low-level: {low.shape}")
            >>> print(f"High-level: {high.shape}")
            >>> 
            >>> # Typical output:
            >>> # Input: torch.Size([2, 4, 128, 128, 128])
            >>> # Low-level: torch.Size([2, 128, 32, 32, 32])
            >>> # High-level: torch.Size([2, 2048, 8, 8, 8])
        """
        # Entry Flow: Extract low and intermediate features
        low_level, x = self.entry(x)

        # Middle Flow: Refine features (16 residual blocks)
        x = self.middle(x)

        # Exit Flow: Extract high-level semantic features
        high_level = self.exit(x)

        return low_level, high_level


# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script to verify Xception backbone functionality.
    
    This script demonstrates:
    1. Backbone instantiation with different output strides
    2. Architecture visualization with torchinfo
    3. Output shape verification
    4. Parameter counting
    
    Usage:
        python backbone.py
    """
    # Detectar device disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("=" * 80)
    print("Xception 3D Backbone - Architecture Summary")
    print("=" * 80)

    # Test both output strides
    for os in [16, 32]:
        print(f"\n{'=' * 80}")
        print(f"Backbone with output_stride={os}")
        print(f"{'=' * 80}\n")

        # Instantiate model
        model = BackboneXception(in_channels=4, output_stride=os)
        model = model.to(device)

        # Display architecture
        summary(
            model,
            input_size=(1, 4, 128, 128, 128),
            col_names=["input_size", "output_size", "num_params"],
            depth=2,
            device=str(device)
        )

        # Test forward pass
        print(f"\n{'-' * 80}")
        print("Testing forward pass...")
        print(f"{'-' * 80}")
        
        with torch.no_grad():
            x = torch.randn(1, 4, 128, 128, 128).to(device)
            low, high = model(x)
            
        print(f"Input shape:      {x.shape}")
        print(f"Low-level shape:  {low.shape}")
        print(f"High-level shape: {high.shape}")
        
        # Verify expected shapes
        expected_low = (1, 128, 32, 32, 32)
        if os == 16:
            expected_high = (1, 2048, 8, 8, 8)
        else:
            expected_high = (1, 2048, 4, 4, 4)
            
        assert low.shape == expected_low, f"Low-level shape mismatch!"
        assert high.shape == expected_high, f"High-level shape mismatch!"
        
        print(f"✓ All shapes verified for output_stride={os}")
    
    print(f"\n{'=' * 80}")
    print("Backbone Test Completed Successfully!")
    print(f"{'=' * 80}")
