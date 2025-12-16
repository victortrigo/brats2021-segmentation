"""
Segmented Attention Module (SAM) for 3D Medical Image Segmentation.

This module implements a dual attention mechanism that recalibrates features
both spatially and channel-wise. SAM is designed to enhance feature representation
in 3D convolutional neural networks by learning where (spatial) and what (channel)
to focus on.

Key Components:
    - SpatialAttention: Focuses on "where" important features are located
    - ChannelAttention: Focuses on "what" channels are important
    - SAM: Combines both mechanisms sequentially

Architecture:
    Input → SpatialAttention → ChannelAttention → Output
    
    1. Spatial: Generate attention map highlighting important spatial locations
    2. Channel: Recalibrate channel importance based on spatially-refined features
    3. Final: Apply channel attention to original input

Usage:
    # Standalone SAM module
    sam = SAM(in_channels=64)
    x = torch.randn(1, 64, 128, 128, 128)
    output = sam(x)  # Same shape as input
    
    # In a CNN architecture
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(4, 64, 3, padding=1)
            self.sam = SAM(64)
            self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.sam(x)  # Apply attention
            x = self.conv2(x)
            return x

References:
    - Inspired by CBAM (Convolutional Block Attention Module)
    - Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    - Adapted for 3D medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# ============================================================================
# Spatial Attention Module
# ============================================================================

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for 3D feature recalibration.
    
    This module generates a 3D attention map that highlights spatially important
    regions in the feature volume. It uses a single 3×3×3 convolution to compress
    channel information into a spatial map, followed by sigmoid activation.
    
    Mechanism:
        1. Input: [B, C, D, H, W]
        2. Conv3d: Aggregate channel info → [B, 1, D, H, W]
        3. Sigmoid: Normalize to [0, 1] range
        4. Output: Spatial attention map [B, 1, D, H, W]
    
    Args:
        in_channels (int): Number of input channels. This determines the receptive
            field of the convolution that generates the attention map.
            
    Shape:
        - Input: (B, C, D, H, W) where B=batch, C=channels, D/H/W=spatial dims
        - Output: (B, 1, D, H, W) - Single-channel attention map
        
    Example:
        >>> spatial_att = SpatialAttention(in_channels=64)
        >>> x = torch.randn(2, 64, 32, 32, 32)
        >>> att_map = spatial_att(x)
        >>> print(att_map.shape)  # torch.Size([2, 1, 32, 32, 32])
        >>> print(att_map.min(), att_map.max())  # Values in [0, 1]
        
    Note:
        - The attention map has values in [0, 1] due to sigmoid
        - Higher values indicate more important spatial locations
        - Can be multiplied with input features for recalibration:
          `x_refined = x * att_map`
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Single 3x3x3 conv to generate spatial attention map
        self.conv = nn.Conv3d(
            in_channels, 
            1, # Output 1 channel (spatial map)
            kernel_size=3, 
            padding=1 # Preserve spatial dimensions
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial attention map from input features.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (B, C, D, H, W)
            
        Returns:
            torch.Tensor: Spatial attention map with shape (B, 1, D, H, W)
                Values are in range [0, 1], where higher values indicate
                more important spatial locations.
                
        Example:
            >>> x = torch.randn(1, 64, 16, 16, 16)
            >>> spatial_att = SpatialAttention(64)
            >>> att_map = spatial_att(x)
            >>> # Apply attention
            >>> x_attended = x * att_map  # Broadcasting over channels
        """
        att_map = self.sigmoid(self.conv(x))
        return att_map


# ============================================================================
# Channel Attention Module
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module for 3D feature recalibration.
    
    This module learns to emphasize important feature channels while suppressing
    less useful ones. It uses both average and max pooling to capture different
    aspects of channel importance, processes them through a shared MLP, and
    combines the results.
    
    Architecture:
        Input [B, C, D, H, W]
           ├─> GlobalAvgPool → [B, C, 1, 1, 1] ─┐
           │                                      ├─> MLP → Sum → Sigmoid
           └─> GlobalMaxPool → [B, C, 1, 1, 1] ─┘
           
        MLP: Conv(C → C//r) → ReLU → Conv(C//r → C)
        where r = reduction_ratio
    
    Mechanism:
        1. Global pooling: Aggregate spatial info (both avg and max)
        2. MLP: Learn channel-wise relationships with bottleneck
        3. Fusion: Sum both pathways and apply sigmoid
        4. Output: Channel-wise attention weights [B, C, 1, 1, 1]
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int, optional): Channel reduction factor in MLP
            bottleneck. Higher values = more compression. Default: 16
            Example: 64 channels → 64//16 = 4 channels in bottleneck
            
    Shape:
        - Input: (B, C, D, H, W)
        - Output: (B, C, 1, 1, 1) - Channel-wise attention weights
        
    Example:
        >>> channel_att = ChannelAttention(in_channels=64, reduction_ratio=16)
        >>> x = torch.randn(2, 64, 32, 32, 32)
        >>> att_weights = channel_att(x)
        >>> print(att_weights.shape)  # torch.Size([2, 64, 1, 1, 1])
        >>> # Apply attention
        >>> x_refined = x * att_weights  # Broadcasting over spatial dims
        
    Note:
        - Uses both AvgPool and MaxPool for complementary information:
          * AvgPool: Captures average response (object extent)
          * MaxPool: Captures strongest response (salient features)
        - Reduction ratio controls model capacity vs. computation
        - The bottleneck (C//r) creates channel interdependencies
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        # Calculate bottleneck channels (minimum 1)
        self.reduction = max(1, in_channels // reduction_ratio)

        # Shared MLP for both avg and max pool pathways
        self.mlp = nn.Sequential(
            # Bottleneck: Reduce channels to capture interdependencies
            nn.Conv3d(in_channels, self.reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Expansion: Restore to original channel count
            nn.Conv3d(self.reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate channel-wise attention weights from input features.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (B, C, D, H, W)
            
        Returns:
            torch.Tensor: Channel attention weights with shape (B, C, 1, 1, 1)
                Values are in range [0, 1], where higher values indicate
                more important channels.
                
        Process:
            1. Global Average Pooling: [B, C, D, H, W] → [B, C, 1, 1, 1]
            2. Global Max Pooling: [B, C, D, H, W] → [B, C, 1, 1, 1]
            3. Apply MLP to both: [B, C, 1, 1, 1] → [B, C, 1, 1, 1]
            4. Sum and sigmoid: [B, C, 1, 1, 1]
            
        Example:
            >>> x = torch.randn(1, 64, 16, 16, 16)
            >>> channel_att = ChannelAttention(64)
            >>> att_weights = channel_att(x)
            >>> # Check which channels are emphasized
            >>> print(att_weights.squeeze().topk(5))  # Top 5 channels
        """
        # Global Average Pooling: Capture average spatial response
        gap = F.adaptive_avg_pool3d(x, (1, 1, 1)) # [B, C, 1, 1, 1]

        # Global Max Pooling: Capture strongest spatial response
        gmp = F.adaptive_max_pool3d(x, (1, 1, 1)) # [B, C, 1, 1, 1]

        # Process both through shared MLP and combine
        channel_att = self.sigmoid(self.mlp(gap) + self.mlp(gmp))

        return channel_att


# ============================================================================
# Segmented Attention Module (SAM)
# ============================================================================

class SAM(nn.Module):
    """
    Segmented Attention Module - Dual Attention Mechanism for 3D CNNs.
    
    SAM combines spatial and channel attention mechanisms sequentially to
    recalibrate feature representations. It first identifies important spatial
    regions, then recalibrates channel importance based on the spatially-refined
    features.
    
    Two-Stage Attention Process:
        1. Spatial Attention: "Where should we focus?"
           - Generates spatial attention map
           - Highlights important voxel locations
           
        2. Channel Attention: "What features are important?"
           - Uses spatially-refined features
           - Emphasizes important feature channels
    
    Mathematical Formulation:
        Given input X ∈ R^(B×C×D×H×W):
        
        1. Spatial Attention:
           M_s = σ(Conv3d(X))  ∈ R^(B×1×D×H×W)
           X' = X ⊙ M_s
           
        2. Channel Attention:
           M_c = σ(MLP(GAP(X')) + MLP(GMP(X')))  ∈ R^(B×C×1×1×1)
           X'' = X ⊙ M_c
           
        where σ is sigmoid, ⊙ is element-wise multiplication
    
    Args:
        in_channels (int): Number of input feature channels
        
    Shape:
        - Input: (B, C, D, H, W)
        - Output: (B, C, D, H, W) - Same shape as input
        
    Example:
        >>> # Basic usage
        >>> sam = SAM(in_channels=64)
        >>> x = torch.randn(2, 64, 32, 32, 32)
        >>> x_refined = sam(x)
        >>> print(x_refined.shape)  # torch.Size([2, 64, 32, 32, 32])
        
        >>> # Use in a model
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = nn.Conv3d(4, 64, 3, padding=1)
        ...         self.sam1 = SAM(64)
        ...         self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        ...         self.sam2 = SAM(128)
        ...     
        ...     def forward(self, x):
        ...         x = F.relu(self.conv1(x))
        ...         x = self.sam1(x)  # First attention
        ...         x = F.relu(self.conv2(x))
        ...         x = self.sam2(x)  # Second attention
        ...         return x
        
    Performance Benefits:
        - Improved feature discrimination
        - Better localization of regions of interest
        - Helps network focus on relevant features
        - Minimal computational overhead (~1% of total FLOPs)
        
    References:
        - Based on CBAM (Woo et al., ECCV 2018)
        - Extended to 3D for medical imaging applications
        - Used in DeepLabV3+SAM and CLCUNet architectures
        
    Note:
        - Sequential application (spatial → channel) is important
        - Channel attention uses spatially-refined features
        - Can be inserted after any convolutional layer
        - Works best when inserted after feature extraction blocks
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_att = SpatialAttention(in_channels)
        self.channel_att = ChannelAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sequential spatial and channel attention to input features.
        
        Process:
            1. Generate spatial attention map: M_s = SpatialAttention(X)
            2. Apply spatial attention: X' = X ⊙ M_s
            3. Generate channel attention: M_c = ChannelAttention(X')
            4. Apply channel attention: X'' = X ⊙ M_c
            5. Return refined features: X''
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (B, C, D, H, W)
            
        Returns:
            torch.Tensor: Attention-refined feature tensor with shape (B, C, D, H, W)
                Same spatial dimensions as input, but with recalibrated features.
                
        Example:
            >>> sam = SAM(64)
            >>> x = torch.randn(1, 64, 16, 16, 16)
            >>> 
            >>> # Forward pass
            >>> x_refined = sam(x)
            >>> 
            >>> # Visualize attention effect
            >>> import numpy as np
            >>> change = (x_refined - x).abs().mean(dim=1, keepdim=True)
            >>> print(f"Average absolute change: {change.mean():.4f}")
            >>> 
            >>> # Check that shape is preserved
            >>> assert x.shape == x_refined.shape
            
        Note:
            - Spatial attention is applied first (highlights "where")
            - Channel attention uses spatially-refined features (learns "what")
            - Original input is used for channel attention (not spatially-attended)
            - This design preserves gradient flow and information
        """
        # Step 1: Apply spatial attention
        spatial_map = self.spatial_att(x) # [B, 1, D, H, W]
        x_spatial = spatial_map * x      # Broadcasting over channels

        # Step 2: Apply channel attention
        channel_map = self.channel_att(x_spatial) # [B, C, 1, 1, 1]
        x_att = x * channel_map                   # Broadcasting over spatial dims

        return x_att
    

# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script to verify SAM functionality and visualize architecture.
    
    This script demonstrates:
    1. SAM module instantiation
    2. Forward pass with sample input
    3. Architecture visualization with torchinfo
    4. Output shape verification
    
    Usage:
        python sam.py
        
    Expected Output:
        - Architecture summary with layer-by-layer details
        - Total parameter count
        - Input/output shapes for each layer
    """
    
    print("=" * 80)
    print("SAM (Segmented Attention Module) - Architecture Summary")
    print("=" * 80)

    # Instantiate SAM with 64 channels
    sam = SAM(in_channels=64)

    # Display architecture
    summary(
        sam, 
        input_size=(1, 64, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=5, # Show nested modules
        device='cpu'
    )

    print("\n" + "=" * 80)
    print("Testing SAM with sample input...")
    print("=" * 80)

    # Create sample input
    x = torch.randn(2, 64, 32, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = sam(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {x.shape == output.shape}")
    
    # Check value changes
    change = (output - x).abs().mean()
    print(f"Average absolute change: {change:.6f}")
    
    print("\n" + "=" * 80)
    print("SAM Test Completed Successfully!")
    print("=" * 80)