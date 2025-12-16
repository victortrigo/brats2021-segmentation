"""
Convolutional Building Blocks for 3D Medical Image Segmentation.

This module provides reusable 3D convolutional blocks that serve as fundamental
components for various segmentation architectures (U-Net, DeepLabV3+, CLCUNet).

Classes:
    DoubleConv: Standard double convolution block with configurable activation
    SeparableConv: Depthwise separable convolution for efficient feature extraction

Usage:
    >>> from convs import DoubleConv, SeparableConv
    >>> block = DoubleConv(in_channels=4, out_channels=64, activation='relu')
    >>> output = block(input_tensor)
"""

import torch
import torch.nn as nn
from torchinfo import summary

class DoubleConv(nn.Module):
    """
    Double 3D convolution block with batch normalization and configurable activation.

    This is a fundamental building block used across multiple architectures (U-Net,
    CLCUNet, etc.). It applies two consecutive 3x3x3 convolutions, each followed by
    batch normalization and an activation function (ReLU or LeakyReLU).

    Architecture:
        Conv3d(3x3x3) → BatchNorm3d → Activation →
        Conv3d(3x3x3) → BatchNorm3d → Activation

    Key Features:
        - Preserves spatial dimensions (padding=1 with kernel_size=3)
        - No bias terms (bias=False) since BatchNorm handles centering
        - Configurable activation function (ReLU or LeakyReLU)
        - In-place operations for memory efficiency
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        activation (str): Activation function type. Options:
            - 'relu': Standard ReLU activation (default)
            - 'leakyrelu': Leaky ReLU with negative slope 0.01
            
    Attributes:
        conv (nn.Sequential): Sequential container with conv layers, batch norms, and activations
        
    Shape:
        - Input: (B, C_in, D, H, W)
        - Output: (B, C_out, D, H, W)
        where B = batch size, C = channels, D/H/W = depth/height/width
        
    Example:
        >>> # Standard ReLU activation
        >>> double_conv = DoubleConv(in_channels=4, out_channels=64, activation='relu')
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> output = double_conv(x)
        >>> print(output.shape)  # torch.Size([1, 64, 128, 128, 128])
        >>> 
        >>> # LeakyReLU for better gradient flow
        >>> leaky_conv = DoubleConv(in_channels=64, out_channels=128, activation='leakyrelu')
        
    Note:
        - ReLU is standard for most architectures
        - LeakyReLU is used in CLCUNet to prevent dying ReLU problem
        - Spatial dimensions are preserved (same padding)
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()

        # Select activation function based on parameter
        if activation.lower() == "leakyrelu":
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)

        # Build sequential block: Conv → BN → Act → Conv → BN → Act
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act_layer,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the double convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, D, H, W)
        """
        return self.conv(x)
    
    
class SeparableConv(nn.Module):
    """
    Depthwise Separable 3D Convolution block.
    
    Separable convolutions decompose a standard convolution into two operations:
    1. Depthwise convolution: Applies a single filter per input channel
    2. Pointwise convolution: 1x1x1 convolution to mix channels
    
    This design significantly reduces computational cost and parameters compared
    to standard convolutions, making it ideal for efficient architectures like
    Xception and MobileNet-inspired models.
    
    Architecture:
        DepthwiseConv3d(KxKxK) → PointwiseConv3d(1x1x1) → BatchNorm3d → ReLU
    
    Computational Savings:
        Standard 3D conv params: K³ × C_in × C_out
        Separable conv params: K³ × C_in + C_in × C_out
        Reduction factor: ~9× for 3x3x3 kernels when C_in ≈ C_out
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the depthwise convolution kernel (default: 3)
        stride (int): Stride of the depthwise convolution (default: 1)
        padding (int): Padding for the depthwise convolution (default: 1)
            For stride=1, padding=1 preserves spatial dimensions with kernel_size=3
            
    Attributes:
        depthwise (nn.Conv3d): Depthwise 3D convolution (groups=in_channels)
        pointwise (nn.Conv3d): Pointwise 1x1x1 convolution for channel mixing
        bn (nn.BatchNorm3d): Batch normalization after convolutions
        relu (nn.ReLU): Activation function applied at the end
        
    Shape:
        - Input: (B, C_in, D, H, W)
        - Output: (B, C_out, D', H', W')
        where D' = (D + 2×padding - kernel_size) / stride + 1
        
    Example:
        >>> # Standard separable convolution (stride=1, preserves dimensions)
        >>> sep_conv = SeparableConv(in_channels=64, out_channels=128)
        >>> x = torch.randn(1, 64, 32, 32, 32)
        >>> output = sep_conv(x)
        >>> print(output.shape)  # torch.Size([1, 128, 32, 32, 32])
        >>> 
        >>> # Separable convolution with stride=2 for downsampling
        >>> downsample = SeparableConv(in_channels=128, out_channels=256, stride=2)
        >>> output = downsample(x)
        >>> print(output.shape)  # torch.Size([1, 256, 16, 16, 16])
        
    Note:
        - Used extensively in Xception-based backbones (DeepLabV3+)
        - More parameter-efficient than standard convolutions
        - Slightly less expressive than standard convolutions but good trade-off
        - Always uses ReLU activation (not configurable like DoubleConv)
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3, 
            stride: int = 1, 
            padding: int = 1
        ):
        super().__init__()

        # Depthwise convolution: each input channel is convolved separately
        self.depthwise = nn.Conv3d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride,
            padding=padding, 
            groups=in_channels,  #Key: groups=in_channels makes it depthwise
            bias=False
        )

        # Pointwise convolution: 1x1x1 conv to mix channels
        self.pointwise = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )

        # Batch normalization and activation
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the separable convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, D', H', W')
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================================
# Module Testing and Architecture Visualization
# ============================================================================

if __name__ == "__main__":
    """
    Test script to visualize architectures and parameter counts.
    
    This script demonstrates the usage of both convolutional blocks and displays
    their architectures, input/output shapes, and parameter counts using torchinfo.
    
    Run with: python convs.py
    """

    print("=" * 70)
    print("3D CONVOLUTIONAL BLOCKS - ARCHITECTURE SUMMARY")
    print("=" * 70)

    # Define test models with different configurations
    models = [
        ("DoubleConv ReLU", 
         DoubleConv(in_channels=4, out_channels=64, activation="relu")),
        ("DoubleConv LeakyReLU",
         DoubleConv(in_channels=4, out_channels=64, activation="leakyrelu")),
        ("SeparableConv", 
         SeparableConv(in_channels=4, out_channels=64)),
    ]

    # Display architecture for each model
    for name, model in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {name}")
        print('=' * 70)

        summary(
            model,
            input_size=(1, 4, 128, 128, 128),
            col_names=["input_size", "output_size", "num_params"],
            depth=4
        )

    # Parameter comparison
    print("=" * 70)
    print("PARAMETER COMPARISON")
    print("=" * 70)
    print(f"{'Block Type':<30} {'Parameters':<15} {'Efficiency'}")
    print("-" * 70)

    # Count parameters for each block
    double_conv_params = sum(p.numel() for p in models[0][1].parameters())
    sep_conv_params = sum(p.numel() for p in models[2][1].parameters())
    
    print(f"{'DoubleConv (4→64 channels)':<30} {double_conv_params:<15,} {'Standard'}")
    print(f"{'SeparableConv (4→64 channels)':<30} {sep_conv_params:<15,} "
          f"{'~' + str(round(double_conv_params/sep_conv_params, 1)) + 'x fewer params'}")
    print("=" * 70)
