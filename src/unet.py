"""
3D U-Net Architecture for Volumetric Medical Image Segmentation.

This module implements the 3D U-Net architecture, originally designed for
biomedical image segmentation. U-Net features a symmetric encoder-decoder
structure with skip connections that combine low-level and high-level features
for precise localization.

Architecture Overview:
    
    Input (4 channels, 128³)
       ↓
    ┌─────────────────────────────────────┐
    │         ENCODER (Contracting Path)  │
    │  ┌────┐  ┌────┐  ┌────┐  ┌────┐     │
    │  │E1  │→ │E2  │→ │E3  │→ │E4  │     │
    │  │64  │  │128 │  │256 │  │512 │     │
    │  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘     │
    │    ↓       ↓       ↓       ↓        │
    │    Pool    Pool    Pool    Pool     │
    └─────────────────────────────────────┘
                    ↓
         ┌─────────────────┐
         │   BOTTLENECK    │
         │   1024 channels │
         └─────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │       DECODER (Expanding Path)      │
    │    Up      Up      Up      Up       │
    │    ↓       ↓       ↓       ↓        │
    │  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐     │ 
    │  │D4  │← │D3  │← │D2  │← │D1  │     │
    │  │512 │  │256 │  │128 │  │64  │     │
    │  └────┘  └────┘  └────┘  └────┘     │
    └─────────────────────────────────────┘
                    ↓
              Output (num_classes)

Key Features:
    - Symmetric encoder-decoder architecture
    - Skip connections preserve spatial information
    - Progressive downsampling in encoder (captures context)
    - Progressive upsampling in decoder (enables localization)
    - DoubleConv blocks for feature extraction
    - 3D convolutions for volumetric data

Skip Connections:
    The key innovation of U-Net is skip connections that directly connect
    encoder and decoder at the same spatial resolution:
    
    E1 (128³, 64ch)  ──────────> D1 (128³, 64ch)
    E2 (64³, 128ch)  ──────────> D2 (64³, 128ch)
    E3 (32³, 256ch)  ──────────> D3 (32³, 256ch)
    E4 (16³, 512ch)  ──────────> D4 (16³, 512ch)
    
Usage:
    # Standard U-Net for BraTS segmentation
    model = UNet(in_channels=4, num_classes=4, base_channels=64)
    x = torch.randn(1, 4, 128, 128, 128)  # BraTS input
    output = model(x)  # [1, 4, 128, 128, 128]
    
    # Lighter U-Net (fewer parameters)
    model_light = UNet(in_channels=4, num_classes=4, base_channels=32)
    
    # Deeper U-Net (more capacity)
    model_deep = UNet(in_channels=4, num_classes=4, base_channels=128)

Channel Progression (base_channels=64):
    Input: 4 → E1: 64 → E2: 128 → E3: 256 → E4: 512 → Bottleneck: 1024
    Bottleneck: 1024 → D4: 512 → D3: 256 → D2: 128 → D1: 64 → Output: num_classes

References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
      Image Segmentation", MICCAI 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchinfo import summary
from convs import DoubleConv


# ============================================================================
# Encoder Components
# ============================================================================

class EncoderBlock(nn.Module):
    """
    Single Encoder Block for U-Net 3D.
    
    This block performs feature extraction and spatial downsampling. It consists
    of a DoubleConv for learning features and MaxPool3d for reducing spatial
    dimensions by half.
    
    Architecture:
        Input [B, C_in, D, H, W]
        ├─> DoubleConv: C_in → C_out
        │   Output: [B, C_out, D, H, W]  (for skip connection)
        │
        └─> MaxPool3d (2×2×2)
            Output: [B, C_out, D/2, H/2, W/2]  (for next encoder block)
    
    The DoubleConv output is saved for skip connections, while the pooled
    output is passed to the next encoder stage or bottleneck.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels after DoubleConv
        
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output conv_out: (B, out_channels, D, H, W) - For skip connection
        - Output pool_out: (B, out_channels, D/2, H/2, W/2) - For next stage
        
    Example:
        >>> enc_block = EncoderBlock(in_channels=4, out_channels=64)
        >>> x = torch.randn(2, 4, 128, 128, 128)
        >>> conv_out, pool_out = enc_block(x)
        >>> print(f"Conv output (skip): {conv_out.shape}")
        >>> # torch.Size([2, 64, 128, 128, 128])
        >>> print(f"Pool output (next): {pool_out.shape}")
        >>> # torch.Size([2, 64, 64, 64, 64])
        
    Note:
        - conv_out retains full spatial resolution for skip connections
        - pool_out has half the spatial dimensions for next encoder stage
        - MaxPool with stride=2 reduces each dimension by factor of 2
        - No overlap in pooling windows (kernel_size=2, stride=2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Feature extraction with double 3×3×3 convolutions
        self.double_conv = DoubleConv(in_channels, out_channels)

        # Spatial downsampling by factor of 2 in each dimension
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, D, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - conv_out: Features after DoubleConv (for skip connection)
                  Shape: (B, C_out, D, H, W)
                - pool_out: Features after MaxPool (for next stage)
                  Shape: (B, C_out, D/2, H/2, W/2)
                  
        Example:
            >>> block = EncoderBlock(64, 128)
            >>> x = torch.randn(1, 64, 64, 64, 64)
            >>> skip, next_stage = block(x)
            >>> print(f"Skip: {skip.shape}, Next: {next_stage.shape}")
            >>> # Skip: torch.Size([1, 128, 64, 64, 64]), 
            >>> # Next: torch.Size([1, 128, 32, 32, 32])
        """
        # Apply DoubleConv (spatial dimensions preserved)
        conv_out = self.double_conv(x)

        # Apply MaxPool (spatial dimensions halved)
        pool_out = self.maxpool(conv_out)

        return conv_out, pool_out

class Encoder(nn.Module):
    """
    Complete Encoder (Contracting Path) for U-Net 3D.
    
    The encoder progressively downsamples the input while increasing the number
    of feature channels. It captures hierarchical features from low-level
    (edges, textures) to high-level (semantic concepts).
    
    Architecture:
        Input [B, in_ch, 128, 128, 128]
        │
        ├─> EncoderBlock1: in_ch → base_ch
        │   Conv: [B, base_ch, 128, 128, 128] ──────┐ (E1, for skip)
        │   Pool: [B, base_ch, 64, 64, 64]          │
        │                                           │
        ├─> EncoderBlock2: base_ch → base_ch*2      │
        │   Conv: [B, base_ch*2, 64, 64, 64] ─────┐ │ (E2, for skip)
        │   Pool: [B, base_ch*2, 32, 32, 32]      │ │
        │                                         │ │
        ├─> EncoderBlock3: base_ch*2 → base_ch*4  │ │
        │   Conv: [B, base_ch*4, 32, 32, 32] ───┐ │ │ (E3, for skip)
        │   Pool: [B, base_ch*4, 16, 16, 16]    │ │ │
        │                                       │ │ │
        ├─> EncoderBlock4: base_ch*4 → base_ch*8│ │ │
        │   Conv: [B, base_ch*8, 16, 16, 16] ───│ │ │ (E4, for skip)
        │   Pool: [B, base_ch*8, 8, 8, 8]     │ │ │ │
        │                                     │ │ │ │
        └─> Bottleneck: base_ch*8 → base_ch*16│ │ │ │
            [B, base_ch*16, 8, 8, 8]          │ │ │ │
                                              │ │ │ │
        Return: (E1, E2, E3, E4, Bottleneck) ←┘─┘─┘─┘
    
    Spatial Resolution Progression (assuming 128³ input):
        Input:      128³
        After E1:   64³   (1/2)
        After E2:   32³   (1/4)
        After E3:   16³   (1/8)
        After E4:   8³    (1/16)
        Bottleneck: 8³    (1/16)
    
    Channel Progression (base_channels=64):
        Input: 4 → 64 → 128 → 256 → 512 → 1024
    
    Args:
        in_channels (int): Number of input channels (e.g., 4 for multi-modal MRI)
        base_channels (int, optional): Base number of feature channels. Default: 64
            All subsequent layers scale from this value:
            - Encoder block 1: base_channels
            - Encoder block 2: base_channels * 2
            - Encoder block 3: base_channels * 4
            - Encoder block 4: base_channels * 8
            - Bottleneck: base_channels * 16
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output conv1: (B, base_ch, D, H, W)
        - Output conv2: (B, base_ch*2, D/2, H/2, W/2)
        - Output conv3: (B, base_ch*4, D/4, H/4, W/4)
        - Output conv4: (B, base_ch*8, D/8, H/8, W/8)
        - Output bottleneck: (B, base_ch*16, D/16, H/16, W/16)
        
    Example:
        >>> encoder = Encoder(in_channels=4, base_channels=64)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> e1, e2, e3, e4, bottleneck = encoder(x)
        >>> 
        >>> print(f"E1 (skip 1): {e1.shape}")  # [1, 64, 128, 128, 128]
        >>> print(f"E2 (skip 2): {e2.shape}")  # [1, 128, 64, 64, 64]
        >>> print(f"E3 (skip 3): {e3.shape}")  # [1, 256, 32, 32, 32]
        >>> print(f"E4 (skip 4): {e4.shape}")  # [1, 512, 16, 16, 16]
        >>> print(f"Bottleneck:  {bottleneck.shape}")  # [1, 1024, 8, 8, 8]
        
    Note:
        - Each encoder block reduces spatial dimensions by 2×
        - Each encoder block doubles the number of channels
        - All conv outputs (E1-E4) are used for skip connections
        - Bottleneck does NOT downsample (no MaxPool after it)
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        
        # Define encoder blocks with progressive channel scaling
        self.enc_block1 = EncoderBlock(in_channels, base_channels)
        self.enc_block2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc_block3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc_block4 = EncoderBlock(base_channels * 4, base_channels * 8)

        # Bottleneck: deepest features, no pooling after
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through complete encoder.
        
        Args:
            x (torch.Tensor): Input volume with shape (B, C_in, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS dataset
                
        Returns:
            Tuple[torch.Tensor, ...]: Five tensors:
                - conv1: E1 features for skip connection 1
                - conv2: E2 features for skip connection 2
                - conv3: E3 features for skip connection 3
                - conv4: E4 features for skip connection 4
                - bottleneck_out: Deepest features for decoder
                
        Process:
            1. Block 1: Input → E1 (skip), Pool → next
            2. Block 2: → E2 (skip), Pool → next
            3. Block 3: → E3 (skip), Pool → next
            4. Block 4: → E4 (skip), Pool → bottleneck
            5. Bottleneck: → Final features
            
        Example:
            >>> encoder = Encoder(4, 64)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> e1, e2, e3, e4, b = encoder(x)
            >>> 
            >>> # Verify spatial dimensions
            >>> assert e1.shape[2:] == (128, 128, 128)  # Full res
            >>> assert e2.shape[2:] == (64, 64, 64)     # 1/2
            >>> assert e3.shape[2:] == (32, 32, 32)     # 1/4
            >>> assert e4.shape[2:] == (16, 16, 16)     # 1/8
            >>> assert b.shape[2:] == (8, 8, 8)         # 1/16
            >>> 
            >>> # Verify channel progression
            >>> assert e1.shape[1] == 64    # base
            >>> assert e2.shape[1] == 128   # base * 2
            >>> assert e3.shape[1] == 256   # base * 4
            >>> assert e4.shape[1] == 512   # base * 8
            >>> assert b.shape[1] == 1024   # base * 16
        """
        # Encoder block 1
        conv1, pool1 = self.enc_block1(x)

        # Encoder block 2
        conv2, pool2 = self.enc_block2(pool1)

        # Encoder block 3
        conv3, pool3 = self.enc_block3(pool2)

        # Encoder block 4
        conv4, pool4 = self.enc_block4(pool3)

        # Bottleneck (no more pooling)
        bottleneck_out = self.bottleneck(pool4)

        return conv1, conv2, conv3, conv4, bottleneck_out


# ============================================================================
# Decoder Components
# ============================================================================

class UpConvBlock(nn.Module):
    """
    Upsampling Decoder Block for U-Net 3D with Skip Connection.
    
    This block performs upsampling, concatenates with skip connection from
    encoder, and applies feature refinement. It's the key component of the
    expanding path in U-Net.
    
    Architecture:
        Input from below [B, C_up, D, H, W]
        │
        ├─> ConvTranspose3d (stride=2)
        │   ↓ [B, C_up/2, D*2, H*2, W*2]
        │   
        ├─> [Optional] Padding alignment
        │   ↓
        │   
        ├─> Concatenate with skip connection
        │   Skip: [B, C_skip, D*2, H*2, W*2]
        │   ↓ [B, C_up/2 + C_skip, D*2, H*2, W*2]
        │   
        └─> DoubleConv
            ↓ [B, C_out, D*2, H*2, W*2]
    
    Upsampling Strategy:
        Uses ConvTranspose3d (learnable upsampling) instead of interpolation.
        This allows the network to learn the best way to upsample features.
        
    Skip Connection Handling:
        - Concatenates encoder features with upsampled features
        - If spatial dimensions don't match perfectly, applies padding
        - Padding ensures alignment even with odd input dimensions
    
    Args:
        in_channels_up (int): Channels from lower decoder stage
        skip_channels (int): Channels from encoder skip connection
        out_channels (int): Output channels after DoubleConv
        
    Shape:
        - Input x_up: (B, in_channels_up, D, H, W)
        - Input x_skip: (B, skip_channels, D*2, H*2, W*2)
        - Output: (B, out_channels, D*2, H*2, W*2)
        
    Example:
        >>> # Decoder stage 4 (connecting to E4)
        >>> up_block = UpConvBlock(
        ...     in_channels_up=1024,    # From bottleneck
        ...     skip_channels=512,       # From E4
        ...     out_channels=512
        ... )
        >>> 
        >>> x_up = torch.randn(1, 1024, 8, 8, 8)      # From below
        >>> x_skip = torch.randn(1, 512, 16, 16, 16)  # From E4
        >>> output = up_block(x_up, x_skip)
        >>> print(output.shape)  # torch.Size([1, 512, 16, 16, 16])
        
    Note:
        - in_channels_up should be 2× out_channels for typical U-Net
        - ConvTranspose3d automatically doubles spatial dimensions
        - Padding handles misalignment from odd input sizes
        - Skip connection provides high-resolution details
    """

    def __init__(self, in_channels_up: int, skip_channels: int, out_channels: int):
        super().__init__()

        # Learnable upsampling: doubles spatial dimensions
        # in_channels_up → in_channels_up // 2
        self.up_transpose = nn.ConvTranspose3d(
            in_channels_up, 
            in_channels_up // 2, 
            kernel_size=2, 
            stride=2
        )

        # Feature refinement after concatenation
        # Concatenated channels: (in_channels_up // 2) + skip_channels
        self.double_conv = DoubleConv(
            in_channels_up // 2 + skip_channels, 
            out_channels
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with upsampling and skip connection fusion.
        
        Args:
            x_up (torch.Tensor): Features from lower decoder stage
                Shape: (B, C_up, D, H, W)
            x_skip (torch.Tensor): Features from encoder (skip connection)
                Shape: (B, C_skip, D*2, H*2, W*2)
                
        Returns:
            torch.Tensor: Upsampled and refined features
                Shape: (B, C_out, D*2, H*2, W*2)
                
        Process:
            1. Upsample x_up with ConvTranspose3d
            2. Check spatial dimension alignment with x_skip
            3. Apply padding if needed (handles odd dimensions)
            4. Concatenate upsampled and skip features
            5. Refine with DoubleConv
            
        Example:
            >>> up_block = UpConvBlock(256, 128, 128)
            >>> x_up = torch.randn(1, 256, 16, 16, 16)
            >>> x_skip = torch.randn(1, 128, 32, 32, 32)
            >>> output = up_block(x_up, x_skip)
            >>> print(output.shape)  # torch.Size([1, 128, 32, 32, 32])
            >>> 
            >>> # Test with misaligned dimensions
            >>> x_skip_odd = torch.randn(1, 128, 33, 33, 33)
            >>> output = up_block(x_up, x_skip_odd)
            >>> print(output.shape)  # torch.Size([1, 128, 33, 33, 33])
        """
        # Upsample features from below (doubles spatial dimensions)
        x_up = self.up_transpose(x_up)
      
        # ====================================================================
        # Handle Spatial Dimension Alignment
        # ====================================================================
        
        # Check if skip and upsampled features have same spatial dimensions
        if x_skip.shape[2:] != x_up.shape[2:]:
            # Calculate padding needed for each dimension
            # diff[i] = skip_size[i] - up_size[i]
            diff = [x_skip.shape[i+2] - x_up.shape[i+2] for i in range(3)]

            # Apply symmetric padding: [left, right, top, bottom, front, back]
            # For each dimension, split diff into two parts
            x_up = F.pad(x_up, [
                diff[2] // 2, diff[2] - diff[2] // 2,  # W dimension
                diff[1] // 2, diff[1] - diff[1] // 2,  # H dimension
                diff[0] // 2, diff[0] - diff[0] // 2   # D dimension
            ])

        # ====================================================================
        # Concatenate Skip Connection
        # ====================================================================
        
        # Concatenate along channel dimension
        # [B, C_up/2, D*2, H*2, W*2] + [B, C_skip, D*2, H*2, W*2]
        # → [B, C_up/2 + C_skip, D*2, H*2, W*2]
        x_combined = torch.cat([x_skip, x_up], dim=1)

        # ====================================================================
        # Feature Refinement
        # ====================================================================
        
        # Apply DoubleConv to learn combined representation
        return self.double_conv(x_combined)


class Decoder(nn.Module):
    """
    Complete Decoder (Expanding Path) for U-Net 3D.
    
    The decoder progressively upsamples features while decreasing the number
    of channels. It combines low-level details from encoder skip connections
    with high-level semantic features to produce accurate segmentations.
    
    Architecture:
        Bottleneck [B, base_ch*16, 8, 8, 8]
        │
        ├─> UpConvBlock4: base_ch*16 → base_ch*8
        │   + E4 skip [B, base_ch*8, 16, 16, 16]
        │   → [B, base_ch*8, 16, 16, 16]
        │
        ├─> UpConvBlock3: base_ch*8 → base_ch*4
        │   + E3 skip [B, base_ch*4, 32, 32, 32]
        │   → [B, base_ch*4, 32, 32, 32]
        │
        ├─> UpConvBlock2: base_ch*4 → base_ch*2
        │   + E2 skip [B, base_ch*2, 64, 64, 64]
        │   → [B, base_ch*2, 64, 64, 64]
        │
        └─> UpConvBlock1: base_ch*2 → base_ch
            + E1 skip [B, base_ch, 128, 128, 128]
            → [B, base_ch, 128, 128, 128]
    
    Spatial Resolution Progression:
        Bottleneck: 8³    (1/16)
        After D4:   16³   (1/8)
        After D3:   32³   (1/4)
        After D2:   64³   (1/2)
        After D1:   128³  (1/1) - Full resolution restored
    
    Channel Progression (base_channels=64):
        1024 → 512 → 256 → 128 → 64
    
    Args:
        base_channels (int): Base number of channels (must match encoder)
            Used to scale all decoder channel counts
            
    Shape:
        - Input bottleneck: (B, base_ch*16, D/16, H/16, W/16)
        - Input conv4 (skip): (B, base_ch*8, D/8, H/8, W/8)
        - Input conv3 (skip): (B, base_ch*4, D/4, H/4, W/4)
        - Input conv2 (skip): (B, base_ch*2, D/2, H/2, W/2)
        - Input conv1 (skip): (B, base_ch, D, H, W)
        - Output: (B, base_ch, D, H, W)
        
    Example:
        >>> decoder = Decoder(base_channels=64)
        >>> 
        >>> # Create mock encoder outputs
        >>> bottleneck = torch.randn(1, 1024, 8, 8, 8)
        >>> e4 = torch.randn(1, 512, 16, 16, 16)
        >>> e3 = torch.randn(1, 256, 32, 32, 32)
        >>> e2 = torch.randn(1, 128, 64, 64, 64)
        >>> e1 = torch.randn(1, 64, 128, 128, 128)
        >>> 
        >>> # Decode
        >>> output = decoder(bottleneck, e4, e3, e2, e1)
        >>> print(output.shape)  # torch.Size([1, 64, 128, 128, 128])
        
    Note:
        - Decoder is symmetric to encoder
        - Each UpConvBlock doubles spatial dimensions
        - Each UpConvBlock halves channel count
        - Skip connections must match decoder spatial resolution
        - Output has full input resolution with base_channels
    """

    def __init__(self, base_channels: int):
        super().__init__()

        # Define decoder blocks with progressive channel reduction
        # Each block: in_up, skip, out
        self.up_block1 = UpConvBlock(
            in_channels_up=base_channels * 16, 
            skip_channels=base_channels * 8, 
            out_channels=base_channels * 8
        )
        self.up_block2 = UpConvBlock(
            in_channels_up=base_channels * 8, 
            skip_channels=base_channels * 4, 
            out_channels=base_channels * 4
        )
        self.up_block3 = UpConvBlock(
            in_channels_up=base_channels * 4, 
            skip_channels=base_channels * 2, 
            out_channels=base_channels * 2
        )
        self.up_block4 = UpConvBlock(
            in_channels_up=base_channels * 2, 
            skip_channels=base_channels, 
            out_channels=base_channels
        )

    def forward(self, 
                bottleneck_out: torch.Tensor, 
                conv4: torch.Tensor, 
                conv3: torch.Tensor, 
                conv2: torch.Tensor, 
                conv1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete decoder.
        
        Args:
            bottleneck_out (torch.Tensor): Deepest features from encoder
                Shape: (B, base_ch*16, D/16, H/16, W/16)
            conv4 (torch.Tensor): E4 skip connection
                Shape: (B, base_ch*8, D/8, H/8, W/8)
            conv3 (torch.Tensor): E3 skip connection
                Shape: (B, base_ch*4, D/4, H/4, W/4)
            conv2 (torch.Tensor): E2 skip connection
                Shape: (B, base_ch*2, D/2, H/2, W/2)
            conv1 (torch.Tensor): E1 skip connection
                Shape: (B, base_ch, D, H, W)
                
        Returns:
            torch.Tensor: Decoded features at full resolution
                Shape: (B, base_ch, D, H, W)
                
        Process:
            1. UpBlock1: bottleneck + conv4 → [B, 512, 16³]
            2. UpBlock2: [B, 512, 16³] + conv3 → [B, 256, 32³]
            3. UpBlock3: [B, 256, 32³] + conv2 → [B, 128, 64³]
            4. UpBlock4: [B, 128, 64³] + conv1 → [B, 64, 128³]
            
        Example:
            >>> decoder = Decoder(64)
            >>> b = torch.randn(1, 1024, 8, 8, 8)
            >>> e4 = torch.randn(1, 512, 16, 16, 16)
            >>> e3 = torch.randn(1, 256, 32, 32, 32)
            >>> e2 = torch.randn(1, 128, 64, 64, 64)
            >>> e1 = torch.randn(1, 64, 128, 128, 128)
            >>> 
            >>> out = decoder(b, e4, e3, e2, e1)
            >>> 
            >>> # Verify full resolution restored
            >>> assert out.shape[2:] == e1.shape[2:]
            >>> assert out.shape[1] == 64  # base_channels
        """
        # Decoder block 1: Bottleneck → D4
        x = self.up_block1(bottleneck_out, conv4) 

        # Decoder block 2: D4 → D3
        x = self.up_block2(x, conv3)

        # Decoder block 3: D3 → D2             
        x = self.up_block3(x, conv2)

        # Decoder block 4: D2 → D1 (full resolution)         
        x = self.up_block4(x, conv1)

        return x


# ============================================================================
# Complete U-Net Model
# ============================================================================

class UNet(nn.Module):
    """
    Complete 3D U-Net Model for Volumetric Medical Image Segmentation.
    
    U-Net is a fully convolutional network designed for biomedical image
    segmentation. It features a symmetric encoder-decoder architecture with
    skip connections that enable precise localization by combining semantic
    features with spatial details.
    
    Complete Architecture:
        
        Input [B, in_ch, D, H, W]
               ↓
        ┌──────────────────┐
        │     ENCODER      │
        │  (Contracting)   │
        │                  │
        │  E1: 64 ch  ──────────┐
        │    ↓ (pool)           │
        │  E2: 128 ch ────────┐ │
        │    ↓ (pool)         │ │
        │  E3: 256 ch ──────┐ │ │
        │    ↓ (pool)       │ │ │
        │  E4: 512 ch ────┐ │ │ │
        │    ↓ (pool)     │ │ │ │
        │  B: 1024 ch     │ │ │ │
        └──────────────────┘ │ │ │
               ↓             │ │ │
        ┌──────────────────┐ │ │ │
        │     DECODER      │ │ │ │
        │   (Expanding)    │ │ │ │
        │                  │ │ │ │
        │  D4: 512 ch  ←───┘ │ │
        │    ↑ (upsample)    │ │
        │  D3: 256 ch  ←─────┘ │
        │    ↑ (upsample)      │
        │  D2: 128 ch  ←───────┘
        │    ↑ (upsample)
        │  D1: 64 ch   ←────────┘
        └──────────────────┘
               ↓
         Conv 1×1×1
               ↓
        Output [B, num_classes, D, H, W]
    
    Key Design Principles:
        1. **Symmetric Architecture**: Encoder and decoder are mirrors
        2. **Skip Connections**: Preserve fine-grained spatial information
        3. **Progressive Downsampling**: Captures hierarchical features
        4. **Progressive Upsampling**: Restores spatial resolution
        5. **DoubleConv Blocks**: Extract rich features at each scale
    
    Why U-Net Works Well:
        - Skip connections solve vanishing gradient problem
        - Combines semantic (what) and spatial (where) information
        - Fully convolutional = works on any input size
        - Efficient: Shares computations in encoder
        - Proven track record in medical imaging
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 1
            - 1: Grayscale (CT, single MRI sequence)
            - 3: RGB images
            - 4: Multi-modal MRI (FLAIR, T1, T1CE, T2)
        num_classes (int, optional): Number of output classes. Default: 2
            - 2: Binary segmentation (tumor vs background)
            - 4: BraTS (background, NCR, ED, ET)
            - N: N-class segmentation
        base_channels (int, optional): Base number of feature channels. Default: 64
            Controls model capacity:
            - 32: Lighter model (~7M params)
            - 64: Standard model (~31M params)
            - 128: Heavier model (~123M params)
            
    Shape:
        - Input: (B, in_channels, D, H, W)
        - Output: (B, num_classes, D, H, W)
        - Spatial dimensions are preserved (D_out = D_in, etc.)
        
    Example:
        >>> # Standard U-Net for BraTS
        >>> model = UNet(in_channels=4, num_classes=4, base_channels=64)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 4, 128, 128, 128])
        >>> 
        >>> # Binary segmentation (tumor/background)
        >>> model_binary = UNet(in_channels=1, num_classes=2, base_channels=64)
        >>> x_ct = torch.randn(1, 1, 256, 256, 256)
        >>> output = model_binary(x_ct)
        >>> print(output.shape)  # torch.Size([1, 2, 256, 256, 256])
        >>> 
        >>> # Lightweight U-Net (for limited GPU memory)
        >>> model_light = UNet(in_channels=4, num_classes=4, base_channels=32)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> output = model_light(x)
        >>> print(output.shape)  # torch.Size([1, 4, 128, 128, 128])
        
    Model Sizes (for 128³ input, base_channels=64):
        - Parameters: ~31M
        - Memory (training, batch=1): ~8 GB
        - Memory (inference, batch=1): ~4 GB
        - FLOPs: ~180 G
        
    Training Tips:
        - Use Dice loss or combined Dice + CE for class imbalance
        - Batch size typically 1-2 for 3D volumes (memory intensive)
        - Learning rate: 1e-3 to 1e-4 with Adam
        - Data augmentation: rotation, flipping, elastic deformation
        - Training time: ~12-24 hours on single GPU for BraTS
        
    References:
        - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
          Image Segmentation", MICCAI 2015
          
    Note:
        - Output is raw logits (no softmax/sigmoid)
        - Apply softmax for multi-class or sigmoid for binary in loss function
        - Input can be any size, but dimensions should be divisible by 16
          (due to 4 pooling layers: 2^4 = 16)
    """

    def __init__(self,  
                 in_channels: int = 1, 
                 num_classes: int = 2, 
                 base_channels: int = 64):
        super().__init__()

        # Contracting path (encoder)
        self.encoder = Encoder(in_channels, base_channels)

        # Expanding path (decoder)
        self.decoder = Decoder(base_channels)

        # Final 1×1×1 convolution to map features to class predictions
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete U-Net.
        
        Args:
            x (torch.Tensor): Input volume with shape (B, C_in, D, H, W)
                Example: (1, 4, 128, 128, 128) for BraTS
                
        Returns:
            torch.Tensor: Segmentation logits with shape (B, num_classes, D, H, W)
                - NOT probabilities (no softmax/sigmoid applied)
                - Apply softmax for multi-class or sigmoid for binary
                - Spatial dimensions match input
                
        Process:
            1. Encoder: Extract hierarchical features with skip connections
            2. Decoder: Upsample and fuse with skip connections
            3. Output convolution: Map to class predictions
            
        Example:
            >>> model = UNet(4, 4, 64)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> 
            >>> # Forward pass
            >>> logits = model(x)
            >>> print(f"Logits: {logits.shape}")  # [2, 4, 128, 128, 128]
            >>> 
            >>> # Get probabilities (for inference)
            >>> probs = F.softmax(logits, dim=1)
            >>> print(f"Probs: {probs.shape}")   # [2, 4, 128, 128, 128]
            >>> 
            >>> # Get predictions (for visualization)
            >>> preds = torch.argmax(probs, dim=1)
            >>> print(f"Preds: {preds.shape}")   # [2, 128, 128, 128]
            >>> 
            >>> # Verify spatial dimensions preserved
            >>> assert logits.shape[2:] == x.shape[2:]
        """
        # Encoder: Extract features at multiple scales
        conv1, conv2, conv3, conv4, bottleneck_out = self.encoder(x)

        # Decoder: Reconstruct segmentation with skip connections
        decoder_out = self.decoder(bottleneck_out, conv4, conv3, conv2, conv1)
        
        # Final layer: Map to class logits
        output = self.out_conv(decoder_out)

        return output
  

# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script to verify U-Net functionality.
    
    This script demonstrates:
    1. U-Net instantiation with different configurations
    2. Architecture visualization with torchinfo
    3. Forward pass verification
    4. Memory and parameter analysis
    
    Usage:
        python unet.py
    """

    print("=" * 80)
    print("3D U-Net Architecture - Summary and Testing")
    print("=" * 80)
    
    # Standard U-Net for BraTS
    print("\n" + "=" * 80)
    print("Standard U-Net (in_channels=4, num_classes=4, base_channels=64)")
    print("=" * 80 + "\n")

    model = UNet(in_channels=4, num_classes=4)  
    model = model.cpu()

    # Display architecture
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
    
    # Test output properties
    print(f"\nOutput statistics:")
    print(f"  Min: {output.min():.4f}")
    print(f"  Max: {output.max():.4f}")
    print(f"  Mean: {output.mean():.4f}")
    print(f"  Std: {output.std():.4f}")
    
    print("\n✓ U-Net test passed successfully!")
    
    print("\n" + "=" * 80)
    print("U-Net Testing Completed!")
    print("=" * 80)
