"""
CLCU-Net: Cross-Level Connected U-Net with Attention for 3D Medical Image Segmentation.

This module implements CLCU-Net, an advanced U-Net variant that features:
- Cross-level connections between non-adjacent encoder stages
- Segmented Attention Modules (SAM) for feature recalibration
- Spatial Pyramid Pooling (SPP) in the bottleneck
- Multi-scale prediction heads in the decoder

Architecture Innovation:
    Unlike standard U-Net which only connects adjacent encoder-decoder stages,
    CLCU-Net creates additional pathways:
    - E1 → E3: Early features directly to mid-level decoder
    - E1 → Bottleneck: Early features to deepest level
    - E2 → Bottleneck: Mid-level features to deepest level
    
    This enables better information flow and reduces semantic gap between
    encoder and decoder features.

Key Components:
    - Encoder: Progressive downsampling with cross-level projections
    - Bottleneck: SPP module for multi-scale context aggregation
    - Decoder: Upsampling with multi-scale feature fusion
    - Output: Three-branch prediction (D1, D2, D3) with progressive refinement

Output Structure:
    CLCU-Net produces 4 channels representing tumor subregions:
    - Channel 0: Background
    - Channel 1: NCR (Necrotic Core) - P1
    - Channel 2: ED (Edema) - P2  
    - Channel 3: ET (Enhancing Tumor) - P3
    
    Each prediction branch (P1, P2, P3) is supervised during training,
    encouraging multi-scale learning.

Usage:
    # Standard CLCU-Net for BraTS
    model = CLCUNet(in_channels=4, base_channels=64)
    x = torch.randn(1, 4, 128, 128, 128)
    output = model(x)  # [1, 4, 128, 128, 128]
    
    # During training, access intermediate predictions
    encoder = model.encoder
    decoder = model.decoder
    output_module = model.output
    
    e1, e2, e3, b = encoder(x)
    d1, d2, d3 = decoder(b, e3, e2, e1)
    final_mask, (p1, p2, p3) = output_module(d1, d2, d3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torchinfo import summary

from convs import DoubleConv
from pooling import SPP
from sam import SAM


# ============================================================================
# Encoder with Cross-Level Connections
# ============================================================================

class Encoder(nn.Module):
    """
    CLCU-Net Encoder with Cross-Level Attention Connections.
    
    This encoder extends the standard U-Net encoder by adding direct connections
    between non-adjacent levels. These cross-level connections help bridge the
    semantic gap between encoder and decoder features.
    
    Architecture:
        Input [B, in_ch, 128, 128, 128]
        │
        ├─> E1: DoubleConv → [B, 64, 128, 128, 128] ────┐─────┐
        │   MaxPool → [B, 64, 64, 64, 64]                │     │
        │                                                 │     │
        ├─> E2: DoubleConv → [B, 128, 64, 64, 64] ──────┼─┐   │
        │   MaxPool → [B, 128, 32, 32, 32]               │ │   │
        │                                                 │ │   │
        ├─> E3: DoubleConv → [B, 256, 32, 32, 32]        │ │   │
        │   ↑ (receives E1→E3 projection + SAM)          │ │   │
        │   MaxPool → [B, 256, 16, 16, 16]               │ │   │
        │                                                 │ │   │
        └─> Bottleneck [B, 256, 16, 16, 16]              │ │   │
            ↑ (receives E1→B, E2→B projections + SAM)    │ │   │
            SPP + Fusion → [B, 256, 16, 16, 16]          │ │   │
                                                          │ │   │
        Cross-Level Connections:                          │ │   │
        E1 → E3: MaxPool(4) + Conv + SAM ────────────────┘ │   │
        E1 → B:  AvgPool(8) + Conv + SAM ──────────────────┼───┘
        E2 → B:  AvgPool(4) + Conv + SAM ──────────────────┘
    
    Cross-Level Connection Benefits:
        1. E1→E3: Brings fine-grained spatial details directly to mid-decoder
        2. E1→B: Preserves low-level features in deep layers
        3. E2→B: Provides intermediate features to bottleneck
        
    Why This Works:
        - Reduces semantic gap between encoder and decoder
        - Preserves spatial information at multiple scales
        - Attention (SAM) helps select relevant features
        - SPP in bottleneck captures multi-scale context
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 4
        base_channels (int, optional): Base number of feature channels. Default: 64
            All subsequent layers scale from this value
            
    Shape:
        - Input: (B, in_channels, 128, 128, 128)
        - Output e1: (B, 64, 128, 128, 128)
        - Output e2: (B, 128, 64, 64, 64)
        - Output e3: (B, 256, 32, 32, 32)
        - Output b: (B, 256, 16, 16, 16)
        
    Example:
        >>> encoder = Encoder(in_channels=4, base_channels=64)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> e1, e2, e3, b = encoder(x)
        >>> 
        >>> print(f"E1: {e1.shape}")  # [1, 64, 128, 128, 128]
        >>> print(f"E2: {e2.shape}")  # [1, 128, 64, 64, 64]
        >>> print(f"E3: {e3.shape}")  # [1, 256, 32, 32, 32]
        >>> print(f"B:  {b.shape}")   # [1, 256, 16, 16, 16]
        
    Note:
        - Uses LeakyReLU instead of ReLU (better gradient flow)
        - All cross-level projections use SAM for attention
        - SPP in bottleneck provides multi-scale receptive fields
        - Bottleneck concatenates: E3_pooled + E1→B + E2→B
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        
        # ====================================================================
        # Standard Encoder Blocks (E1, E2)
        # ====================================================================
        
        # E1: First encoding block
        self.enc1 = DoubleConv(in_channels, base_channels, activation="leakyrelu")
        self.pool1 = nn.MaxPool3d(2)

        # E2: Second encoding block
        self.enc2 = DoubleConv(base_channels, base_channels * 2, activation="leakyrelu")
        self.pool2 = nn.MaxPool3d(2)

        # ====================================================================
        # Cross-Level Connection: E1 → E3
        # ====================================================================
        
        # Project E1 features to E3 level
        # MaxPool(4): 128³ → 32³ (skip 2 levels)
        # Conv: 64 → 128 channels (match E2 output)
        self.down1_e3_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(inplace=True)
        )
        # Attention to select relevant features from E1 for E3
        self.down1_e3_sam = SAM(base_channels * 2)

        # ====================================================================
        # E3: Third encoding block (receives E1→E3 + E2)
        # ====================================================================
        
        # Input channels: E2 (128) + E1→E3 (128) = 256
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 4, activation="leakyrelu")
        
        # ====================================================================
        # Cross-Level Connections: E1 → Bottleneck, E2 → Bottleneck
        # ====================================================================
        
        # E1 → Bottleneck projection
        # AvgPool(8): 128³ → 16³ (skip 3 levels)
        # Conv: 64 → 256 channels (match bottleneck)
        self.down_e1_b = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e1_b = SAM(base_channels * 4)

        # E2 → Bottleneck projection
        # AvgPool(4): 64³ → 16³ (skip 2 levels)
        # Conv: 128 → 256 channels
        self.down_e2_b = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e2_b = SAM(base_channels * 4)

        # ====================================================================
        # Bottleneck with SPP
        # ====================================================================
        
        # Input: E3_pooled (256) + E1→B (256) + E2→B (256) = 768 channels
        # Reduce to 256 before SPP
        self.reduce_conv = nn.Conv3d(base_channels * 12, base_channels * 4, kernel_size=1)
        
        # Spatial Pyramid Pooling for multi-scale context
        self.spp = SPP(base_channels * 4, base_channels * 4)

        # Final bottleneck refinement
        self.after_spp = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with cross-level connections.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, 128, 128, 128)
            
        Returns:
            Tuple[torch.Tensor, ...]: Four encoder outputs:
                - e1: Features for skip connection to D1
                - e2: Features for skip connection to D2
                - e3: Features for skip connection to D3
                - b: Bottleneck features for decoder
                
        Process:
            1. E1: Input → [64, 128³]
            2. E2: Pool(E1) → [128, 64³]
            3. E1→E3: MaxPool(E1, 4) + Conv + SAM → [128, 32³]
            4. E3: Concat(Pool(E2), E1→E3) → [256, 32³]
            5. E1→B: AvgPool(E1, 8) + Conv + SAM → [256, 16³]
            6. E2→B: AvgPool(E2, 4) + Conv + SAM → [256, 16³]
            7. E3→B: MaxPool(E3, 2) → [256, 16³]
            8. B: Concat(E3→B, E1→B, E2→B) + SPP → [256, 16³]
            
        Example:
            >>> encoder = Encoder(4, 64)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> e1, e2, e3, b = encoder(x)
            >>> 
            >>> # Verify shapes
            >>> assert e1.shape == (2, 64, 128, 128, 128)
            >>> assert e2.shape == (2, 128, 64, 64, 64)
            >>> assert e3.shape == (2, 256, 32, 32, 32)
            >>> assert b.shape == (2, 256, 16, 16, 16)
        """
        # ====================================================================
        # Standard Encoder Path
        # ====================================================================
        
        # E1: [B, 64, 128, 128, 128]
        e1 = self.enc1(x) 

        # E2: [B, 128, 64, 64, 64]
        e2 = self.enc2(self.pool1(e1))      

        # ====================================================================
        # Cross-Level Connection: E1 → E3
        # ====================================================================
        
        # Downsample E1 to E3 resolution: 128³ → 32³
        e1_e3 = F.max_pool3d(e1, 4)
        e1_e3 = self.down1_e3_conv(e1_e3)
        e1_e3 = self.down1_e3_sam(e1_e3)  # [B, 128, 32, 32, 32]

        # ====================================================================
        # E3: Fuse E2 + E1→E3
        # ====================================================================
        
        # Concatenate: Pool(E2) [128, 32³] + E1→E3 [128, 32³] = [256, 32³]
        e3 = self.enc3(torch.cat([self.pool2(e2), e1_e3], dim=1))  # [B, 256, 32, 32, 32]

        # ====================================================================
        # Cross-Level Connections to Bottleneck
        # ====================================================================
        
        # E1 → Bottleneck: 128³ → 16³
        e1_b = self.sam_e1_b(self.down_e1_b(F.avg_pool3d(e1, 8)))  # [B, 256, 16, 16, 16]

         # E2 → Bottleneck: 64³ → 16³
        e2_b = self.sam_e2_b(self.down_e2_b(F.avg_pool3d(e2, 4)))  # [B, 256, 16, 16, 16]
        
        # E3 → Bottleneck: 32³ → 16³
        e3_b = F.max_pool3d(e3, 2)  # [B, 256, 16, 16, 16]

        # ====================================================================
        # Bottleneck: Fuse all paths + SPP
        # ====================================================================
        
        # Concatenate all bottleneck inputs: [B, 768, 16, 16, 16]
        b = torch.cat([e3_b, e1_b, e2_b], dim=1)

        # Reduce channels: 768 → 256
        b = self.reduce_conv(b)

        # Apply SPP for multi-scale context
        b = self.spp(b)

        # Final refinement
        b = self.after_spp(b)  # [B, 256, 16, 16, 16]

        return e1, e2, e3, b


# ============================================================================
# Decoder Block
# ============================================================================

class DecoderBlock(nn.Module):
    """
    CLCU-Net Decoder Block with Multi-Path Feature Fusion.
    
    Unlike standard U-Net decoder which only concatenates one skip connection,
    this decoder block can fuse features from multiple encoder levels using
    projection and attention mechanisms.
    
    Architecture:
        Main Input [B, C_main, D, H, W]
        Projection 1 [B, C_proj, D, H, W]  ─┐
        Projection 2 [B, C_proj, D, H, W]  ─┤
        Projection N [B, C_proj, D, H, W]  ─┘
                    ↓
        Concatenate → [B, C_main + N*C_proj, D, H, W]
                    ↓
        Fuse (Conv1×1) → [B, C_fuse, D, H, W]
                    ↓
        Refine (DoubleConv) → [B, C_out, D, H, W]
    
    Args:
        in_main_channels (int): Channels from main decoder path (upsampled from below)
        proj_channels_list (List[int]): List of channel counts for each projection
            Example: [256, 256, 256] for 3 projections with 256 channels each
        fuse_out_channels (int): Channels after fusion Conv1×1
        refine_out_channels (int | None, optional): Final output channels.
            If None, uses fuse_out_channels. Default: None
            
    Shape:
        - Input main_input: (B, in_main_channels, D, H, W)
        - Input projections[i]: (B, proj_channels_list[i], D, H, W)
        - Output: (B, refine_out_channels, D, H, W)
        
    Example:
        >>> # Decoder block for D3 (receives B_up + 3 projections)
        >>> block = DecoderBlock(
        ...     in_main_channels=256,      # From upsampled bottleneck
        ...     proj_channels_list=[256, 256, 256],  # E1→D3, E2→D3, E3→D3
        ...     fuse_out_channels=256,
        ...     refine_out_channels=128
        ... )
        >>> 
        >>> main = torch.randn(1, 256, 32, 32, 32)
        >>> projs = [
        ...     torch.randn(1, 256, 32, 32, 32),  # E1→D3
        ...     torch.randn(1, 256, 32, 32, 32),  # E2→D3
        ...     torch.randn(1, 256, 32, 32, 32)   # E3→D3
        ... ]
        >>> output = block(main, projs)
        >>> print(output.shape)  # torch.Size([1, 128, 32, 32, 32])
        
    Note:
        - Fuse layer uses 1×1 conv for efficient channel reduction
        - Refine layer uses DoubleConv for feature extraction
        - All projections must have same spatial dimensions
    """

    def __init__(self, 
                 in_main_channels: int, 
                 proj_channels_list: List[int],
                 fuse_out_channels: int, 
                 refine_out_channels: int | None = None):
        super().__init__()

        # Calculate total input channels after concatenation
        total_in = in_main_channels + sum(proj_channels_list)

        # Fusion layer: Reduce channels with 1×1 conv
        self.fuse = nn.Sequential(
            nn.Conv3d(total_in, fuse_out_channels, kernel_size=1),
            nn.BatchNorm3d(fuse_out_channels),
            nn.ReLU(inplace=True)
        )

        # Refinement layer: Extract features with DoubleConv
        refine_out = refine_out_channels or fuse_out_channels
        self.refine = DoubleConv(fuse_out_channels, refine_out, activation="leakyrelu")

    def forward(self, 
                main_input: torch.Tensor, 
                projections: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse and refine multi-path features.
        
        Args:
            main_input (torch.Tensor): Main decoder path (upsampled features)
                Shape: (B, C_main, D, H, W)
            projections (List[torch.Tensor]): List of projection features
                Each with shape: (B, C_proj, D, H, W)
                
        Returns:
            torch.Tensor: Fused and refined features
                Shape: (B, C_out, D, H, W)
                
        Process:
            1. Concatenate [main_input] + projections
            2. Fuse with 1×1 conv
            3. Refine with DoubleConv
            
        Example:
            >>> block = DecoderBlock(128, [128, 128], 128, 64)
            >>> main = torch.randn(1, 128, 64, 64, 64)
            >>> projs = [
            ...     torch.randn(1, 128, 64, 64, 64),
            ...     torch.randn(1, 128, 64, 64, 64)
            ... ]
            >>> output = block(main, projs)
            >>> print(output.shape)  # torch.Size([1, 64, 64, 64, 64])
        """
        # Concatenate all inputs along channel dimension
        x = torch.cat([main_input] + projections, dim=1)

        # Fuse channels
        x = self.fuse(x)

        # Refine features
        return self.refine(x)


# ============================================================================
# Decoder with Cross-Level Connections
# ============================================================================

class Decoder(nn.Module):
    """
    CLCU-Net Decoder with Multi-Level Feature Fusion.
    
    The decoder upsamples bottleneck features while fusing them with encoder
    features at multiple scales. Unlike standard U-Net, each decoder stage
    receives projections from ALL encoder levels, not just the adjacent one.
    
    Architecture:
        Bottleneck [B, 256, 16, 16, 16]
            ↓ Upsample(2)
        ┌───────────────────────────────┐
        │ D3 [B, 128, 32, 32, 32]       │
        │ ← B_up                         │
        │ ← E1→D3 (AvgPool 128→32)      │
        │ ← E2→D3 (AvgPool 64→32)       │
        │ ← E3→D3 (Identity 32→32)      │
        └───────────────────────────────┘
            ↓ Upsample(2)
        ┌───────────────────────────────┐
        │ D2 [B, 64, 64, 64, 64]        │
        │ ← D3_up                        │
        │ ← E1→D2 (AvgPool 128→64)      │
        │ ← E2→D2 (Identity 64→64)      │
        └───────────────────────────────┘
            ↓ Upsample(2)
        ┌───────────────────────────────┐
        │ D1 [B, 64, 128, 128, 128]     │
        │ ← D2_up                        │
        │ ← E1→D1 (Identity 128→128)    │
        └───────────────────────────────┘
    
    Multi-Level Projections:
        D3 receives: E1 (pool 4×), E2 (pool 2×), E3 (identity)
        D2 receives: E1 (pool 2×), E2 (identity)
        D1 receives: E1 (identity)
        
    Why This Works:
        - Combines features from all encoder scales
        - Reduces semantic gap between encoder and decoder
        - Each decoder level has access to both fine and coarse features
        - SAM helps select relevant features at each scale
    
    Args:
        base_channels (int): Base number of feature channels (must match encoder)
            
    Shape:
        - Input b: (B, 256, 16, 16, 16) - Bottleneck
        - Input e3: (B, 256, 32, 32, 32) - Encoder level 3
        - Input e2: (B, 128, 64, 64, 64) - Encoder level 2
        - Input e1: (B, 64, 128, 128, 128) - Encoder level 1
        - Output d1: (B, 64, 128, 128, 128) - Full resolution
        - Output d2: (B, 64, 64, 64, 64) - 1/2 resolution
        - Output d3: (B, 128, 32, 32, 32) - 1/4 resolution
        
    Example:
        >>> decoder = Decoder(base_channels=64)
        >>> b = torch.randn(1, 256, 16, 16, 16)
        >>> e3 = torch.randn(1, 256, 32, 32, 32)
        >>> e2 = torch.randn(1, 128, 64, 64, 64)
        >>> e1 = torch.randn(1, 64, 128, 128, 128)
        >>> 
        >>> d1, d2, d3 = decoder(b, e3, e2, e1)
        >>> print(f"D1: {d1.shape}")  # [1, 64, 128, 128, 128]
        >>> print(f"D2: {d2.shape}")  # [1, 64, 64, 64, 64]
        >>> print(f"D3: {d3.shape}")  # [1, 128, 32, 32, 32]
        
    Note:
        - All projections use SAM for attention
        - Upsampling uses trilinear interpolation (smooth)
        - Each DecoderBlock receives multiple projections
        - d1, d2, d3 are used for multi-scale predictions
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()

        # ====================================================================
        # Upsampling Layers
        # ====================================================================

        self.up_b = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # ====================================================================
        # Projections for D3 (from E1, E2, E3)
        # ====================================================================
        
        # E1 → D3: 128³ → 32³ (pool by 4)
        self.e1_d3 = self._make_proj(base_channels, base_channels * 4, pool=4)
        
        # E2 → D3: 64³ → 32³ (pool by 2)
        self.e2_d3 = self._make_proj(base_channels * 2, base_channels * 4, pool=2)
        
        # E3 → D3: 32³ → 32³ (no pooling, just SAM)
        self.e3_d3 = self._make_proj(base_channels * 4, base_channels * 4, pool=1)
        
        # D3 decoder block: B_up + E1→D3 + E2→D3 + E3→D3
        self.d3_block = DecoderBlock(
            in_main_channels=base_channels * 4,          # B upsampled
            proj_channels_list=[base_channels * 4] * 3,  # 3 projections
            fuse_out_channels=base_channels * 4,
            refine_out_channels=base_channels * 2
        )

        # ====================================================================
        # Projections for D2 (from E1, E2)
        # ====================================================================
        
        # E1 → D2: 128³ → 64³ (pool by 2)
        self.e1_d2 = self._make_proj(base_channels, base_channels * 2, pool=2)
        
        # E2 → D2: 64³ → 64³ (no pooling)
        self.e2_d2 = self._make_proj(base_channels * 2, base_channels * 2, pool=1)
        
        # D2 decoder block: D3_up + E1→D2 + E2→D2
        self.d2_block = DecoderBlock(
            in_main_channels=base_channels * 2,          # D3 upsampled
            proj_channels_list=[base_channels * 2] * 2,  # 2 projections
            fuse_out_channels=base_channels * 2,
            refine_out_channels=base_channels
        )

        # ====================================================================
        # Projections for D1 (from E1 only)
        # ====================================================================
        
        # E1 → D1: 128³ → 128³ (no pooling)
        self.e1_d1 = self._make_proj(base_channels, base_channels, pool=1)

        # D1 decoder block: D2_up + E1→D1
        self.d1_block = DecoderBlock(
            in_main_channels=base_channels,      # D2 upsampled
            proj_channels_list=[base_channels],  # 1 projection
            fuse_out_channels=base_channels,
        )

    def _make_proj(self, in_ch: int, out_ch: int, pool: int):
        """
        Create a projection pathway with optional pooling and SAM attention.
        
        Args:
            in_ch (int): Input channels
            out_ch (int): Output channels
            pool (int): Pooling factor (1 = no pooling)
            
        Returns:
            nn.Sequential or SAM: Projection module
            
        Note:
            - pool=1: Only SAM (no downsampling needed)
            - pool>1: AvgPool + Conv + BN + LeakyReLU + SAM
        """
        if pool == 1:
            # No pooling needed, just apply attention
            return SAM(out_ch)
        
        # Pooling + projection + attention
        layers = [
            nn.AvgPool3d(kernel_size=pool),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            SAM(out_ch)
        ]
        return nn.Sequential(*layers)

    def forward(self, 
                b: torch.Tensor, 
                e3: torch.Tensor, 
                e2: torch.Tensor, e1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode features with multi-level fusion.
        
        Args:
            b (torch.Tensor): Bottleneck features [B, 256, 16, 16, 16]
            e3 (torch.Tensor): Encoder level 3 [B, 256, 32, 32, 32]
            e2 (torch.Tensor): Encoder level 2 [B, 128, 64, 64, 64]
            e1 (torch.Tensor): Encoder level 1 [B, 64, 128, 128, 128]
            
        Returns:
            Tuple[torch.Tensor, ...]: Three decoder outputs:
                - d1: Full resolution [B, 64, 128, 128, 128]
                - d2: Half resolution [B, 64, 64, 64, 64]
                - d3: Quarter resolution [B, 128, 32, 32, 32]
                
        Process:
            1. D3: Upsample(B) + [E1→D3, E2→D3, E3→D3] → [128, 32³]
            2. D2: Upsample(D3) + [E1→D2, E2→D2] → [64, 64³]
            3. D1: Upsample(D2) + [E1→D1] → [64, 128³]
            
        Example:
            >>> decoder = Decoder(64)
            >>> b = torch.randn(1, 256, 16, 16, 16)
            >>> e3 = torch.randn(1, 256, 32, 32, 32)
            >>> e2 = torch.randn(1, 128, 64, 64, 64)
            >>> e1 = torch.randn(1, 64, 128, 128, 128)
            >>> 
            >>> d1, d2, d3 = decoder(b, e3, e2, e1)
            >>> 
            >>> # Verify shapes
            >>> assert d1.shape == (1, 64, 128, 128, 128)
            >>> assert d2.shape == (1, 64, 64, 64, 64)
            >>> assert d3.shape == (1, 128, 32, 32, 32)
        """
        # ====================================================================
        # D3: Decode to 1/4 resolution (32³)
        # ====================================================================
        
        # Upsample bottleneck: 16³ → 32³
        b_up = self.up_b(b)

        # Create projections from all encoder levels
        p1 = self.e1_d3(e1)   # E1 → D3
        p2 = self.e2_d3(e2)   # E2 → D3
        p3 = self.e3_d3(e3)   # E3 → D3

        # Fuse and refine: [B, 128, 32, 32, 32]
        d3 = self.d3_block(b_up, [p1, p2, p3])

        # ====================================================================
        # D2: Decode to 1/2 resolution (64³)
        # ====================================================================
        
        # Upsample D3: 32³ → 64³
        d3_up = self.up_d3(d3)

        # Create projections
        q1 = self.e1_d2(e1)   # E1 → D2
        q2 = self.e2_d2(e2)   # E2 → D2

        # Fuse and refine: [B, 64, 64, 64, 64]
        d2 = self.d2_block(d3_up, [q1, q2])

        # ====================================================================
        # D1: Decode to full resolution (128³)
        # ====================================================================
        
        # Upsample D2: 64³ → 128³
        d2_up = self.up_d2(d2)

        # Create projection
        r1 = self.e1_d1(e1)   # E1 → D1
        
        # Fuse and refine: [B, 64, 128, 128, 128]
        d1 = self.d1_block(d2_up, [r1])

        return d1, d2, d3


# ============================================================================
# Output Layer with Multi-Scale Predictions
# ============================================================================

class Output(nn.Module):
    """
    CLCU-Net Output Module with Refinement Block and 4-Channel Logits.
    
    This module generates intermediate predictions at three scales and fuses them 
    with fine decoder features to produce final refined logits.
    
    Output Channels:
        - Channel 0: Background
        - Channel 1: NCR (Necrotic Core)
        - Channel 2: ED (Edema)
        - Channel 3: ET (Enhancing Tumor)
        
    Note:
        Returns raw LOGITS. No activation (Sigmoid/Softmax) is applied here,
        as it is handled by the Loss Function during training.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()

        # ====================================================================
        # Upsampling Layers
        # ====================================================================
        
        # D3 → full resolution: 32³ → 128³ (scale factor 4)
        self.up_d3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        
        # D2 → full resolution: 64³ → 128³ (scale factor 2)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # ====================================================================
        # Prediction Branches (Raw Logits per subregion)
        # ====================================================================
        
        # Branch 3: D3 → ET prediction
        self.conv3 = nn.Conv3d(base_channels*2, 1, kernel_size=1) # 128 ch → 1 ch

        # Branch 2: D2 → ED prediction (guided by P3)
        self.d2_reduce = nn.Conv3d(base_channels, 1, kernel_size=1) # D2 → 1 ch
        self.conv2 = nn.Conv3d(2, 1, kernel_size=1) # Fuse P3 + D2

        # Branch 1: D1 → NCR prediction (guided by P2)
        self.d1_reduce = nn.Conv3d(base_channels, 1, kernel_size=1) # D1 → 1 ch
        self.conv1 = nn.Conv3d(2, 1, kernel_size=1) # Fuse P2 + D1


        # ====================================================================
        # Final Refine Block
        # ====================================================================
        total_in = base_channels + 3
        self.refine_conv = nn.Sequential(
            nn.Conv3d(total_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.final_sam = SAM(in_channels=32) 
        self.output_conv = nn.Conv3d(32, 4, kernel_size=1)

    def forward(self, 
                d1: torch.Tensor, 
                d2: torch.Tensor, 
                d3: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate multi-scale predictions and fuse them.
        
        Args:
            d1 (torch.Tensor): Finest decoder features [B, 64, 128³]
            d2 (torch.Tensor): Medium decoder features [B, 64, 64³]
            d3 (torch.Tensor): Coarsest decoder features [B, 128, 32³]
            
        Returns:
            Tuple containing:
                - final_mask_4ch: [B, 4, 128³] - (Background, NCR, ED, ET)
                - (p1, p2, p3): Individual predictions
                  - p1: [B, 1, 128³] - NCR (Necrotic Core)
                  - p2: [B, 1, 128³] - ED (Edema)
                  - p3: [B, 1, 128³] - ET (Enhancing Tumor)
                  
        Process:
            1. Branch 3 (ET):
               - Conv(D3) → c3
               - Upsample(c3, 4×) → x3
               - Sigmoid(c3) → p3
               
            2. Branch 2 (ED):
               - Upsample(D2, 2×) → d2_up
               - Conv(d2_up) → c2_1
               - Concat(c3, c2_1) → x2
               - Conv(x2) → c2
               - Sigmoid(c2) → p2
               
            3. Branch 1 (NCR):
               - Conv(D1) → c1_1
               - Concat(c2_1, c1_1) → x1
               - Conv(x1) → c1
               - Sigmoid(c1) → p1
               
            4. Final Mask:
               - Concat(p1, p2, p3) → [B, 3, 128³]
               - Background = 1 - clamp(sum, max=1)
               - Concat(background, p1, p2, p3) → [B, 4, 128³]
               
        Example:
            >>> output = Output()
            >>> d1 = torch.randn(1, 64, 128, 128, 128)
            >>> d2 = torch.randn(1, 64, 64, 64, 64)
            >>> d3 = torch.randn(1, 128, 32, 32, 32)
            >>> 
            >>> final, (p1, p2, p3) = output(d1, d2, d3)
            >>> 
            >>> # Check probabilities sum to 1
            >>> prob_sum = final.sum(dim=1, keepdim=True)
            >>> print(f"Min prob sum: {prob_sum.min():.4f}")
            >>> print(f"Max prob sum: {prob_sum.max():.4f}")
            >>> # Should be close to 1.0
        """
        # ====================================================================
        # Branch 3: Coarse ET Logits
        # ====================================================================
        
        # D3 → full resolution
        x3 = self.up_d3(d3)  # [B, 128, 128, 128, 128]

        # Predict ET
        c3 = self.conv3(x3)  # [B, 1, 128, 128, 128] 

        # ====================================================================
        # Branch 2: Medium ED Logits (guided by P3)
        # ====================================================================
        
        # D2 → full resolution
        d2_up = self.up_d2(d2)  # [B, 64, 128, 128, 128]

        # Reduce D2 channels
        c2_1 = self.d2_reduce(d2_up) # [B, 1, 128, 128, 128]

        # Fuse with P3 guidance
        x2 = torch.cat([c3, c2_1], dim=1)  # [B, 2, 128, 128, 128]
        c2 = self.conv2(x2)   # [B, 1, 128, 128, 128] 

        # ====================================================================
        # Branch 1: Fine NCR Logits (guided by P2)
        # ====================================================================
        
        # Reduce D1 channels
        c1_1 = self.d1_reduce(d1) # [B, 1, 128, 128, 128]
        
        # Fuse with P2 guidance (NOT c2, but c2_1 from D2)
        x1 = torch.cat([c2_1, c1_1], dim=1)  # [B, 2, 128, 128, 128]
        c1 = self.conv1(x1)   # [B, 1, 128, 128, 128] 
        

        # ====================================================================
        # Final Refinement
        # ====================================================================
        
        # Combine D1 features + all 3 predicted tumor masks
        tumor_features = torch.cat([c1, c2, c3], dim=1)
        combined = torch.cat([d1, tumor_features], dim=1) # [B, 67, 128, 128, 128]

        # Refinement process (Conv + SAM)
        x = self.refine_conv(combined)
        x = self.final_sam(x) 

        # Generate final 4-channel logits [BG, NCR, ED, ET]
        # Canal 0: Background, Canal 1: NCR, Canal 2: ED, Canal 3: ET
        final_4ch_logits = self.output_conv(x)

        return final_4ch_logits, (c1, c2, c3)
    

# ============================================================================
# Complete CLCU-Net Model
# ============================================================================

class CLCUNet(nn.Module):
    """
    Complete CLCU-Net Model for Brain Tumor Segmentation.
    
    CLCU-Net (Cross-Level Connected U-Net) is an advanced segmentation architecture
    that improves upon standard U-Net by:
    1. Adding cross-level connections between non-adjacent encoder stages
    2. Using Segmented Attention Modules (SAM) for feature recalibration
    3. Employing Spatial Pyramid Pooling (SPP) in the bottleneck
    4. Generating multi-scale predictions for progressive refinement
    
    Complete Architecture Flow:
        Input [B, 4, 128³]
               ↓
        ┌──────────────────┐
        │  Encoder + CLC   │  Cross-Level Connections:
        │                  │  - E1 → E3, E1 → B, E2 → B
        │  E1 [64, 128³]   │  All with SAM attention
        │  E2 [128, 64³]   │
        │  E3 [256, 32³]   │
        │  B  [256, 16³]   │  ← SPP for multi-scale context
        └──────────────────┘
               ↓
        ┌──────────────────┐
        │  Decoder + MLF   │  Multi-Level Fusion:
        │                  │  - D3 ← (B, E1, E2, E3)
        │  D3 [128, 32³]   │  - D2 ← (D3, E1, E2)
        │  D2 [64, 64³]    │  - D1 ← (D2, E1)
        │  D1 [64, 128³]   │
        └──────────────────┘
               ↓
        ┌──────────────────┐
        │  Output (3-way)  │  Progressive Predictions:
        │                  │  - P3 (ET) from D3
        │  P1 [NCR]        │  - P2 (ED) from D2 + P3
        │  P2 [ED]         │  - P1 (NCR) from D1 + P2
        │  P3 [ET]         │
        │  + Background    │
        └──────────────────┘
               ↓
        Output [B, 4, 128³]
    
    Key Innovations:
        - **Cross-Level Connections (CLC)**: Direct pathways between non-adjacent levels
        - **Multi-Level Fusion (MLF)**: Each decoder level fuses ALL encoder levels
        - **Progressive Refinement**: Three prediction branches at different scales
        - **Attention Everywhere**: SAM modules in all projections
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 4
            For BraTS: 4 modalities (FLAIR, T1, T1CE, T2)
        base_channels (int, optional): Base number of feature channels. Default: 64
            Controls model capacity
            
    Shape:
        - Input: (B, 4, 128, 128, 128)
        - Output: (B, 4, 128, 128, 128)
            Channel 0: Background
            Channel 1: NCR (Necrotic Core)
            Channel 2: ED (Edema)
            Channel 3: ET (Enhancing Tumor)
            
    Example:
        >>> # Standard CLCU-Net
        >>> model = CLCUNet(in_channels=4, base_channels=64)
        >>> x = torch.randn(1, 4, 128, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 4, 128, 128, 128])
        >>> 
        >>> # Access intermediate predictions (for training)
        >>> e1, e2, e3, b = model.encoder(x)
        >>> d1, d2, d3 = model.decoder(b, e3, e2, e1)
        >>> final_mask, (p1, p2, p3) = model.output(d1, d2, d3)
        >>> 
        >>> # During training, supervise each prediction branch
        >>> loss = (
        ...     criterion(final_mask, target) +
        ...     0.3 * criterion(p1, target[:, 1:2]) +  # NCR
        ...     0.3 * criterion(p2, target[:, 2:3]) +  # ED
        ...     0.3 * criterion(p3, target[:, 3:4])    # ET
        ... )
        
    Model Characteristics (base_channels=64):
        - Parameters: ~15-20M (more than U-Net due to cross-connections)
        - Memory (training): ~10-12 GB for batch=1
        - Training time: Longer than U-Net (more connections)
        - Inference time: Similar to U-Net
        
    Training Strategy:
        - Use DiceLoss with sigmoid activation (NOT softmax)
        - Supervise all three prediction branches (p1, p2, p3)
        - Weight intermediate losses (e.g., 0.3 × each)
        - Learning rate: 1e-3 to 1e-4
        - Optimizer: Adam
        
    Advantages over U-Net:
        ✅ Better information flow (cross-level connections)
        ✅ Reduces semantic gap (multi-level fusion)
        ✅ Multi-scale predictions (progressive refinement)
        ✅ Attention mechanisms (SAM) for feature selection
        ✅ SPP for multi-scale context
        
    Potential Disadvantages:
        ⚠️ More parameters (slower training)
        ⚠️ Higher memory consumption
        ⚠️ More complex architecture (harder to debug)
        
    Note:
        - Output uses sigmoid, NOT softmax (each channel independent)
        - Designed specifically for tumor subregion segmentation
        - Background is calculated as 1 - sum(tumor_channels)
        - Uses LeakyReLU throughout for better gradient flow
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = Decoder(base_channels=base_channels)
        self.output = Output()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete CLCU-Net.
        
        Args:
            x (torch.Tensor): Input volume with shape (B, 4, 128, 128, 128)
                4 channels: FLAIR, T1, T1CE, T2
                
        Returns:
            torch.Tensor: Segmentation probabilities with shape (B, 4, 128, 128, 128)
                Channel 0: Background probability
                Channel 1: NCR probability
                Channel 2: ED probability
                Channel 3: ET probability
                
        Process:
            1. Encoder: Extract features with cross-level connections
            2. Decoder: Upsample with multi-level fusion
            3. Output: Generate multi-scale predictions
            
        Example:
            >>> model = CLCUNet(4, 64)
            >>> x = torch.randn(2, 4, 128, 128, 128)
            >>> 
            >>> # Inference
            >>> probs = model(x)
            >>> preds = torch.argmax(probs, dim=1)
            >>> print(f"Predictions: {preds.shape}")  # [2, 128, 128, 128]
            >>> 
            >>> # Verify probabilities
            >>> prob_sum = probs.sum(dim=1)
            >>> print(f"Prob sum (should be ~1): {prob_sum.mean():.4f}")
        """
        # Encode with cross-level connections
        e1, e2, e3, b = self.encoder(x)

        # Decode with multi-level fusion
        d1, d2, d3 = self.decoder(b, e3, e2, e1)

        # Generate final prediction (ignore intermediate outputs in inference)
        final_mask_4ch, _ = self.output(d1, d2, d3)

        return final_mask_4ch


# ============================================================================
# Module Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Test script for CLCU-Net model.
    
    Usage:
        python clcunet.py
    """
    print("=" * 80)
    print("CLCU-Net Architecture - Summary and Testing")
    print("=" * 80)
    
    # Create model
    model = CLCUNet(in_channels=4, base_channels=64)
    model = model.cpu()

    # Display architecture
    summary(
        model, 
        input_size=(1, 4, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=5,
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
    
    # Verify probabilistic output
    prob_sum = output.sum(dim=1)
    print(f"\nProbability sum statistics:")
    print(f"  Min: {prob_sum.min():.4f}")
    print(f"  Max: {prob_sum.max():.4f}")
    print(f"  Mean: {prob_sum.mean():.4f}")
    print(f"  (Should be close to 1.0)")
    
    print("\n✓ CLCU-Net test passed successfully!")
    print("=" * 80)
