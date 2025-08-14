import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Módulo de atención espacial
class SpatialAttention(nn.Module):
    """
    Aplica atención espacial para recalibrar las características por posición (voxel).
    Usa una única convolución 3D para generar un mapa de atención, seguido de una función sigmoide.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el mapa de atención y lo aplica a la entrada.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma (B, C, D, H, W).
            
        Returns:
            torch.Tensor: Mapa de atención con forma (B, 1, D, H, W).
        """
        att_map = self.sigmoid(self.conv(x))
        return att_map

# Módulo de atención por canal
class ChannelAttention(nn.Module):
    """
    Aplica atención por canal para recalibrar la importancia de cada canal de características.
    Utiliza Global AvgPool y MaxPool para capturar información global, la pasa a través de un MLP
    (conv 3x3 → ReLU → conv 1x1) y usa una sigmoide para generar pesos por canal.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.reduction = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula los pesos de atención por canal y los devuelve.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma (B, C, D, H, W).
            
        Returns:
            torch.Tensor: Pesos de atención con forma (B, C, 1, 1, 1).
        """
        gap = F.adaptive_avg_pool3d(x, (1, 1, 1))
        gmp = F.adaptive_max_pool3d(x, (1, 1, 1))
        channel_att = self.sigmoid(self.mlp(gap) + self.mlp(gmp))
        return channel_att

# Módulo de atención segmentada
class SAM(nn.Module):
    """
    Combina atención espacial y por canal para recalibrar las características de forma secuencial.
    1. Aplica atención espacial.
    2. Usa la salida espacial para recalibrar los canales.
    3. Aplica la atención por canal resultante a la entrada original.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_att = SpatialAttention(in_channels)
        self.channel_att = ChannelAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la atención combinada a las características de entrada.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma (B, C, D, H, W).
            
        Returns:
            torch.Tensor: Tensor de salida recalibrado con la misma forma que la entrada.
        """
        spatial_map = self.spatial_att(x)
        x_spatial = spatial_map * x
        channel_map = self.channel_att(x_spatial)
        x_att = x * channel_map
        return x_att

# Bloque de doble convolución
class DoubleConv(nn.Module):
    """
    Bloque de doble convolución 3D que consiste en dos capas de Conv3D, BatchNorm y LeakyReLU.
    Mantiene la resolución espacial de la entrada debido a un padding de 1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el bloque de doble convolución a la entrada.
        
        Args:
            x (torch.Tensor): Tensor de entrada.
            
        Returns:
            torch.Tensor: Tensor de salida con el número de canales modificado.
        """
        return self.conv(x)

# Módulo de Spatial Pyramid Pooling
class SPP(nn.Module):
    """
    Módulo de Agregación de Pirámide Espacial para capturar contexto multiescala.
    Utiliza AdaptiveAvgPool en diferentes tamaños de salida, proyecta los canales,
    realiza un upsampling a la resolución original y concatena las características.
    """
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: List[int] = [2, 4, 8]):
        super(SPP, self).__init__()
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(output_size=ps),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        total_ch = in_channels + len(pool_sizes) * out_channels
        self.output_conv = nn.Sequential(
            nn.Conv3d(total_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa el tensor de entrada a través de las diferentes ramas del SPP.
        
        Args:
            x (torch.Tensor): Tensor de entrada.
            
        Returns:
            torch.Tensor: Tensor de salida fusionado.
        """
        size = x.shape[2:]
        pooled = [x]
        for layer in self.pool_layers:
            pooled_feat = layer(x)
            upsampled = F.interpolate(pooled_feat, size=size, mode='trilinear', align_corners=False)
            pooled.append(upsampled)
        x_cat = torch.cat(pooled, dim=1)
        return self.output_conv(x_cat)

# Encoder del modelo
class Encoder(nn.Module):
    """
    El encoder de CLCU-Net. Reduce la resolución espacial mientras aumenta los canales.
    Incorpora la conexión de puentes con atención (SAM) y el módulo SPP en el cuello de botella (Bottleneck).
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down1_e3_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(inplace=True)
        )
        self.down1_e3_sam = SAM(base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 4)
        self.down_e1_b = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e1_b = SAM(base_channels * 4)
        self.down_e2_b = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e2_b = SAM(base_channels * 4)
        self.reduce_conv = nn.Conv3d(base_channels * 12, base_channels * 4, kernel_size=1)
        self.spp = SPP(base_channels * 4, base_channels * 4)
        self.after_spp = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pasa la entrada a través de las etapas del encoder y el cuello de botella.
        
        Args:
            x (torch.Tensor): Tensor de entrada inicial.
            
        Returns:
            Tuple[torch.Tensor, ...]: Salidas intermedias de las capas del encoder (e1, e2, e3)
                                      y la salida del cuello de botella (b).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e1_e3 = F.max_pool3d(e1, 4)
        e1_e3 = self.down1_e3_conv(e1_e3)
        e1_e3 = self.down1_e3_sam(e1_e3)
        e3 = self.enc3(torch.cat([self.pool2(e2), e1_e3], dim=1))
        e1_b = self.sam_e1_b(self.down_e1_b(F.avg_pool3d(e1, 8)))
        e2_b = self.sam_e2_b(self.down_e2_b(F.avg_pool3d(e2, 4)))
        e3_b = F.max_pool3d(e3, 2)
        b = torch.cat([e3_b, e1_b, e2_b], dim=1)
        b = self.reduce_conv(b)
        b = self.spp(b)
        b = self.after_spp(b)
        return e1, e2, e3, b

# Bloque de decodificación
class DecoderBlock(nn.Module):
    """
    Bloque de decodificación que fusiona características de una rama principal y proyecciones
    de niveles del encoder. Utiliza una convolución 1x1 para fusionar canales y un bloque
    de doble convolución para refinar las características.
    """
    def __init__(self, in_main_channels: int, proj_channels_list: List[int],
                 fuse_out_channels: int, refine_out_channels: int | None = None):
        super().__init__()
        total_in = in_main_channels + sum(proj_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv3d(total_in, fuse_out_channels, kernel_size=1),
            nn.BatchNorm3d(fuse_out_channels),
            nn.ReLU(inplace=True)
        )
        refine_out = refine_out_channels or fuse_out_channels
        self.refine = DoubleConv(fuse_out_channels, refine_out)

    def forward(self, main_input: torch.Tensor, projections: List[torch.Tensor]) -> torch.Tensor:
        """
        Fusiona y refina las características de entrada.
        
        Args:
            main_input (torch.Tensor): Tensor de la rama principal (e.g., salida de un bloque superior).
            projections (List[torch.Tensor]): Lista de tensores de proyección de los niveles del encoder.
            
        Returns:
            torch.Tensor: Características decodificadas y refinadas.
        """
        x = torch.cat([main_input] + projections, dim=1)
        x = self.fuse(x)
        return self.refine(x)

# Decoder del modelo
class Decoder(nn.Module):
    """
    El decoder de CLCU-Net. Recupera la resolución espacial usando upsampling y fusionando
    características de los niveles del encoder con proyecciones y atención (SAM).
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.up_b = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.e1_d3 = self._make_proj(base_channels, base_channels * 4, pool=4)
        self.e2_d3 = self._make_proj(base_channels * 2, base_channels * 4, pool=2)
        self.e3_d3 = self._make_proj(base_channels * 4, base_channels * 4, pool=1)
        self.d3_block = DecoderBlock(
            in_main_channels=base_channels * 4,
            proj_channels_list=[base_channels * 4] * 3,
            fuse_out_channels=base_channels * 4,
            refine_out_channels=base_channels * 2
        )
        self.e1_d2 = self._make_proj(base_channels, base_channels * 2, pool=2)
        self.e2_d2 = self._make_proj(base_channels * 2, base_channels * 2, pool=1)
        self.d2_block = DecoderBlock(
            in_main_channels=base_channels * 2,
            proj_channels_list=[base_channels * 2] * 2,
            fuse_out_channels=base_channels * 2,
            refine_out_channels=base_channels
        )
        self.e1_d1 = self._make_proj(base_channels, base_channels, pool=1)
        self.d1_block = DecoderBlock(
            in_main_channels=base_channels,
            proj_channels_list=[base_channels],
            fuse_out_channels=base_channels,
        )

    def _make_proj(self, in_ch: int, out_ch: int, pool: int):
        """
        Crea una rama de proyección con o sin pooling, seguida de un módulo SAM.
        
        Args:
            in_ch (int): Número de canales de entrada.
            out_ch (int): Número de canales de salida.
            pool (int): Factor de pooling para la rama.
        """
        if pool == 1:
            return SAM(out_ch)
        layers = [
            nn.AvgPool3d(kernel_size=pool),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            SAM(out_ch)
        ]
        return nn.Sequential(*layers)

    def forward(self, b: torch.Tensor, e3: torch.Tensor, e2: torch.Tensor, e1: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodifica las características del cuello de botella y del encoder.
        
        Args:
            b (torch.Tensor): Salida del cuello de botella.
            e3, e2, e1 (torch.Tensor): Salidas intermedias del encoder.
            
        Returns:
            Tuple[torch.Tensor, ...]: Características decodificadas en diferentes resoluciones (d1, d2, d3).
        """
        b_up = self.up_b(b)
        p1 = self.e1_d3(e1)
        p2 = self.e2_d3(e2)
        p3 = self.e3_d3(e3)
        d3 = self.d3_block(b_up, [p1, p2, p3])

        d3_up = self.up_d3(d3)
        q1 = self.e1_d2(e1)
        q2 = self.e2_d2(e2)
        d2 = self.d2_block(d3_up, [q1, q2])

        d2_up = self.up_d2(d2)
        r1 = self.e1_d1(e1)
        d1 = self.d1_block(d2_up, [r1])
        return d1, d2, d3

# Capa de salida del modelo
class Output(nn.Module):
    """
    Capa de salida para la segmentación. Combina las tres salidas del decoder (d1, d2, d3)
    para generar las máscaras de segmentación multiescala.
    """
    def __init__(self):
        super().__init__()
        self.up_d3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv3 = nn.Conv3d(128, 1, kernel_size=1)
        self.d2_reduce = nn.Conv3d(64, 1, kernel_size=1)
        self.conv2 = nn.Conv3d(2, 1, kernel_size=1)
        self.d1_reduce = nn.Conv3d(64, 1, kernel_size=1)
        self.conv1 = nn.Conv3d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Genera y fusiona las máscaras de segmentación de las tres ramas del decoder.
        
        Args:
            d1, d2, d3 (torch.Tensor): Salidas del decoder.
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]: 
                - final_mask: Máscara final de 3 canales (NCR, ED, ET).
                - p1, p2, p3: Salidas de cada rama de segmentación.
        """
        # Rama 3: d3 (32x32x32) -> upsample a 128x128x128
        x3 = self.up_d3(d3)
        c3 = self.conv3(x3)
        p3 = self.sigmoid(c3)

        # Rama 2: d2 (64x64x64) -> upsample a 128x128x128
        d2_up = self.up_d2(d2)
        c2_1 = self.d2_reduce(d2_up)
        # Se fusiona con la salida de la rama 3 (c3)
        x2 = torch.cat([c3, c2_1], dim=1)
        c2 = self.conv2(x2)
        p2 = self.sigmoid(c2)

        # Rama 1: d1 (128x128x128)
        c1_1 = self.d1_reduce(d1)
        # Se fusiona con la salida de la rama 2 (c2_1)
        x1 = torch.cat([c2_1, c1_1], dim=1)
        c1 = self.conv1(x1)
        p1 = self.sigmoid(c1)

        final_mask = torch.cat([p1, p2, p3], dim=1)
        background = 1 - torch.clamp(final_mask.sum(dim=1, keepdim=True), max=1.0)
        final_mask_4ch = torch.cat([background, final_mask], dim=1)
        return final_mask_4ch, (p1, p2, p3)

# Modelo completo CLCU-Net
class CLCUNet(nn.Module):
    """
    Modelo completo de segmentación CLCU-Net.
    Compuesto por un Encoder, un Decoder y una capa de salida.
    Recibe un volumen 3D y devuelve una máscara de segmentación de 4 canales
    (fondo, NCR, ED, ET).
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = Decoder(base_channels=base_channels)
        self.output = Output()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flujo de datos a través del modelo.
        
        Args:
            x (torch.Tensor): Volumen de entrada (B, C, D, H, W).
            
        Returns:
            torch.Tensor: Máscara de segmentación de 4 canales.
        """
        e1, e2, e3, b = self.encoder(x)
        d1, d2, d3 = self.decoder(b, e3, e2, e1)
        final_mask_4ch, _ = self.output(d1, d2, d3)
        return final_mask_4ch

