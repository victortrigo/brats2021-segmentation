import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type


# Módulo de atención espacial
class SpatialAttention(nn.Module):
    """
    Aplica atención espacial para recalibrar las características por posición (voxel).
    Usa una única convolución 3D para generar un mapa de atención, seguido de una función sigmoide.
    """
    def __init__(self, in_channels: int):
        """
        Inicializa el módulo de atención espacial.

        Args:
            in_channels (int): Número de canales del tensor de entrada.
        """
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
        """
        Inicializa el módulo de atención por canal.

        Args:
            in_channels (int): Número de canales del tensor de entrada.
            reduction_ratio (int): Relación de reducción para el MLP.
        """
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
        """
        Inicializa el módulo SAM.

        Args:
            in_channels (int): Número de canales del tensor de entrada.
        """
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


class SeparableConv(nn.Module):
    """
    Bloque de convolución 3D separable.
    Consiste en una convolución depthwise y una pointwise.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Inicializa el bloque de convolución separable.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            kernel_size (int): Tamaño del kernel.
            stride (int): Paso de la convolución.
            padding (int): Relleno de la convolución.
        """
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la convolución separable a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida de la convolución.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EntryFlow(nn.Module):
    """
    Primer flujo del backbone. Reduce las dimensiones espaciales y aumenta los canales.
    """
    def __init__(self, in_channels: int):
        """
        Inicializa el EntryFlow.

        Args:
            in_channels (int): Número de canales de entrada.
        """
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            SeparableConv(64, 128, stride=1),
            SeparableConv(128, 128, stride=1),
            SeparableConv(128, 128, stride=2)
        )
        self.residual2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(128)
        )
        self.block3 = nn.Sequential(
            SeparableConv(128, 256, stride=1),
            SeparableConv(256, 256, stride=1),
            SeparableConv(256, 256, stride=2)
        )
        self.residual3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(256)
        )
        self.block4 = nn.Sequential(
            SeparableConv(256, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1)
        )
        self.residual4 = nn.Sequential(
            nn.Conv3d(256, 728, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(728)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica el flujo de entrada a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Características de bajo nivel y de alto nivel.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.residual2(x)
        x = self.block2(x)
        x = x + residual
        x = self.relu(x)
        low_level = x
        residual = self.residual3(x)
        x = self.block3(x)
        x = x + residual
        x = self.relu(x)
        residual = self.residual4(x)
        x = self.block4(x)
        x = x + residual
        x = self.relu(x)
        return low_level, x


class MiddleFlow(nn.Module):
    """
    Flujo intermedio del backbone. Compuesto por 16 bloques residuales.
    """
    def __init__(self):
        """
        Inicializa el MiddleFlow.
        """
        super(MiddleFlow, self).__init__()
        self.blocks = nn.ModuleList([self._make_block() for _ in range(16)])

    def _make_block(self):
        return nn.Sequential(
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el flujo intermedio a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida del MiddleFlow.
        """
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        return x


class ExitFlow(nn.Module):
    """
    Flujo de salida del backbone. Produce la salida de alto nivel.
    """
    def __init__(self):
        """
        Inicializa el ExitFlow.
        """
        super(ExitFlow, self).__init__()
        self.block = nn.Sequential(
            SeparableConv(728, 1024, stride=1),
            SeparableConv(1024, 1024, stride=1),
            SeparableConv(1024, 1024, stride=1)
        )
        self.residual = nn.Sequential(
            nn.Conv3d(728, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1024)
        )
        self.sepconv1 = SeparableConv(1024, 1536, stride=1)
        self.sepconv2 = SeparableConv(1536, 1536, stride=1)
        self.sepconv3 = SeparableConv(1536, 2048, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el flujo de salida a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida de alto nivel del ExitFlow.
        """
        residual = self.residual(x)
        x = self.block(x)
        x = x + residual
        x = self.relu(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        return x


class BackboneXception(nn.Module):
    """
    Backbone completo de Xception 3D, compuesto por los tres flujos.
    """
    def __init__(self, in_channels: int = 4):
        """
        Inicializa el BackboneXception.

        Args:
            in_channels (int): Número de canales del tensor de entrada.
        """
        super(BackboneXception, self).__init__()
        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow()
        self.exit = ExitFlow()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica el backbone a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Características de bajo nivel y de alto nivel.
        """
        low_level, x = self.entry(x)
        x = self.middle(x)
        high_level = self.exit(x)
        return low_level, high_level


class ASPP(nn.Module):
    """
    Módulo Atrous Spatial Pyramid Pooling (ASPP) con atención SAM en cada rama.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 2, 4, 6]):
        """
        Inicializa el módulo ASPP.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            atrous_rates (list[int]): Tasas de atrous para las convoluciones.
        """
        super(ASPP, self).__init__()
        # Se crean cinco instancias de SAM, una para cada rama del ASPP
        self.sam1 = SAM(in_channels=out_channels)
        self.sam2 = SAM(in_channels=out_channels)
        self.sam3 = SAM(in_channels=out_channels)
        self.sam4 = SAM(in_channels=out_channels)
        self.sam5 = SAM(in_channels=out_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                      dilation=atrous_rates[2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3_6 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[3],
                      dilation=atrous_rates[3], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el módulo ASPP a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida del ASPP.
        """
        # Aplicamos SAM a cada una de las 5 ramas de ASPP
        x1 = self.sam1(self.conv1(x))
        x2 = self.sam2(self.conv3_2(x))
        x3 = self.sam3(self.conv3_4(x))
        x4 = self.sam4(self.conv3_6(x))
        
        x5 = self.global_pool(x)
        x5 = self.sam5(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x


class Encoder(nn.Module):
    """
    Combinación del backbone Xception y el módulo ASPP.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        """
        Inicializa el Encoder.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
        """
        super(Encoder, self).__init__()
        self.backbone = BackboneXception(in_channels=in_channels)
        self.aspp = ASPP(in_channels=2048, out_channels=out_channels)
        
        # Módulo de convolución 1x1x1 y atención SAM para la rama del backbone
        self.backbone_branch_conv = nn.Conv3d(in_channels=2048, out_channels=256, kernel_size=1)
        self.backbone_branch_sam = SAM(in_channels=256)

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica el encoder a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Características de bajo nivel y salida del ASPP.
        """
        low_level, high_level = self.backbone(x)
        aspp_output = self.aspp(high_level)
        
        # Procesar las características de alto nivel con la rama de atención SAM
        high_level_attention = self.backbone_branch_conv(high_level)
        high_level_attention = self.backbone_branch_sam(high_level_attention)

        # Concatenar la salida del ASPP y la rama de atención
        concat_output = torch.cat([aspp_output, high_level_attention], dim=1)

        # Retornar las características de bajo nivel y el nuevo output del encoder
        return low_level, concat_output


class Decoder(nn.Module):
    """
    Decodificador de DeepLabv3+ que combina características de alto y bajo nivel.
    """
    def __init__(self, low_level_in, num_classes):
        """
        Inicializa el Decoder.

        Args:
            low_level_in (int): Número de canales de las características de bajo nivel.
            num_classes (int): Número de clases para la segmentación.
        """
        super(Decoder, self).__init__()
        self.sam_low_level = SAM(in_channels=low_level_in)
        
        self.conv_low = nn.Sequential(
            nn.Conv3d(low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Sequential(
            nn.Conv3d(256 + 256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, aspp_output, low_level_feat):
        """
        Aplica el decodificador a la entrada.

        Args:
            aspp_output (torch.Tensor): Salida del módulo ASPP.
            low_level_feat (torch.Tensor): Características de bajo nivel.

        Returns:
            torch.Tensor: Mapa de segmentación con la forma (B, num_classes, D, H, W).
        """
        aspp_output = F.interpolate(aspp_output, size=low_level_feat.shape[2:], mode='trilinear', align_corners=False)
        
        low_level_feat_att = self.sam_low_level(low_level_feat)
        
        low_level_feat_conv = self.conv_low(low_level_feat_att)
        
        x = torch.cat([aspp_output, low_level_feat_conv], dim=1)
        x = self.conv_final(x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class DeepLabV3PlusSAM(nn.Module):
    """
    Implementación completa del modelo DeepLabv3+ con mecanismo de atención SAM.
    """
    def __init__(self, in_channels: int, num_classes: int):
        """
        Inicializa el modelo DeepLabV3PlusSAM.

        Args:
            in_channels (int): Número de canales del tensor de entrada.
            num_classes (int): Número de clases para la segmentación.
        """
        super(DeepLabV3PlusSAM, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(low_level_in=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el modelo completo a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Mapa de segmentación.
        """
        low_level, aspp_output = self.encoder(x)
        output = self.decoder(aspp_output, low_level)
        return output
