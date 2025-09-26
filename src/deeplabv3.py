import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchinfo import summary

from backbone import BackboneXception
from pooling import ASPP


class Encoder(nn.Module):
    """
    Codificador de DeepLabV3+.
    Combina:
      - Backbone Xception 3D para extracción jerárquica de características.
      - Módulo ASPP (Atrous Spatial Pyramid Pooling) para capturar contexto multiescala.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        """
        Args:
            in_channels (int): Número de canales de entrada (ej. 4 para imágenes médicas multicanal).
            out_channels (int): Número de canales de salida tras el ASPP.
        """
        super(Encoder, self).__init__()
        self.backbone = BackboneXception(in_channels=in_channels)
        self.aspp = ASPP(in_channels=2048, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Entrada (B, C, D, H, W).

        Returns:
            low_level (torch.Tensor): Características de bajo nivel (skip connections).
            aspp_output (torch.Tensor): Características enriquecidas tras ASPP.
        """
        low_level, high_level = self.backbone(x)
        aspp_output = self.aspp(high_level)
        return low_level, aspp_output


class Decoder(nn.Module):
    """
    Decodificador de DeepLabV3+ 3D.
    - Combina características de bajo nivel y alto nivel.
    - Refina la predicción con convoluciones y upsampling trilineal.
    """
    def __init__(self, low_level_in, num_classes):
        """
        Args:
            low_level_in (int): Número de canales en las características de bajo nivel.
            num_classes (int): Número de clases para segmentación.
        """
        super(Decoder, self).__init__()
        # Proyección de características de bajo nivel a 48 canales
        self.conv_low = nn.Sequential(
            nn.Conv3d(low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )

        # Bloques convolucionales para fusión
        self.conv_final = nn.Sequential(
            nn.Conv3d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

        # Convolución final que predice los mapas de clase
        self.out_conv = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, aspp_output, low_level_feat):
        """
        Args:
            aspp_output (torch.Tensor): Salida del ASPP.
            low_level_feat (torch.Tensor): Características de bajo nivel.

        Returns:
            torch.Tensor: Predicciones de segmentación (B, num_classes, D, H, W).
        """
        # Upsampling de ASPP a tamaño de low_level_feat
        aspp_output = F.interpolate(
            aspp_output, 
            size=low_level_feat.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )

        # Procesar low_level features
        low_level_feat = self.conv_low(low_level_feat)

        # Concatenar y refinar
        x = torch.cat([aspp_output, low_level_feat], dim=1)
        x = self.conv_final(x)

        # Upsampling a resolución original aproximada (factor 4)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)

        return self.out_conv(x)


class DeepLabV3Plus(nn.Module):
    """
    Implementación de DeepLabV3+ en 3D.
    Arquitectura:
      - Encoder: Backbone + ASPP.
      - Decoder: combinación de low-level y high-level features.
    """
    def __init__(self, in_channels: int, num_classes: int):
        """
        Args:
            in_channels (int): Número de canales de entrada.
            num_classes (int): Número de clases de salida.
        """
        super(DeepLabV3Plus, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(low_level_in=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Entrada (B, C, D, H, W).

        Returns:
            torch.Tensor: Predicciones (B, num_classes, D, H, W).
        """
        low_level, aspp_output = self.encoder(x)
        return self.decoder(aspp_output, low_level)


if __name__ == "__main__":
    # Test rápido para verificar arquitectura y parámetros
    model = DeepLabV3Plus(in_channels=4, num_classes=4)  
    summary(
        model, 
        input_size=(1, 4, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=5
    )
