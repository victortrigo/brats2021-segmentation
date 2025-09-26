import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchinfo import summary

from sam import SAM
from backbone import BackboneXception
from pooling import ASPP


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
        self.aspp = ASPP(in_channels=2048, out_channels=out_channels, use_sam=True)
        
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

if __name__ == "__main__":
    # Prueba rápida del modelo DeepLabV3+
    input_size = (1, 4, 128, 128, 128)  # Ejemplo con un batch size de 1 y una sola modalidad
    model = DeepLabV3PlusSAM(in_channels=4, num_classes=4)  # Ejemplo para segmentación en 3 clases
    summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params"], depth=5)