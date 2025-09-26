import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchinfo import summary

class SeparableConv(nn.Module):
    """
    Bloque de convolución 3D separable.
    Consiste en una convolución depthwise y una pointwise.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super(MiddleFlow, self).__init__()
        self.blocks = nn.ModuleList([self._make_block() for _ in range(16)])

    def _make_block(self):
        block = nn.Sequential(
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1)
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super(BackboneXception, self).__init__()
        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow()
        self.exit = ExitFlow()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_level, x = self.entry(x)
        x = self.middle(x)
        high_level = self.exit(x)
        return low_level, high_level


class ASPP(nn.Module):
    """
    Módulo Atrous Spatial Pyramid Pooling (ASPP).
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 2, 4, 6]):
        super(ASPP, self).__init__()
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
        x1 = self.conv1(x)
        x2 = self.conv3_2(x)
        x3 = self.conv3_4(x)
        x4 = self.conv3_6(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x


class Encoder(nn.Module):
    """
    Combinación del backbone Xception y el módulo ASPP.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super(Encoder, self).__init__()
        self.backbone = BackboneXception(in_channels=in_channels)
        self.aspp = ASPP(in_channels=2048, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_level, high_level = self.backbone(x)
        aspp_output = self.aspp(high_level)
        return low_level, aspp_output


class Decoder(nn.Module):
    """
    Decodificador de DeepLabv3+ que combina características de alto y bajo nivel.
    """
    def __init__(self, low_level_in, num_classes):
        super(Decoder, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv3d(low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Sequential(
            nn.Conv3d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, aspp_output, low_level_feat):
        aspp_output = F.interpolate(aspp_output, size=low_level_feat.shape[2:], mode='trilinear', align_corners=False)
        low_level_feat = self.conv_low(low_level_feat)
        x = torch.cat([aspp_output, low_level_feat], dim=1)
        x = self.conv_final(x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class DeepLabV3Plus(nn.Module):
    """
    Implementación completa del modelo DeepLabv3+.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super(DeepLabV3Plus, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(low_level_in=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_level, aspp_output = self.encoder(x)
        output = self.decoder(aspp_output, low_level)
        return output


if __name__ == "__main__":
    # Prueba rápida del modelo DeepLabV3+
    input_size = (1, 4, 128, 128, 128)  # Ejemplo con un batch size de 1 y una sola modalidad
    model = DeepLabV3Plus(in_channels=4, num_classes=4)  # Ejemplo para segmentación en 3 clases
    summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params"], depth=5)