import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Bloque de doble convolución 3D.

    Este módulo aplica dos convoluciones 3D consecutivas, cada una seguida de una 
    normalización por lotes (`BatchNorm3d`) y una función de activación ReLU. 
    Mantiene la resolución espacial (D, H, W) del tensor de entrada. El uso de 
    `bias=False` es una buena práctica cuando se utiliza `BatchNorm` para evitar 
    parámetros redundantes.

    Args:
        in_channels (int): Número de canales de entrada.
        out_channels (int): Número de canales de salida.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class EncoderBlock(nn.Module):
    """
    Bloque de contracción del encoder 3D.

    Combina un `DoubleConv` para extraer características y un `MaxPool3d` para
    reducir la resolución espacial a la mitad. Devuelve tanto la salida del 
    `DoubleConv` (para las conexiones de salto) como la salida del `MaxPool` 
    (para la siguiente etapa del encoder).

    Args:
        in_channels (int): Número de canales de entrada.
        out_channels (int): Número de canales de salida después de la doble convolución.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.double_conv(x)
        pool_out = self.maxpool(conv_out)
        return conv_out, pool_out

class UNetEncoder(nn.Module):
    """
    Ruta de contracción (encoder) para la arquitectura UNet 3D.

    Define una secuencia de `EncoderBlock` para reducir progresivamente el tamaño
    espacial del volumen de entrada y aumentar el número de canales, capturando 
    características de bajo nivel y semánticas.
    """
    def __init__(self, in_channels: int):
        super(UNetEncoder, self).__init__()
        self.enc_block1 = EncoderBlock(in_channels, 64)
        self.enc_block2 = EncoderBlock(64, 128)
        self.enc_block3 = EncoderBlock(128, 256)
        self.enc_block4 = EncoderBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        conv1, pool1 = self.enc_block1(x)
        conv2, pool2 = self.enc_block2(pool1)
        conv3, pool3 = self.enc_block3(pool2)
        conv4, pool4 = self.enc_block4(pool3)
        bottleneck_out = self.bottleneck(pool4)
        return conv1, conv2, conv3, conv4, bottleneck_out

class UpConvBlock(nn.Module):
    """
    Bloque de expansión (decoder) de UNet 3D con conexión de salto.

    Realiza el upsampling del tensor del nivel inferior, lo concatena con el 
    tensor de la conexión de salto del encoder y aplica una doble convolución
    para refinar las características. Incluye un manejo de padding para alinear
    las dimensiones espaciales.

    Args:
        in_channels_up (int): Canales del tensor de upsampling.
        skip_channels (int): Canales del tensor de la conexión de salto.
        out_channels (int): Canales de salida después de la doble convolución.
    """
    def __init__(self, in_channels_up: int, skip_channels: int, out_channels: int):
        super(UpConvBlock, self).__init__()
        self.up_transpose = nn.ConvTranspose3d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels_up // 2 + skip_channels, out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up_transpose(x_up)
      
        # Alineación de dimensiones espaciales con padding
        if x_skip.shape[2:] != x_up.shape[2:]:
            diff = [x_skip.shape[i+2] - x_up.shape[i+2] for i in range(3)]
            x_up = F.pad(x_up, [diff[2] // 2, diff[2] - diff[2] // 2,
                                diff[1] // 2, diff[1] - diff[1] // 2,
                                diff[0] // 2, diff[0] - diff[0] // 2])

        x_combined = torch.cat([x_skip, x_up], dim=1)
        return self.double_conv(x_combined)

class UNetDecoder(nn.Module):
    """
    Ruta de expansión (decoder) para la arquitectura UNet 3D.

    Define una secuencia de `UpConvBlock` para reconstruir la segmentación. 
    Combina características de alta resolución del encoder con características 
    semánticas de baja resolución del cuello de botella.
    """
    def __init__(self, base_channels: int):
        super(UNetDecoder, self).__init__()
        self.up_block1 = UpConvBlock(in_channels_up=base_channels * 16, skip_channels=base_channels * 8, out_channels=base_channels * 8)
        self.up_block2 = UpConvBlock(in_channels_up=base_channels * 8, skip_channels=base_channels * 4, out_channels=base_channels * 4)
        self.up_block3 = UpConvBlock(in_channels_up=base_channels * 4, skip_channels=base_channels * 2, out_channels=base_channels * 2)
        self.up_block4 = UpConvBlock(in_channels_up=base_channels * 2, skip_channels=base_channels, out_channels=base_channels)
      
    def forward(self, bottleneck_out: torch.Tensor, conv4: torch.Tensor, conv3: torch.Tensor, conv2: torch.Tensor, conv1: torch.Tensor) -> torch.Tensor:
        x = self.up_block1(bottleneck_out, conv4) 
        x = self.up_block2(x, conv3)             
        x = self.up_block3(x, conv2)             
        x = self.up_block4(x, conv1)             
        return x

class UNet(nn.Module):
    """
    Modelo UNet 3D completo para segmentación de imágenes volumétricas.

    Este modelo implementa la arquitectura UNet de tres dimensiones. Se compone 
    de un `UNetEncoder` para la extracción de características, un `UNetDecoder`
    para la reconstrucción de la segmentación y una capa final de convolución 
    para obtener el mapa de salida con el número de clases deseado.

    Args:
        in_channels (int): Número de canales de la imagen de entrada 
                           (e.g., 1 para escala de grises, 4 para datos médicos multimodales).
        num_classes (int): Número de clases para la segmentación 
                           (e.g., 2 para binario, >2 para multiclase).
    """
    def __init__(self,  in_channels: int = 1, num_classes: int = 2, base_channels: int = 64):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(base_channels)
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ruta de contracción (encoder)
        conv1, conv2, conv3, conv4, bottleneck_out = self.encoder(x)

        # Ruta de expansión (decoder)
        decoder_out = self.decoder(bottleneck_out, conv4, conv3, conv2, conv1)
        
        # Capa de salida para generar las clases de segmentación
        output = self.out_conv(decoder_out)
        return output
  
