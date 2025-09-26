import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torchinfo import summary

from convs import DoubleConv
from pooling import SPP
from sam import SAM


# Encoder del modelo
class Encoder(nn.Module):
    """
    El encoder de CLCU-Net.
    - Reduce progresivamente la resolución espacial (downsampling).
    - Incrementa los canales para capturar más información semántica.
    - Usa conexiones con atención (SAM) y un módulo SPP en el bottleneck.
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        # Primer bloque convolucional
        self.enc1 = DoubleConv(in_channels, base_channels, activation="leakyrelu")
        self.pool1 = nn.MaxPool3d(2)

        # Segundo bloque convolucional
        self.enc2 = DoubleConv(base_channels, base_channels * 2, activation="leakyrelu")
        self.pool2 = nn.MaxPool3d(2)

        # Rama e1 -> e3 con atención
        self.down1_e3_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(inplace=True)
        )
        self.down1_e3_sam = SAM(base_channels * 2)

        # Tercer bloque convolucional
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 4, activation="leakyrelu")
        
        # Conexiones e1 -> bottleneck
        self.down_e1_b = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e1_b = SAM(base_channels * 4)

        # Conexiones e2 -> bottleneck
        self.down_e2_b = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.sam_e2_b = SAM(base_channels * 4)

        # Bottleneck final con reducción de canales y SPP
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
        e1 = self.enc1(x)                   # [B, 64, 128, 128, 128]
        e2 = self.enc2(self.pool1(e1))      # [B, 128, 64, 64, 64]

        # Ruta de proyección e1 -> e3
        e1_e3 = F.max_pool3d(e1, 4)
        e1_e3 = self.down1_e3_conv(e1_e3)
        e1_e3 = self.down1_e3_sam(e1_e3)

        e3 = self.enc3(torch.cat([self.pool2(e2), e1_e3], dim=1)) # [B, 256, 32, 32, 32]

        # Conexiones hacia el bottleneck
        e1_b = self.sam_e1_b(self.down_e1_b(F.avg_pool3d(e1, 8)))
        e2_b = self.sam_e2_b(self.down_e2_b(F.avg_pool3d(e2, 4)))
        e3_b = F.max_pool3d(e3, 2)

        # Bottleneck + SPP
        b = torch.cat([e3_b, e1_b, e2_b], dim=1)
        b = self.reduce_conv(b)
        b = self.spp(b)
        b = self.after_spp(b)
        return e1, e2, e3, b


# -----------------------------
# Bloque de decodificación
# -----------------------------
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
        self.refine = DoubleConv(fuse_out_channels, refine_out, activation="leakyrelu")

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

# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    """
    El decoder:
    - Recupera resolución espacial mediante upsampling.
    - Fusiona características del encoder con proyecciones y atención.
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.up_b = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Proyecciones para d3
        self.e1_d3 = self._make_proj(base_channels, base_channels * 4, pool=4)
        self.e2_d3 = self._make_proj(base_channels * 2, base_channels * 4, pool=2)
        self.e3_d3 = self._make_proj(base_channels * 4, base_channels * 4, pool=1)
        self.d3_block = DecoderBlock(
            in_main_channels=base_channels * 4,
            proj_channels_list=[base_channels * 4] * 3,
            fuse_out_channels=base_channels * 4,
            refine_out_channels=base_channels * 2
        )

        # Proyecciones para d2
        self.e1_d2 = self._make_proj(base_channels, base_channels * 2, pool=2)
        self.e2_d2 = self._make_proj(base_channels * 2, base_channels * 2, pool=1)
        self.d2_block = DecoderBlock(
            in_main_channels=base_channels * 2,
            proj_channels_list=[base_channels * 2] * 2,
            fuse_out_channels=base_channels * 2,
            refine_out_channels=base_channels
        )

        # Proyecciones para d1
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
            d1: resolución completa (128³)
            d2: resolución media (64³)
            d3: resolución baja (32³)
        """
        b_up = self.up_b(b)
        p1, p2, p3 = self.e1_d3(e1), self.e2_d3(e2), self.e3_d3(e3)
        d3 = self.d3_block(b_up, [p1, p2, p3])

        d3_up = self.up_d3(d3)
        q1, q2 = self.e1_d2(e1), self.e2_d2(e2)
        d2 = self.d2_block(d3_up, [q1, q2])

        d2_up = self.up_d2(d2)
        r1 = self.e1_d1(e1)
        d1 = self.d1_block(d2_up, [r1])
        return d1, d2, d3

# Capa de salida del modelo
class Output(nn.Module):
    """
    Capa de salida simplificada:
    - Una sola conv 1x1 convierte canales de d1 en clases.
    - Devuelve logits de tamaño [B, num_classes, D, H, W].
    """
    def __init__(self):
        super().__init__()
        self.up_d3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Reducción de canales intermedios
        self.conv3 = nn.Conv3d(128, 1, kernel_size=1) # Rama d3
        self.d2_reduce = nn.Conv3d(64, 1, kernel_size=1) # Rama d2
        self.conv2 = nn.Conv3d(2, 1, kernel_size=1) # Fusión d2 + d3
        self.d1_reduce = nn.Conv3d(64, 1, kernel_size=1) # Rama d1
        self.conv1 = nn.Conv3d(2, 1, kernel_size=1) # Fusión d1 + d2

        
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

if __name__ == "__main__":
    # Prueba rápida del modelo CLCU-Net
    model = CLCUNet(in_channels=4, base_channels=64)
    summary(
        model, 
        input_size=(1, 4, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=5
    )
