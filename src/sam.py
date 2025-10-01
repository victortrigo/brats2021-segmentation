import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

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
    
if __name__ == "__main__":
    # Prueba rápida del módulo SAM
    sam = SAM(in_channels=64)
    summary(
        sam, 
        input_size=(1, 64, 128, 128, 128), 
        col_names=["input_size", "output_size", "num_params"], 
        depth=5
    )

