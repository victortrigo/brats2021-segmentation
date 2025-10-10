import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchinfo import summary
from sam import SAM

# Módulo de Spatial Pyramid Pooling
class SPP(nn.Module):
    """
    Módulo de Agregación de Pirámide Espacial para capturar contexto multiescala.
    Utiliza AdaptiveAvgPool en diferentes tamaños de salida, proyecta los canales,
    realiza un upsampling a la resolución original y concatena las características.
    """
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: List[int] = [2, 4, 8]):
        super().__init__()
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


# Módulo de Atroux Spatial Pyramid Pooling
class ASPP(nn.Module):
    """
    Módulo Atrous Spatial Pyramid Pooling (ASPP), opcionalmente con atención SAM en cada rama.
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 2, 4, 6], use_sam=False):
        super().__init__()
        self.use_sam = use_sam

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

        if self.use_sam:
            self.sam1 = SAM(in_channels=out_channels)
            self.sam2 = SAM(in_channels=out_channels)
            self.sam3 = SAM(in_channels=out_channels)
            self.sam4 = SAM(in_channels=out_channels)
            self.sam5 = SAM(in_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv3_2(x)
        x3 = self.conv3_4(x)
        x4 = self.conv3_6(x)
        x5 = self.global_pool(x)

        if self.use_sam:
            x1 = self.sam1(x1)
            x2 = self.sam2(x2)
            x3 = self.sam3(x3)
            x4 = self.sam4(x4)
            x5 = self.sam5(x5)

        x5 = F.interpolate(x5, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x
    

if __name__ == "__main__":
    modelos = [
        ("SPP", SPP(in_channels=256, out_channels=256)),
        ("ASPP", ASPP(in_channels=256, out_channels=256)),
        ("ASPP + SAM", ASPP(in_channels=256, out_channels=256, use_sam=True)),
    ]

    for nombre, model in modelos:
        print(f"\n==== {nombre} ====\n")
        summary(
            model,
            input_size=(1, 256, 16, 16, 16),
            col_names=["input_size", "output_size", "num_params"],
            depth=3
        )

