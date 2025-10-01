import torch
import torch.nn as nn
from typing import Tuple
from torchinfo import summary

from convs import SeparableConv


class EntryFlow(nn.Module):
    """
    Entry Flow del backbone Xception 3D.
    - Reduce las dimensiones espaciales de la entrada.
    - Incrementa progresivamente el número de canales.
    - Introduce bloques residuales con convoluciones separables en 3D.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Primer bloque de reducción inicial
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

        # Bloque 2 con residual connection
        self.block2 = nn.Sequential(
            SeparableConv(64, 128, stride=1),
            SeparableConv(128, 128, stride=1),
            SeparableConv(128, 128, stride=2)
        )
        self.residual2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(128)
        )

        # Bloque 3 con residual connection
        self.block3 = nn.Sequential(
            SeparableConv(128, 256, stride=1),
            SeparableConv(256, 256, stride=1),
            SeparableConv(256, 256, stride=2)
        )
        self.residual3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(256)
        )

        # Bloque 4 con residual connection
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
        Args:
            x (torch.Tensor): Entrada del modelo con forma (B, C, D, H, W).

        Returns:
            low_level (torch.Tensor): Características de bajo nivel (usadas en el decoder).
            x (torch.Tensor): Características profundas extraídas.
        """
        # Bloques iniciales
        x = self.conv1(x)
        x = self.conv2(x)

        # Bloque 2
        residual = self.residual2(x)
        x = self.block2(x) + residual
        x = self.relu(x)
        low_level = x # Para skip connection

        # Bloque 3
        residual = self.residual3(x)
        x = self.block3(x) + residual
        x = self.relu(x)

        # Bloque 4
        residual = self.residual4(x)
        x = self.block4(x) + residual
        x = self.relu(x)

        return low_level, x


class MiddleFlow(nn.Module):
    """
    Middle Flow del backbone Xception 3D.
    - Conjunto de 16 bloques residuales con convoluciones separables.
    - Mantiene dimensionalidad, profundizando la representación.
    """
    def __init__(self, num_blocks: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([self._make_block() for _ in range(num_blocks)])

    def _make_block(self):
        """Crea un bloque de 3 convoluciones separables 3D."""
        block = nn.Sequential(
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1),
            SeparableConv(728, 728, stride=1)
        )
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor procesado por 16 bloques residuales.
        """
        for block in self.blocks:
            residual = x
            x = block(x) + residual
        return x


class ExitFlow(nn.Module):
    """
    Exit Flow del backbone Xception 3D.
    - Extrae características de alto nivel para la etapa de clasificación o decodificación.
    """
    def __init__(self, output_stride: int = 16):
        super().__init__()
        
        sepconv2_stride = 2 if output_stride == 32 else 1


        # Bloque inicial con residual connection
        self.block = nn.Sequential(
            SeparableConv(728, 1024, stride=1),
            SeparableConv(1024, 1024, stride=1),
            SeparableConv(1024, 1024, stride=2) 
        )
        self.residual = nn.Sequential(
            nn.Conv3d(728, 1024, kernel_size=1, stride=2, bias=False), 
            nn.BatchNorm3d(1024)
        )

        # Progresiva expansión de canales
        self.sepconv1 = SeparableConv(1024, 1536, stride=1)
        self.sepconv2 = SeparableConv(1536, 1536, stride=sepconv2_stride) 
        self.sepconv3 = SeparableConv(1536, 2048, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Entrada del flujo intermedio.

        Returns:
            torch.Tensor: Características de alto nivel con 2048 canales.
        """
        residual = self.residual(x)
        x = self.block(x) + residual
        x = self.relu(x)

        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        return x


class BackboneXception(nn.Module):
    """
    Backbone Xception 3D completo.
    - Compuesto por EntryFlow, MiddleFlow y ExitFlow.
    - Inspirado en la arquitectura Xception para imágenes 3D.
    """
    def __init__(self, 
                 in_channels: int = 4, 
                 output_stride: int = 16,
                 num_middle_blocks: int = 16):
        super().__init__()

        # Validación estricta: Solo se permiten 16 o 32
        ALLOWED_STRIDES = {16, 32}
        if output_stride not in ALLOWED_STRIDES:
            raise ValueError(
                f"Output Stride '{output_stride}' no es compatible para el backbone Xception 3D. "
                f"Solo se permiten: {ALLOWED_STRIDES}."
            )
        
        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow(num_blocks=num_middle_blocks)
        self.exit = ExitFlow(output_stride=output_stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Entrada (B, C, D, H, W).

        Returns:
            low_level (torch.Tensor): Características de bajo nivel (para skip connections).
            high_level (torch.Tensor): Características de alto nivel (para el decoder o clasificador).
        """
        low_level, x = self.entry(x)
        x = self.middle(x)
        high_level = self.exit(x)
        return low_level, high_level



if __name__ == "__main__":
    """
    Args:
        x (torch.Tensor): Entrada (B, C, D, H, W).

    Returns:
        low_level (torch.Tensor): Características de bajo nivel (para skip connections).
        high_level (torch.Tensor): Características de alto nivel (para el decoder o clasificador).
    """
    for os in [16, 32]:
        print(f"\n==== Backbone con output_stride={os} ====\n")
        model = BackboneXception(in_channels=4, output_stride=os)
        summary(
            model,
            input_size=(1, 4, 128, 128, 128),
            col_names=["input_size", "output_size", "num_params"],
            depth=2
        )
