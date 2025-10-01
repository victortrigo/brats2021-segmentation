import torch
import torch.nn as nn
from torchinfo import summary

class DoubleConv(nn.Module):
    """
    Bloque de doble convolución 3D configurable con función de activación.

    Aplica dos convoluciones 3D consecutivas, cada una seguida de BatchNorm3d y una función de activación configurable (ReLU o LeakyReLU).
    Mantiene la resolución espacial (D, H, W) del tensor de entrada.

    Args:
        in_channels (int): Número de canales de entrada.
        out_channels (int): Número de canales de salida.
        activation (str): 'relu' o 'leakyrelu' (por defecto 'relu').
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        if activation.lower() == "leakyrelu":
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act_layer,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            act_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
    
class SeparableConv(nn.Module):
    """
    Bloque de convolución 3D separable.
    Consiste en una convolución depthwise y una pointwise.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
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

# if __name__ == "__main__":
#     model = DoubleConv(in_channels=4, out_channels=64, activation="relu")
#     summary(model, input_size=(1, 4, 128, 128, 128), col_names=["input_size", "output_size", "num_params"], depth=4)
#     model = DoubleConv(in_channels=4, out_channels=64, activation="leakyrelu")
#     summary(model, input_size=(1, 4, 128, 128, 128), col_names=["input_size", "output_size", "num_params"], depth=4)
#     model = SeparableConv(in_channels=4, out_channels=64)
#     summary(model, input_size=(1, 4, 128, 128, 128), col_names=["input_size", "output_size", "num_params"], depth=4)

if __name__ == "__main__":
    modelos = [
        ("DoubleConv ReLU", DoubleConv(in_channels=4, out_channels=64, activation="relu")),
        ("DoubleConv LeakyReLU", DoubleConv(in_channels=4, out_channels=64, activation="leakyrelu")),
        ("SeparableConv", SeparableConv(in_channels=4, out_channels=64)),
    ]

    for nombre, model in modelos:
        print(f"\n==== {nombre} ====\n")
        summary(
            model,
            input_size=(1, 4, 128, 128, 128),
            col_names=["input_size", "output_size", "num_params"],
            depth=4
        )