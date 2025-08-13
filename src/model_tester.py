import torch
import torch.nn as nn
from typing import Tuple, Type
from torchinfo import summary 

class ModelTester:
    """
    Clase para probar y resumir modelos de PyTorch.
    
    Esta clase puede probar cualquier modelo que siga un formato estándar
    de `forward` y que acepte una entrada de tensor con el tamaño especificado.
    """
    def __init__(self, model_class: Type[nn.Module], input_size: Tuple[int, ...], **model_kwargs):
        """
        Inicializa el probador de modelos.

        Args:
            model_class (Type[nn.Module]): La clase del modelo a probar (e.g., UNet).
            input_size (Tuple[int, ...]): La forma del tensor de entrada (e.g., (1, 1, 64, 128, 128)).
            **model_kwargs: Argumentos de palabras clave para el constructor del modelo.
        """
        self.model = model_class(**model_kwargs)
        self.input_size = input_size
        self.model_kwargs = model_kwargs

    def run_test(self):
        """Ejecuta la prueba del modelo, la inferencia y el resumen."""
        print(f"--- Probando el modelo {self.model.__class__.__name__} ---")
        try:
            # Crear un tensor de entrada simulado
            dummy_input = torch.randn(self.input_size)
            
            # Realizar la inferencia
            output = self.model(dummy_input)

            print("✅ Modelo creado y probado con éxito.")
            print(f"Forma de entrada: {dummy_input.shape}")
            print(f"Forma de salida: {output.shape}")

            # Generar el resumen del modelo
            print("\n--- Resumen del modelo ---")
            print(summary(
                self.model,
                input_size=self.input_size,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                depth=5
            ))
        except Exception as e:
            print(f"❌ Error al probar el modelo: {e}")
            raise
