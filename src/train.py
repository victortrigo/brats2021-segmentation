import sys
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Union


class Meter(object):
    """
    Los medidores (Meters) proporcionan una forma de hacer un seguimiento de estadísticas importantes de manera online. 
    Esta clase es abstracta, pero provee una interfaz estándar para que todos los medidores la sigan.
    """

    def reset(self) -> None:
        """
        Reinicia el medidor a la configuración por defecto.
        """
        pass

    def add(self, value: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """
        Registra un nuevo valor en el medidor.

        Args:
            value: El valor a incluir.
            n: El número de elementos representados por `value`.
        """
        pass

    def value(self) -> Tuple[float, float]:
        """
        Obtiene el valor del medidor en el estado actual (media y desviación estándar).
        """
        pass


class AverageValueMeter(Meter):
    """
    Un medidor para calcular el promedio y la desviación estándar de una serie de valores.
    """
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val: float = 0.0

    def add(self, value: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """
        Añade un valor al medidor y actualiza las estadísticas.

        Args:
            value: El valor a agregar.
            n: El número de elementos representados por el valor (por defecto 1).
        """
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.detach().cpu().numpy().mean()
        elif isinstance(value, np.ndarray):
            value = value.mean()
            
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean: float = np.nan
            self.std: float = np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # Forzar una copia para evitar problemas de referencia
            self.std = float('inf')
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self) -> Tuple[float, float]:
        """
        Devuelve el valor actual promedio y la desviación estándar.
        """
        return self.mean, self.std

    def reset(self) -> None:
        """
        Reinicia el medidor a su estado inicial.
        """
        self.n: int = 0
        self.sum: float = 0.0
        self.var: float = 0.0
        self.val: float = 0.0
        self.mean: float = np.nan
        self.mean_old: float = 0.0
        self.m_s: float = 0.0
        self.std: float = np.nan


class Epoch:
    """
    Clase base para las épocas de entrenamiento y validación.
    """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: List[nn.Module],
        stage_name: str,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Inicializa una época.

        Args:
            model: El modelo de red neuronal.
            loss: La función de pérdida.
            metrics: Una lista de funciones de métricas.
            stage_name: Nombre de la etapa (e.g., "train" o "valid").
            device: El dispositivo para ejecutar el modelo (e.g., "cpu" o "cuda").
            verbose: Si se debe mostrar el progreso en la barra de tqdm.
        """
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self) -> None:
        """
        Mueve el modelo, la pérdida y las métricas al dispositivo especificado.
        """
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs: Dict[str, float]) -> str:
        """
        Formatea el diccionario de logs en una cadena de texto.
        """
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Método abstracto para actualizar un lote de datos. Debe ser implementado
        por las subclases.
        """
        raise NotImplementedError

    def on_epoch_start(self) -> None:
        """
        Hook para ejecutar código al inicio de la época.
        """
        pass

    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Ejecuta una época completa sobre un dataloader.

        Args:
            dataloader: El cargador de datos para la época.

        Returns:
            Un diccionario con las métricas y la pérdida promediadas de la época.
        """
        self.on_epoch_start()

        logs: Dict[str, float] = {}
        loss_meter = AverageValueMeter()
        metrics_meters: Dict[str, AverageValueMeter] = {
            metric.name: AverageValueMeter() for metric in self.metrics
        }

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                y = y.long()
                loss, y_pred = self.batch_update(x, y)

                # Actualizar logs de pérdida
                loss_value: float = loss.cpu().detach().numpy().mean()
                loss_meter.add(loss_value)
                loss_logs: Dict[str, float] = {f"{self.loss.__name__}": loss_meter.mean}
                logs.update(loss_logs)
                y_squeezed = y.squeeze(dim=1)

                # Actualizar logs de métricas
                for metric_fn in self.metrics:
                    metric_value: float = metric_fn(y_pred, y_squeezed).cpu().detach().numpy().mean()
                    metrics_meters[metric_fn.name].add(metric_value)
                metrics_logs: Dict[str, float] = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    """
    Clase para la época de entrenamiento, maneja el paso de `forward`,
    cálculo de pérdida, `backward` y `optimizer.step()`.
    """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: List[nn.Module],
        optimizer: Optimizer,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Inicializa una época de entrenamiento.

        Args:
            optimizer: El optimizador para actualizar los pesos del modelo.
        """
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self) -> None:
        """
        Establece el modelo en modo de entrenamiento.
        """
        self.model.train()

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza un paso de entrenamiento para un lote.

        Args:
            x: Tensor de entrada.
            y: Tensor de etiquetas.

        Returns:
            La pérdida calculada y la predicción del modelo.
        """
        self.optimizer.zero_grad()
        prediction = self.model.forward(x.float())
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    """
    Clase para la época de validación, maneja el paso de `forward` sin
    actualizar los pesos del modelo.
    """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: List[nn.Module],
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Inicializa una época de validación.
        """
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self) -> None:
        """
        Establece el modelo en modo de evaluación.
        """
        self.model.eval()

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza un paso de validación para un lote.

        Args:
            x: Tensor de entrada.
            y: Tensor de etiquetas.

        Returns:
            La pérdida calculada y la predicción del modelo.
        """
        with torch.no_grad():
            prediction = self.model.forward(x.float())
            loss = self.loss(prediction, y)
        return loss, prediction
      
