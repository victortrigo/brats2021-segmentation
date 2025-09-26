import re
from typing import Optional, List, Union, Callable
import torch
import torch.nn as nn


class BaseObject(nn.Module):
    """
    Clase base para métricas y pérdidas, hereda de nn.Module.
    Proporciona un nombre automático y funcional para el objeto.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    @property
    def __name__(self) -> str:
        """
        Genera automáticamente un nombre snake_case a partir del nombre de la clase
        si no se proporciona un nombre.
        """
        if self._name is None:
            name: str = self.__class__.__name__
            s1: str = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    """
    Clase base para las métricas.
    """
    pass


class Loss(BaseObject):
    """
    Clase base para las funciones de pérdida.
    Sobrecarga operadores para combinar pérdidas de forma sencilla.
    """
    def __add__(self, other: 'Loss') -> 'SumOfLosses':
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: 'Loss') -> 'SumOfLosses':
        return self.__add__(other)

    def __mul__(self, value: Union[int, float]) -> 'MultipliedLoss':
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other: Union[int, float]) -> 'MultipliedLoss':
        return self.__mul__(other)


class SumOfLosses(Loss):
    """
    Combina dos funciones de pérdida sumándolas.
    """
    def __init__(self, l1: Loss, l2: Loss):
        name: str = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    """
    Multiplica una función de pérdida por un valor constante.
    """
    def __init__(self, loss: Loss, multiplier: Union[int, float]):
        # resolver el nombre
        if len(loss.__name__.split("+")) > 1:
            name: str = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name: str = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.multiplier * self.loss.forward(*inputs)


class Activation(nn.Module):
    """
    Wrapper para funciones de activación de PyTorch o personalizadas.
    """
    def __init__(self, name: Optional[Union[str, Callable]], **params):
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class ArgMax(nn.Module):
    """
    Clase auxiliar para aplicar argmax como una capa.
    """
    def __init__(self, dim: int = 1, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=self.dim)


def _take_channels(*xs: torch.Tensor, ignore_channels: Optional[List[int]] = None) -> List[torch.Tensor]:
    """
    Selecciona canales específicos de un tensor.
    """
    if ignore_channels is None:
        return list(xs)
    else:
        channels: List[int] = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return list(xs)


def _threshold(x: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
    """
    Aplica un umbral a un tensor para binarizarlo.
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7, threshold: Optional[float] = None,
        ignore_channels: Optional[List[int]] = None) -> torch.Tensor:
    """Calcula la Intersección sobre Unión (IoU) entre el valor real (ground truth) y la predicción.
    Args:
        pr (torch.Tensor): tensor de predicción
        gt (torch.Tensor): tensor del valor real
        eps (float): épsilon para evitar la división por cero
        threshold: umbral para la binarización de las salidas
    Returns:
        float: puntuación IoU (Jaccard)
    """
    
    pr = _threshold(pr, threshold=threshold)
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt_processed * pr_processed)
    union = torch.sum(gt_processed) + torch.sum(pr_processed) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr: torch.Tensor, gt: torch.Tensor, beta: float = 1, eps: float = 1e-7,
            threshold: Optional[float] = None, ignore_channels: Optional[List[int]] = None) -> torch.Tensor:
    """Calcula la puntuación F (F-score) entre el valor real y la predicción.
    Args:
        pr (torch.Tensor): tensor de predicción
        gt (torch.Tensor): tensor del valor real
        beta (float): constante positiva
        eps (float): épsilon para evitar la división por cero
        threshold: umbral para la binarización de las salidas
    Returns:
        float: puntuación F
    """
    
    pr = _threshold(pr, threshold=threshold)
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt_processed * pr_processed)
    fp = torch.sum(pr_processed) - tp
    fn = torch.sum(gt_processed) - tp

    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score


def accuracy(pr: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5,
             ignore_channels: Optional[List[int]] = None) -> torch.Tensor:
    """Calcula la puntuación de precisión (accuracy) entre el valor real y la predicción.
    Args:
        pr (torch.Tensor): tensor de predicción
        gt (torch.Tensor): tensor del valor real
        eps (float): épsilon para evitar la división por cero
        threshold: umbral para la binarización de las salidas
    Returns:
        float: puntuación de precisión
    """
    
    pr = _threshold(pr, threshold=threshold)
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Convertir a long para la comparación de igualdad si es necesario
    if pr_processed.dtype != gt_processed.dtype:
        gt_processed = gt_processed.type(pr_processed.dtype)
        
    tp = torch.sum(gt_processed == pr_processed, dtype=pr_processed.dtype)
    score = tp / gt_processed.view(-1).shape[0]
    return score


def precision(pr: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7, threshold: Optional[float] = None,
              ignore_channels: Optional[List[int]] = None) -> torch.Tensor:
    """Calcula la puntuación de precisión (precision) entre el valor real y la predicción.
    Args:
        pr (torch.Tensor): tensor de predicción
        gt (torch.Tensor): tensor del valor real
        eps (float): épsilon para evitar la división por cero
        threshold: umbral para la binarización de las salidas
    Returns:
        float: puntuación de precisión
    """
    
    pr = _threshold(pr, threshold=threshold)
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt_processed * pr_processed)
    fp = torch.sum(pr_processed) - tp

    score = (tp + eps) / (tp + fp + eps)
    return score


def recall(pr: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7, threshold: Optional[float] = None,
           ignore_channels: Optional[List[int]] = None) -> torch.Tensor:
    """Calcula la Exhaustividad (Recall) entre el valor real y la predicción.
    Args:
        pr (torch.Tensor): una lista de elementos predichos
        gt (torch.Tensor): una lista de elementos que deben ser predichos
        eps (float): épsilon para evitar la división por cero
        threshold: umbral para la binarización de las salidas
    Returns:
        float: puntuación de exhaustividad
    """
    
    pr = _threshold(pr, threshold=threshold)
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt_processed * pr_processed)
    fn = torch.sum(gt_processed) - tp

    score = (tp + eps) / (tp + fn + eps)
    return score


class JaccardLoss(Loss):
    """
    Función de pérdida de Jaccard (1 - Jaccard/IoU).
    """
    def __init__(self, eps: float = 1.0, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return 1.0 - jaccard(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(Loss):
    """
    Función de pérdida de Dice (1 - F-score).
    """
    def __init__(self, eps: float = 1.0, beta: float = 1.0, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return 1.0 - f_score(
            y_pr_activated,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, Loss):
    """Pérdida L1 de PyTorch con herencia de Loss."""
    pass


class MSELoss(nn.MSELoss, Loss):
    """Pérdida MSE de PyTorch con herencia de Loss."""
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, Loss):
    """Pérdida de Entropía Cruzada de PyTorch con herencia de Loss."""
    pass


class NLLLoss(nn.NLLLoss, Loss):
    """Pérdida NLL de PyTorch con herencia de Loss."""
    pass


class BCELoss(nn.BCELoss, Loss):
    """Pérdida de Entropía Cruzada Binaria de PyTorch con herencia de Loss."""
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Loss):
    """Pérdida de Entropía Cruzada Binaria con logits de PyTorch con herencia de Loss."""
    pass


class IoU(Metric):
    """
    Métrica de Intersección sobre Unión (IoU).
    """
    __name__ = "iou_score"

    def __init__(self, eps: float = 1e-7, threshold: float = 0.5, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'iou_score')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return iou(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(Metric):
    """
    Métrica F-score.
    """
    __name__ = 'fscore'
    
    def __init__(self, beta: float = 1, eps: float = 1e-7, threshold: float = 0.5,
                 activation: Optional[str] = None, ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'fscore')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return f_score(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(Metric):
    """
    Métrica de precisión (Accuracy).
    """
    __name__ = 'accuracy'
    
    def __init__(self, threshold: float = 0.5, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'accuracy')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return accuracy(
            y_pr_activated,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(Metric):
    """
    Métrica de recall.
    """
    __name__ = 'recall'
    
    def __init__(self, eps: float = 1e-7, threshold: float = 0.5, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'recall')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return recall(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(Metric):
    """
    Métrica de precisión (Precision).
    """
    __name__ = 'precision'
    
    def __init__(self, eps: float = 1e-7, threshold: float = 0.5, activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'precision')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        y_pr_activated = self.activation(y_pr)
        return precision(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
