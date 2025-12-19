"""
Metrics and Loss Functions for Medical Image Segmentation.

This module provides a comprehensive collection of metrics and loss functions
specifically designed for semantic segmentation tasks, with particular focus
on medical image segmentation (e.g., brain tumor segmentation in BraTS).

Key Components:
    - **Base Classes**: BaseObject, Metric, Loss
    - **Loss Functions**: DiceLoss, JaccardLoss, BCE, CrossEntropy, etc.
    - **Metrics**: IoU, Dice (F-score), Accuracy, Precision, Recall
    - **Activation Wrappers**: Sigmoid, Softmax, ArgMax
    - **Loss Arithmetic**: Combine losses with +, * operators

Design Philosophy:
    - Modular: Each metric/loss is a self-contained module
    - Composable: Losses can be combined (e.g., Dice + CrossEntropy)
    - Flexible: Support for various activation functions
    - Efficient: Optimized for GPU computation
    - Medical-focused: Handle class imbalance and small objects

Common Use Cases:
    
    1. Training with Dice Loss:
        >>> criterion = DiceLoss()
        >>> loss = criterion(predictions, targets)
        
    2. Combined Loss:
        >>> criterion = DiceLoss() + 0.5 * CrossEntropyLoss()
        >>> loss = criterion(predictions, targets)
        
    3. Evaluation Metrics:
        >>> metrics = [IoU(), Fscore(), Accuracy()]
        >>> for metric in metrics:
        ...     score = metric(predictions, targets)
        
    4. Multi-class Segmentation:
        >>> criterion = DiceLoss(activation='softmax')
        >>> iou = IoU(threshold=None)  # For softmax output

Activation Functions:
    - None/Identity: For raw logits
    - Sigmoid: Binary or multi-label segmentation
    - Softmax: Multi-class segmentation (mutually exclusive)
    - ArgMax: Convert probabilities to class indices

Class Imbalance Handling:
    Dice and Jaccard losses are particularly effective for imbalanced
    datasets (common in medical imaging where background >> foreground).
    They focus on overlap rather than pixel-wise accuracy.
"""


import re
from typing import Optional, List, Union, Callable
import torch
import torch.nn as nn


# ============================================================================
# Base Classes
# ============================================================================

class BaseObject(nn.Module):
    """
    Base class for all metrics and losses.
    
    Provides automatic naming functionality by converting class names
    to snake_case format. For example, "DiceLoss" becomes "dice_loss".
    
    Args:
        name (str, optional): Custom name for the object. If None, generates
            name automatically from class name. Default: None
            
    Example:
        >>> class MyCustomMetric(BaseObject):
        ...     pass
        >>> metric = MyCustomMetric()
        >>> print(metric.__name__)
        >>> # 'my_custom_metric'
        
    Note:
        - Inherits from nn.Module for seamless PyTorch integration
        - Automatic GPU/CPU handling through .to(device)
        - State dict support for checkpointing
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    @property
    def __name__(self) -> str:
        """
        Generate or return the object name.
        
        Returns:
            str: Snake_case name derived from class name or custom name
            
        Example:
            >>> loss = DiceLoss()
            >>> print(loss.__name__)
            >>> # 'dice_loss'
            >>> 
            >>> metric = IoU(name='custom_iou')
            >>> print(metric.__name__)
            >>> # 'custom_iou'
        """
        if self._name is None:
            # Convert CamelCase to snake_case
            name: str = self.__class__.__name__
            # Insert underscore before uppercase letters
            s1: str = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    """
    Base class for all metrics.
    
    Metrics are used to evaluate model performance during training and
    validation. Unlike losses, metrics are not used for backpropagation.
    
    Common Metrics:
        - IoU (Intersection over Union / Jaccard Index)
        - Dice Score (F1-Score)
        - Accuracy
        - Precision
        - Recall
        
    Example:
        >>> class CustomMetric(Metric):
        ...     def forward(self, y_pred, y_true):
        ...         return torch.mean((y_pred - y_true).abs())
        >>> 
        >>> metric = CustomMetric()
        >>> score = metric(predictions, targets)
        
    Note:
        - All metrics should implement forward() method
        - Output should be a scalar tensor or float
        - Metrics are typically computed with torch.no_grad()
    """
    pass


class Loss(BaseObject):
    """
    Base class for all loss functions.
    
    Losses are used during training to compute gradients for backpropagation.
    This class provides operator overloading for easy loss combination.
    
    Operator Overloading:
        - Addition (+): Combine losses
          Example: DiceLoss() + CrossEntropyLoss()
          
        - Multiplication (*): Weight losses
          Example: 0.5 * DiceLoss()
          
    Common Loss Functions:
        - DiceLoss: For segmentation with class imbalance
        - JaccardLoss: Similar to Dice, different formulation
        - CrossEntropyLoss: Standard classification loss
        - BCELoss: Binary cross-entropy
        
    Example:
        >>> # Single loss
        >>> criterion = DiceLoss()
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Combined loss
        >>> criterion = DiceLoss() + 0.5 * JaccardLoss()
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Weighted loss
        >>> criterion = 2.0 * DiceLoss()
        >>> loss = criterion(predictions, targets)
        
    Note:
        - All losses should implement forward() method
        - Output should be differentiable scalar tensor
        - Lower values indicate better fit
    """

    def __add__(self, other: 'Loss') -> 'SumOfLosses':
        """
        Add two loss functions.
        
        Args:
            other (Loss): Another loss function to add
            
        Returns:
            SumOfLosses: Combined loss that computes sum of both
            
        Example:
            >>> loss = DiceLoss() + JaccardLoss()
            >>> result = loss(pred, target)
            >>> # result = dice_loss(pred, target) + jaccard_loss(pred, target)
        """
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other: 'Loss') -> 'SumOfLosses':
        """Right-hand addition (a + Loss)."""
        return self.__add__(other)

    def __mul__(self, value: Union[int, float]) -> 'MultipliedLoss':
        """
        Multiply loss by a scalar weight.
        
        Args:
            value (int or float): Weight multiplier
            
        Returns:
            MultipliedLoss: Weighted loss
            
        Example:
            >>> loss = 0.5 * DiceLoss()
            >>> result = loss(pred, target)
            >>> # result = 0.5 * dice_loss(pred, target)
        """
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other: Union[int, float]) -> 'MultipliedLoss':
        """Right-hand multiplication (scalar * Loss)."""
        return self.__mul__(other)


class SumOfLosses(Loss):
    """
    Represents the sum of two loss functions.
    
    This class is created automatically when using the + operator between
    two Loss objects. It computes both losses and returns their sum.
    
    Args:
        l1 (Loss): First loss function
        l2 (Loss): Second loss function
        
    Example:
        >>> # Created automatically with + operator
        >>> combined = DiceLoss() + JaccardLoss()
        >>> 
        >>> # Or manually
        >>> combined = SumOfLosses(DiceLoss(), JaccardLoss())
        >>> 
        >>> # Use like any loss
        >>> loss_value = combined(predictions, targets)
        
    Note:
        - Name is automatically set to "loss1 + loss2"
        - Both losses receive same inputs
        - Gradients flow through both losses
    """

    def __init__(self, l1: Loss, l2: Loss):
        name: str = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute sum of both losses.
        
        Args:
            *inputs: Arguments passed to both loss functions
            
        Returns:
            torch.Tensor: Sum of both loss values
        """
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    """
    Represents a loss function multiplied by a constant.
    
    This class is created automatically when using the * operator between
    a scalar and a Loss object. It scales the loss by the multiplier.
    
    Args:
        loss (Loss): Base loss function
        multiplier (int or float): Scaling factor
        
    Example:
        >>> # Created automatically with * operator
        >>> weighted = 0.5 * DiceLoss()
        >>> 
        >>> # Or manually
        >>> weighted = MultipliedLoss(DiceLoss(), 0.5)
        >>> 
        >>> # Use like any loss
        >>> loss_value = weighted(predictions, targets)
        >>> # Returns: 0.5 * dice_loss(predictions, targets)
        
    Note:
        - Name is automatically set to "multiplier * loss_name"
        - Useful for balancing multiple losses
        - Common pattern: main_loss + 0.5 * auxiliary_loss
    """

    def __init__(self, loss: Loss, multiplier: Union[int, float]):
        # Generate descriptive name
        if len(loss.__name__.split("+")) > 1:
            # Loss is already a sum, add parentheses
            name: str = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name: str = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled loss.
        
        Args:
            *inputs: Arguments passed to base loss function
            
        Returns:
            torch.Tensor: Loss value multiplied by multiplier
        """
        return self.multiplier * self.loss.forward(*inputs)


# ============================================================================
# Activation Functions
# ============================================================================

class Activation(nn.Module):
    """
    Wrapper for activation functions.
    
    Provides a unified interface for various activation functions used in
    segmentation tasks. Supports both PyTorch built-in activations and
    custom callables.
    
    Supported Activations:
        - 'identity' or None: No activation (pass-through)
        - 'sigmoid': Binary segmentation, multi-label
        - 'softmax': Multi-class segmentation (mutually exclusive)
        - 'softmax2d': Softmax over spatial dimensions (uncommon)
        - 'logsoftmax': Log of softmax (for NLLLoss)
        - 'argmax': Convert probabilities to class indices
        - 'argmax2d': ArgMax over channel dimension
        - Callable: Custom activation function
        
    Args:
        name (str or Callable, optional): Activation function name or callable
        **params: Additional parameters for the activation
        
    Example:
        >>> # Sigmoid for binary segmentation
        >>> act = Activation('sigmoid')
        >>> probs = act(logits)
        >>> 
        >>> # Softmax for multi-class
        >>> act = Activation('softmax', dim=1)
        >>> probs = act(logits)
        >>> 
        >>> # Custom activation
        >>> def custom_act(x):
        ...     return torch.tanh(x) * 0.5 + 0.5
        >>> act = Activation(custom_act)
        >>> output = act(input)
        
    Note:
        - Activations are typically applied before computing metrics
        - Some losses (e.g., CrossEntropy) include activation internally
        - For training, consider whether activation should be in loss or model
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
            raise ValueError(
                f'Activation should be callable/sigmoid/softmax/logsoftmax/None; '
                f'got {name}'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated tensor
        """
        return self.activation(x)


class ArgMax(nn.Module):
    """
    ArgMax operation as a PyTorch module.
    
    Returns the indices of maximum values along a specified dimension.
    Useful for converting probability predictions to class labels.
    
    Args:
        dim (int, optional): Dimension along which to apply argmax. Default: 1
            For segmentation: dim=1 (channel dimension)
        **kwargs: Additional arguments (unused, for compatibility)
        
    Example:
        >>> argmax = ArgMax(dim=1)
        >>> probs = torch.randn(2, 4, 128, 128, 128)
        >>> preds = argmax(probs)
        >>> print(preds.shape)
        >>> # torch.Size([2, 128, 128, 128])
        >>> # Class indices for each voxel
        
    Note:
        - Output has one less dimension than input (dimension is reduced)
        - Common pattern: softmax → argmax for final predictions
        - Not differentiable (use for inference only)
    """

    def __init__(self, dim: int = 1, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute argmax along specified dimension.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Indices of maximum values
        """
        return torch.argmax(x, dim=self.dim)


# ============================================================================
# Helper Functions
# ============================================================================

def _take_channels(
        *xs: torch.Tensor, 
        ignore_channels: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
    """
    Select specific channels from tensors.
    
    Useful for excluding certain classes from metric/loss computation
    (e.g., ignore background class).
    
    Args:
        *xs: Input tensors
        ignore_channels (List[int], optional): Channel indices to exclude
            
    Returns:
        List[torch.Tensor]: Tensors with selected channels
        
    Example:
        >>> pred = torch.randn(2, 4, 64, 64, 64)
        >>> target = torch.randn(2, 4, 64, 64, 64)
        >>> 
        >>> # Ignore background (channel 0)
        >>> pred_fg, target_fg = _take_channels(pred, target, ignore_channels=[0])
        >>> print(pred_fg.shape)
        >>> # torch.Size([2, 3, 64, 64, 64])
        
    Note:
        - All input tensors must have same shape
        - Channels are indexed starting from 0
        - Useful for computing metrics on foreground only
    """
    if ignore_channels is None:
        return list(xs)
    else:
        # Select all channels except ignored ones
        channels: List[int] = [
            channel for channel in range(xs[0].shape[1]) 
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) 
            for x in xs
        ]
        return list(xs)


def _threshold(x: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
    """
    Apply threshold to convert probabilities to binary predictions.
    
    Args:
        x (torch.Tensor): Input tensor (typically probabilities)
        threshold (float, optional): Threshold value. If None, no thresholding
            Common values: 0.5 for binary segmentation
            
    Returns:
        torch.Tensor: Binarized tensor (0s and 1s) or original if threshold=None
        
    Example:
        >>> probs = torch.tensor([0.2, 0.6, 0.8, 0.4])
        >>> binary = _threshold(probs, threshold=0.5)
        >>> print(binary)
        >>> # tensor([0., 1., 1., 0.])
        
    Note:
        - Values > threshold become 1, others become 0
        - Output dtype matches input dtype
        - Used in metrics computation
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


# ============================================================================
# Core Metric Functions
# ============================================================================

def iou(
        pr: torch.Tensor, 
        gt: torch.Tensor, 
        eps: float = 1e-7, 
        threshold: Optional[float] = None,
        ignore_channels: Optional[List[int]] = None
    ) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU / Jaccard Index).
    
    IoU measures the overlap between prediction and ground truth:
    IoU = (Intersection) / (Union)
    IoU = (TP) / (TP + FP + FN)
    
    Where:
        - TP: True Positives (correct predictions)
        - FP: False Positives (incorrect predictions)
        - FN: False Negatives (missed targets)
    
    Range: [0, 1], where 1 is perfect segmentation
    
    Args:
        pr (torch.Tensor): Predictions (B, C, *spatial)
            Can be probabilities [0, 1] or logits
        gt (torch.Tensor): Ground truth labels (B, C, *spatial)
            Binary (0 or 1) or one-hot encoded
        eps (float, optional): Small constant to avoid division by zero. 
            Default: 1e-7
        threshold (float, optional): Threshold for binarizing predictions.
            Default: None (assumes already binary)
        ignore_channels (List[int], optional): Channels to exclude from computation
            Example: [0] to ignore background. Default: None
            
    Returns:
        torch.Tensor: IoU score (scalar)
        
    Example:
        >>> pred = torch.sigmoid(torch.randn(2, 4, 64, 64, 64))
        >>> target = torch.randint(0, 2, (2, 4, 64, 64, 64)).float()
        >>> 
        >>> # Compute IoU
        >>> score = iou(pred, target, threshold=0.5)
        >>> print(f"IoU: {score:.4f}")
        >>> 
        >>> # Ignore background channel
        >>> score = iou(pred, target, threshold=0.5, ignore_channels=[0])
        >>> print(f"Foreground IoU: {score:.4f}")
        
    Note:
        - Also known as Jaccard Index
        - Commonly used in segmentation competitions
        - Sensitive to small objects (low IoU if prediction is off)
        - eps prevents division by zero for empty predictions
        
    Mathematical Formula:
        IoU = (sum(pred * gt) + eps) / (sum(pred) + sum(gt) - sum(pred * gt) + eps)
    """
    # Apply threshold if specified
    pr = _threshold(pr, threshold=threshold)

    # Select channels if specified
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Calculate intersection and union
    intersection = torch.sum(gt_processed * pr_processed)
    union = torch.sum(gt_processed) + torch.sum(pr_processed) - intersection + eps

    return (intersection + eps) / union


# Alias for consistency
jaccard = iou


def f_score(
        pr: torch.Tensor, 
        gt: torch.Tensor, 
        beta: float = 1, 
        eps: float = 1e-7,
        threshold: Optional[float] = None, 
        ignore_channels: Optional[List[int]] = None
    ) -> torch.Tensor:
    """
    Calculate F-score (F-beta score / Dice coefficient when beta=1).
    
    F-score is the harmonic mean of precision and recall:
    F_beta = ((1 + beta²) × Precision × Recall) / (beta² × Precision + Recall)
    
    When beta=1 (default), this is the F1-score or Dice coefficient:
    F1 = (2 × TP) / (2 × TP + FP + FN)
    
    Range: [0, 1], where 1 is perfect segmentation
    
    Args:
        pr (torch.Tensor): Predictions (B, C, *spatial)
        gt (torch.Tensor): Ground truth labels (B, C, *spatial)
        beta (float, optional): Weight of recall in harmonic mean. Default: 1
            - beta < 1: Emphasize precision (fewer false positives)
            - beta = 1: Equal weight (F1-score / Dice)
            - beta > 1: Emphasize recall (fewer false negatives)
        eps (float, optional): Small constant to avoid division by zero.
            Default: 1e-7
        threshold (float, optional): Threshold for binarizing predictions
        ignore_channels (List[int], optional): Channels to exclude
            
    Returns:
        torch.Tensor: F-score (scalar)
        
    Example:
        >>> pred = torch.sigmoid(torch.randn(2, 4, 64, 64, 64))
        >>> target = torch.randint(0, 2, (2, 4, 64, 64, 64)).float()
        >>> 
        >>> # F1-score (Dice coefficient)
        >>> f1 = f_score(pred, target, beta=1, threshold=0.5)
        >>> print(f"F1/Dice: {f1:.4f}")
        >>> 
        >>> # F2-score (emphasize recall)
        >>> f2 = f_score(pred, target, beta=2, threshold=0.5)
        >>> print(f"F2: {f2:.4f}")
        >>> 
        >>> # Ignore background
        >>> dice_fg = f_score(pred, target, beta=1, threshold=0.5, ignore_channels=[0])
        
    Note:
        - When beta=1, equivalent to Dice coefficient
        - More robust to class imbalance than accuracy
        - Commonly used loss function (1 - F1) for segmentation
        - Range [0, 1]: higher is better
        
    Mathematical Formula:
        F_beta = ((1 + beta²) × TP + eps) / ((1 + beta²) × TP + beta² × FN + FP + eps)
    """
    # Apply threshold
    pr = _threshold(pr, threshold=threshold)

    # Select channels
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Calculate true positives, false positives, false negatives
    tp = torch.sum(gt_processed * pr_processed)
    fp = torch.sum(pr_processed) - tp
    fn = torch.sum(gt_processed) - tp

    # Calculate F-score
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    
    return score


def accuracy(
        pr: torch.Tensor, 
        gt: torch.Tensor, 
        threshold: float = 0.5,
        ignore_channels: Optional[List[int]] = None
    ) -> torch.Tensor:
    """
    Calculate pixel/voxel-wise accuracy.
    
    Accuracy measures the proportion of correctly classified pixels:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Range: [0, 1], where 1 is perfect classification
    
    Args:
        pr (torch.Tensor): Predictions (B, C, *spatial)
        gt (torch.Tensor): Ground truth labels (B, C, *spatial)
        threshold (float, optional): Threshold for binarizing predictions.
            Default: 0.5
        ignore_channels (List[int], optional): Channels to exclude
            
    Returns:
        torch.Tensor: Accuracy score (scalar)
        
    Example:
        >>> pred = torch.sigmoid(torch.randn(2, 4, 64, 64, 64))
        >>> target = torch.randint(0, 2, (2, 4, 64, 64, 64)).float()
        >>> 
        >>> acc = accuracy(pred, target, threshold=0.5)
        >>> print(f"Accuracy: {acc:.4f}")
        >>> 
        >>> # Foreground accuracy only
        >>> acc_fg = accuracy(pred, target, threshold=0.5, ignore_channels=[0])
        
    Note:
        - Can be misleading for imbalanced datasets
        - Example: 95% background → 95% accuracy by predicting all background
        - Use Dice/IoU for segmentation tasks with class imbalance
        - Good for balanced multi-class problems
        
    Warning:
        Not recommended as primary metric for medical image segmentation
        due to extreme class imbalance (background >> foreground).
    """
    # Apply threshold
    pr = _threshold(pr, threshold=threshold)

    # Select channels
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Ensure matching dtypes for comparison
    if pr_processed.dtype != gt_processed.dtype:
        gt_processed = gt_processed.type(pr_processed.dtype)
    
    # Count correct predictions
    tp = torch.sum(gt_processed == pr_processed, dtype=pr_processed.dtype)
    
    # Calculate accuracy
    score = tp / gt_processed.view(-1).shape[0]
    
    return score


def precision(
        pr: torch.Tensor, 
        gt: torch.Tensor, 
        eps: float = 1e-7, 
        threshold: Optional[float] = None,
        ignore_channels: Optional[List[int]] = None
    ) -> torch.Tensor:
    """
    Calculate Precision (Positive Predictive Value).
    
    Precision measures the proportion of positive predictions that are correct:
    Precision = TP / (TP + FP)
    
    Range: [0, 1], where 1 means no false positives
    
    Args:
        pr (torch.Tensor): Predictions (B, C, *spatial)
        gt (torch.Tensor): Ground truth labels (B, C, *spatial)
        eps (float, optional): Small constant to avoid division by zero.
            Default: 1e-7
        threshold (float, optional): Threshold for binarizing predictions
        ignore_channels (List[int], optional): Channels to exclude
            
    Returns:
        torch.Tensor: Precision score (scalar)
        
    Example:
        >>> pred = torch.sigmoid(torch.randn(2, 4, 64, 64, 64))
        >>> target = torch.randint(0, 2, (2, 4, 64, 64, 64)).float()
        >>> 
        >>> prec = precision(pred, target, threshold=0.5)
        >>> print(f"Precision: {prec:.4f}")
        
    Note:
        - High precision: Few false positives (conservative predictions)
        - Low precision: Many false positives (over-segmentation)
        - Trade-off with recall (can't maximize both simultaneously)
        - Important when false positives are costly
        
    Use Case:
        Medical imaging: High precision means predicted tumors are likely real
        (fewer false alarms, but might miss some tumors)
    """
    # Apply threshold
    pr = _threshold(pr, threshold=threshold)

    # Select channels
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Calculate TP and FP
    tp = torch.sum(gt_processed * pr_processed)
    fp = torch.sum(pr_processed) - tp

    # Calculate precision
    score = (tp + eps) / (tp + fp + eps)
    
    return score


def recall(
        pr: torch.Tensor, 
        gt: torch.Tensor, 
        eps: float = 1e-7, 
        threshold: Optional[float] = None,
        ignore_channels: Optional[List[int]] = None
    ) -> torch.Tensor:
    """
    Calculate Recall (Sensitivity / True Positive Rate).
    
    Recall measures the proportion of actual positives that are detected:
    Recall = TP / (TP + FN)
    
    Range: [0, 1], where 1 means no false negatives
    
    Args:
        pr (torch.Tensor): Predictions (B, C, *spatial)
        gt (torch.Tensor): Ground truth labels (B, C, *spatial)
        eps (float, optional): Small constant to avoid division by zero.
            Default: 1e-7
        threshold (float, optional): Threshold for binarizing predictions
        ignore_channels (List[int], optional): Channels to exclude
            
    Returns:
        torch.Tensor: Recall score (scalar)
        
    Example:
        >>> pred = torch.sigmoid(torch.randn(2, 4, 64, 64, 64))
        >>> target = torch.randint(0, 2, (2, 4, 64, 64, 64)).float()
        >>> 
        >>> rec = recall(pred, target, threshold=0.5)
        >>> print(f"Recall: {rec:.4f}")
        
    Note:
        - High recall: Few false negatives (captures most positives)
        - Low recall: Many false negatives (under-segmentation)
        - Trade-off with precision
        - Important when false negatives are costly
        
    Use Case:
        Medical imaging: High recall means most tumors are detected
        (might have false alarms, but rarely misses real tumors)
    """
    # Apply threshold
    pr = _threshold(pr, threshold=threshold)

    # Select channels
    pr_processed, gt_processed = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Calculate TP and FN
    tp = torch.sum(gt_processed * pr_processed)
    fn = torch.sum(gt_processed) - tp

    # Calculate recall
    score = (tp + eps) / (tp + fn + eps)

    return score


# ============================================================================
# Loss Functions
# ============================================================================

class JaccardLoss(Loss):
    """
    Jaccard Loss (1 - IoU).
    
    Jaccard loss is the complement of IoU (Intersection over Union).
    Minimizing Jaccard loss is equivalent to maximizing IoU.
    
    Formula:
        Loss = 1 - IoU
        Loss = 1 - (Intersection / Union)
        
    Range: [0, 1], where 0 is perfect segmentation
    
    Advantages:
        ✓ Handles class imbalance well
        ✓ Directly optimizes IoU metric
        ✓ Differentiable approximation to set operations
        ✓ Works well for small objects
        
    Args:
        eps (float, optional): Small constant to avoid division by zero.
            Default: 1.0 (more stable than smaller values during training)
        activation (str, optional): Activation to apply to predictions.
            Options: None, 'sigmoid', 'softmax'. Default: None
        ignore_channels (List[int], optional): Channels to exclude from loss
            
    Example:
        >>> # Binary segmentation with sigmoid
        >>> criterion = JaccardLoss(activation='sigmoid')
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Multi-class with softmax
        >>> criterion = JaccardLoss(activation='softmax')
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Ignore background (channel 0)
        >>> criterion = JaccardLoss(ignore_channels=[0])
        >>> loss = criterion(predictions, targets)
        
    Note:
        - Use eps=1.0 for training stability
        - Activation should match your task (sigmoid/softmax)
        - Can be combined with other losses (e.g., + CrossEntropy)
        - Gradient-friendly: smooth approximation to discrete IoU
    """

    def __init__(self, 
                 eps: float = 1.0, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate Jaccard loss.
        
        Args:
            y_pr (torch.Tensor): Predictions (B, C, *spatial)
            y_gt (torch.Tensor): Ground truth (B, C, *spatial)
            
        Returns:
            torch.Tensor: Loss value (scalar)
        """
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
    Dice Loss (1 - Dice Coefficient).
    
    Dice loss is the complement of Dice coefficient (F1-score).
    Widely used in medical image segmentation due to strong performance
    on imbalanced datasets.
    
    Formula:
        Loss = 1 - Dice
        Loss = 1 - (2 × Intersection) / (Pred + GT)
        Loss = 1 - (2 × TP) / (2 × TP + FP + FN)
        
    Range: [0, 1], where 0 is perfect segmentation
    
    Advantages:
        ✓ Handles severe class imbalance
        ✓ Directly optimizes Dice metric (common in challenges)
        ✓ Smooth gradients
        ✓ Works better than BCE for small objects
        ✓ State-of-the-art for medical segmentation
        
    Args:
        eps (float, optional): Small constant to avoid division by zero.
            Default: 1.0 (recommended for stability)
        beta (float, optional): Weight of recall in F-score. Default: 1.0
            Usually keep at 1.0 for standard Dice
        activation (str, optional): Activation to apply to predictions.
            Options: None, 'sigmoid', 'softmax'. Default: None
        ignore_channels (List[int], optional): Channels to exclude from loss
            
    Example:
        >>> # Standard Dice loss for multi-class segmentation
        >>> criterion = DiceLoss()
        >>> loss = criterion(logits, targets)
        >>> 
        >>> # Binary segmentation with sigmoid
        >>> criterion = DiceLoss(activation='sigmoid')
        >>> loss = criterion(logits, targets)
        >>> 
        >>> # Multi-class with softmax
        >>> criterion = DiceLoss(activation='softmax')
        >>> loss = criterion(logits, targets)
        >>> 
        >>> # For CLCUNet (uses sigmoid per channel)
        >>> criterion = DiceLoss(activation='sigmoid')
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Combined loss (common practice)
        >>> criterion = DiceLoss() + 0.5 * nn.CrossEntropyLoss()
        >>> loss = criterion(predictions, targets)
        
    Training Tips:
        - Start with DiceLoss alone
        - If convergence is slow, try Dice + 0.5 × CrossEntropy
        - Use activation='softmax' for standard multi-class
        - Use activation='sigmoid' for multi-label or CLCUNet
        - eps=1.0 is more stable than eps=1e-7 during early training
        
    Note:
        - More stable than pure IoU/Jaccard loss
        - Commonly used in medical imaging competitions
        - Can be combined with CrossEntropy for faster convergence
        - Gradient magnitude can be small when Dice is high (near convergence)
    """

    def __init__(self, 
                 eps: float = 1.0, 
                 beta: float = 1.0, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            y_pr (torch.Tensor): Predictions (B, C, *spatial)
                Can be logits or probabilities depending on activation
            y_gt (torch.Tensor): Ground truth (B, C, *spatial)
                Binary or one-hot encoded
            
        Returns:
            torch.Tensor: Loss value (scalar)
        """
        y_pr_activated = self.activation(y_pr)
        return 1.0 - f_score(
            y_pr_activated,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


# ============================================================================
# PyTorch Built-in Losses (with Loss base class)
# ============================================================================

class L1Loss(nn.L1Loss, Loss):
    """
    L1 Loss (Mean Absolute Error) with Loss arithmetic support.
    
    Inherits from both nn.L1Loss and Loss for operator overloading.
    """
    pass


class MSELoss(nn.MSELoss, Loss):
    """
    Mean Squared Error Loss with Loss arithmetic support.
    
    Inherits from both nn.MSELoss and Loss for operator overloading.
    """
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, Loss):
    """
    Cross-Entropy Loss with Loss arithmetic support.
    
    Standard loss for multi-class classification. Combines LogSoftmax
    and NLLLoss in a single class for numerical stability.
    
    Note:
        - Expects raw logits (no softmax)
        - Targets should be class indices (not one-hot)
        - Commonly combined with Dice: DiceLoss() + 0.5 * CrossEntropyLoss()
    """
    pass


class NLLLoss(nn.NLLLoss, Loss):
    """
    Negative Log-Likelihood Loss with Loss arithmetic support.
    
    Requires log-probabilities as input (use after LogSoftmax).
    """
    pass


class BCELoss(nn.BCELoss, Loss):
    """
    Binary Cross-Entropy Loss with Loss arithmetic support.
    
    For binary segmentation. Expects probabilities [0, 1] as input.
    
    Note:
        - Apply sigmoid to logits before using BCELoss
        - Or use BCEWithLogitsLoss for numerical stability
    """
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Loss):
    """
    Binary Cross-Entropy with Logits Loss with Loss arithmetic support.
    
    More numerically stable than BCELoss. Combines sigmoid and BCE.
    
    Note:
        - Expects raw logits (no sigmoid)
        - Preferred over BCELoss for stability
    """
    pass


# ============================================================================
# Metric Classes
# ============================================================================

class IoU(Metric):
    """
    Intersection over Union (IoU / Jaccard Index) Metric.
    
    Measures overlap between prediction and ground truth.
    
    Args:
        eps (float, optional): Small constant for numerical stability.
            Default: 1e-7
        threshold (float, optional): Threshold for binarizing predictions.
            Default: 0.5. Set to None for softmax outputs.
        activation (str, optional): Activation to apply. Default: None
        ignore_channels (List[int], optional): Channels to exclude
        **kwargs: Additional arguments (including custom name)
        
    Example:
        >>> iou_metric = IoU(threshold=0.5)
        >>> score = iou_metric(predictions, targets)
        >>> print(f"IoU: {score:.4f}")
        
    Note:
        - __name__ is set to 'iou_score' by default
        - Can override with name='custom_iou' in constructor
    """

    __name__ = "iou_score"

    def __init__(self, 
                 eps: float = 1e-7, 
                 threshold: float = 0.5, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'iou_score')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU metric.
        
        Args:
            y_pr (torch.Tensor): Predictions
            y_gt (torch.Tensor): Ground truth
            
        Returns:
            torch.Tensor: IoU score
        """
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
    F-score (Dice Coefficient when beta=1) Metric.
    
    Args:
        beta (float, optional): Weight of recall. Default: 1
        eps (float, optional): Small constant. Default: 1e-7
        threshold (float, optional): Binarization threshold. Default: 0.5
        activation (str, optional): Activation function. Default: None
        ignore_channels (List[int], optional): Channels to exclude
        **kwargs: Additional arguments
        
    Example:
        >>> dice_metric = Fscore(beta=1, threshold=0.5)
        >>> score = dice_metric(predictions, targets)
        >>> print(f"Dice: {score:.4f}")
    """

    __name__ = 'fscore'
    
    def __init__(self, beta: float = 1, 
                 eps: float = 1e-7, 
                 threshold: float = 0.5,
                 activation: Optional[str] = None, 
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'fscore')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Calculate F-score."""
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
    Accuracy Metric (proportion of correct predictions).
    
    Note: Not recommended as primary metric for imbalanced segmentation.
    """

    __name__ = 'accuracy'
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'accuracy')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Calculate accuracy."""
        y_pr_activated = self.activation(y_pr)
        return accuracy(
            y_pr_activated,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(Metric):
    """
    Recall (Sensitivity) Metric.
    
    Measures proportion of actual positives correctly identified.
    """

    __name__ = 'recall'
    
    def __init__(self, 
                 eps: float = 1e-7, 
                 threshold: float = 0.5, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'recall')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Calculate recall."""
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
    Precision (Positive Predictive Value) Metric.
    
    Measures proportion of positive predictions that are correct.
    """

    __name__ = 'precision'
    
    def __init__(self, 
                 eps: float = 1e-7, 
                 threshold: float = 0.5, 
                 activation: Optional[str] = None,
                 ignore_channels: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.name = kwargs.get('name', 'precision')

    def forward(self, y_pr: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Calculate precision."""
        y_pr_activated = self.activation(y_pr)
        return precision(
            y_pr_activated,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
