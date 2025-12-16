"""
Training Loop Infrastructure for 3D Medical Image Segmentation.

This module provides the core training and validation loop infrastructure using
an epoch-based approach with metric tracking. It includes utilities for monitoring
training progress, computing running statistics, and managing train/validation phases.

Classes:
    Meter: Abstract base class for metric tracking
    AverageValueMeter: Tracks running mean and standard deviation of metrics
    Epoch: Base class for training/validation epochs with device management
    TrainEpoch: Handles forward pass, backpropagation, and optimizer steps
    ValidEpoch: Handles validation without gradient computation
    
Key Features:
    - Real-time metric tracking with tqdm progress bars
    - Automatic device management (CPU/CUDA)
    - Online statistics computation (mean, std)
    - Flexible metric system (loss + custom metrics)
    - Separate train/validation phases
"""
    
import sys
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Union


class Meter(object):
    """
        Abstract base class for online metric computation.
    
    Meters provide a standardized interface for tracking statistics during training.
    They accumulate values across batches and compute running statistics without
    storing all individual values in memory.
    
    All concrete meter implementations should override:
        - reset(): Initialize/reset internal state
        - add(): Incorporate a new value
        - value(): Return current metric value(s)
    """

    def reset(self) -> None:
        """
        Reset the meter to its initial state.
        
        Called at the beginning of each epoch to clear previous statistics.
        """
        pass

    def add(self, value: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """
        Add a new value to the meter.
        
        Args:
            value: The value to incorporate. Can be:
                - float: Single scalar value
                - np.ndarray: NumPy array (mean will be computed)
                - torch.Tensor: PyTorch tensor (mean will be computed)
            n: Number of samples this value represents (for weighted averaging)
        """
        pass

    def value(self) -> Tuple[float, float]:
        """
        Get the current value of the meter.
        
        Returns:
            Tuple[float, float]: (mean, standard_deviation)
        """
        pass


class AverageValueMeter(Meter):
    """
    Computes and stores the running mean and standard deviation.
    
    This meter uses Welford's online algorithm to compute variance in a
    numerically stable way. It's ideal for tracking loss and metric values
    across batches without storing all individual values.
    
    Algorithm:
        Uses a single-pass algorithm for computing variance:
        - Update mean incrementally: μ_new = μ_old + (x - μ_old) / n
        - Update variance using: M_n = M_(n-1) + (x - μ_old)(x - μ_new)
        - Variance = M_n / (n - 1)
    
    Attributes:
        val (float): Most recently added value
        n (int): Total number of values added
        sum (float): Sum of all values
        mean (float): Running mean
        std (float): Running standard deviation
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val: float = 0.0

    def add(self, value: Union[float, np.ndarray, torch.Tensor], n: int = 1) -> None:
        """
        Add a new value and update running statistics.
        
        Args:
            value: Value to add. If array/tensor, uses mean.
            n: Number of samples this value represents (default: 1)
            
        Note:
            - Converts tensors/arrays to scalar by taking mean
            - Updates mean and variance using Welford's algorithm
        """
        # Convert various input types to scalar float
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.detach().cpu().numpy().mean()
        elif isinstance(value, np.ndarray):
            value = value.mean()
        
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        # Compute statistics based on number of samples
        if self.n == 0:
            self.mean: float = np.nan
            self.std: float = np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # Force copy to avoid reference issues
            self.std = float('inf')
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            # Welford's online algorithm for mean and variance
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self) -> Tuple[float, float]:
        """
        Get current mean and standard deviation.
        
        Returns:
            Tuple[float, float]: (mean, std)
        """
        return self.mean, self.std

    def reset(self) -> None:
        """
        Reset all statistics to initial state.
        
        Called at the beginning of each epoch.
        """
        self.n: int = 0
        self.sum: float = 0.0
        self.var: float = 0.0
        self.val: float = 0.0
        self.mean: float = np.nan
        self.mean_old: float = 0.0
        self.m_s: float = 0.0
        self.std: float = np.nan


# ============================================================================
# Epoch Management
# ============================================================================

class Epoch:
    """
    Base class for training and validation epochs.
    
    This abstract class provides the common infrastructure for running one epoch
    of training or validation. It handles:
    - Device management (moving model/loss/metrics to CPU/GPU)
    - Progress bar visualization with tqdm
    - Metric tracking across batches
    - Logging and formatting
    
    Subclasses must implement:
        - on_epoch_start(): Setup before epoch (e.g., model.train() or model.eval())
        - batch_update(): Process one batch (forward, backward, optimizer step)
    
    Args:
        model (nn.Module): Neural network model
        loss (nn.Module): Loss function
        metrics (List[nn.Module]): List of metric functions to compute
        stage_name (str): Name for display ('train' or 'valid')
        device (str): Device to run on ('cpu' or 'cuda')
        verbose (bool): Whether to show progress bar
        
    Attributes:
        model: The neural network model
        loss: Loss function
        metrics: List of metric functions
        stage_name: Display name for this epoch type
        device: Computation device
        verbose: Progress bar visibility flag
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
        """Initialize epoch with model, loss, metrics, and configuration."""
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self) -> None:
        """
        Move model, loss function, and metrics to the specified device.
        
        This ensures all computations happen on the same device (CPU or CUDA).
        Called automatically during initialization.
        """
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs: Dict[str, float]) -> str:
        """
        Format log dictionary into a human-readable string.
        
        Args:
            logs: Dictionary of metric names and values
            
        Returns:
            str: Formatted string like "loss - 0.234, iou - 0.856"
        """
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one batch of data (abstract method).
        
        Must be implemented by subclasses to define how to:
        - Forward pass the data through the model
        - Compute loss
        - Optionally: backward pass and optimizer step (for training)
        
        Args:
            x: Input tensor (images)
            y: Target tensor (masks)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, predictions)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def on_epoch_start(self) -> None:
        """
        Hook called at the beginning of each epoch.
        
        Subclasses should override this to set model mode:
        - Training: model.train()
        - Validation: model.eval()
        """
        pass

    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Execute one complete epoch over the dataloader.
        
        This is the main method that:
        1. Calls on_epoch_start() to prepare
        2. Iterates over all batches in the dataloader
        3. For each batch:
            - Moves data to device
            - Calls batch_update() for processing
            - Updates loss and metric meters
            - Updates progress bar
        4. Returns final averaged metrics
        
        Args:
            dataloader: PyTorch DataLoader with (images, masks) batches
            
        Returns:
            Dict[str, float]: Dictionary of metric names and their mean values
                Example: {'dice_loss': 0.234, 'iou_score': 0.856}
        """
        self.on_epoch_start()

        logs: Dict[str, float] = {}
        loss_meter = AverageValueMeter()
        metrics_meters: Dict[str, AverageValueMeter] = {
            metric.name: AverageValueMeter() for metric in self.metrics
        }

        # Process all batches with progress bar
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                # Move data to device and ensure correct dtype
                x, y = x.to(self.device), y.to(self.device)
                y = y.long()

                # Process batch (forward, backward, metrics)
                loss, y_pred = self.batch_update(x, y)

                # Update loss meter
                loss_value: float = loss.cpu().detach().numpy().mean()
                loss_meter.add(loss_value)
                loss_logs: Dict[str, float] = {f"{self.loss.__name__}": loss_meter.mean}
                logs.update(loss_logs)

                # Update metric meters
                for metric_fn in self.metrics:
                    metric_value: float = metric_fn(y_pred, y).cpu().detach().numpy().mean()
                    metrics_meters[metric_fn.name].add(metric_value)
                metrics_logs: Dict[str, float] = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                # Update progress bar
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    """
    Training epoch with gradient computation and optimizer updates.
    
    This class handles one epoch of training, including:
    - Forward pass through the model
    - Loss computation
    - Backward pass (gradient computation)
    - Optimizer step (weight updates)
    - Optional learning rate scheduling
    
    The model is set to training mode (model.train()) which enables:
    - Dropout layers
    - Batch normalization updates
    - Gradient computation
    
    Args:
        model (nn.Module): Neural network to train
        loss (nn.Module): Loss function (e.g., DiceLoss, CrossEntropyLoss)
        metrics (List[nn.Module]): Metrics to track (e.g., IoU, Accuracy)
        optimizer (Optimizer): PyTorch optimizer (e.g., Adam, SGD)
        scheduler (optional): Learning rate scheduler (e.g., StepLR)
        device (str): Computation device ('cpu' or 'cuda')
        verbose (bool): Show progress bar
        
    Attributes:
        optimizer: PyTorch optimizer for weight updates
        scheduler: Optional learning rate scheduler

    Note:
        - Gradients are zeroed before each batch (optimizer.zero_grad())
        - Loss.backward() computes gradients
        - optimizer.step() updates weights
        - scheduler.step() is called after optimizer if scheduler exists
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: List[nn.Module],
        optimizer: Optimizer,
        scheduler=None,
        device: str = "cpu",
        verbose: bool = True
    ):
        """Initialize training epoch with model, loss, metrics, and optimizer."""
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self) -> None:
        """
        Establece el modelo en modo de entrenamiento.Set model to training mode.
        
        This enables:
        - Gradient computation
        - Dropout (if present)
        - Batch normalization training mode
        """
        self.model.train()

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one training step on a batch.
        
        Training step sequence:
        1. Zero gradients from previous step
        2. Forward pass through model
        3. Compute loss
        4. Backward pass (compute gradients)
        5. Update weights with optimizer
        6. Optional: Update learning rate with scheduler
        
        Args:
            x (torch.Tensor): Input images, shape (B, C, D, H, W)
            y (torch.Tensor): Target masks, shape (B, 1, D, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, predictions)
                - loss: Scalar loss value
                - predictions: Model output, shape (B, num_classes, D, H, W)
        """
        # Zero gradients from previous iteration
        self.optimizer.zero_grad()

        # Forward pass
        prediction = self.model.forward(x.float())
        loss = self.loss(prediction, y)

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

        # Optional: Update learning rate
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        return loss, prediction


class ValidEpoch(Epoch):
    """
    Validation epoch without gradient computation.
    
    This class handles one epoch of validation/evaluation, including:
    - Forward pass through the model
    - Loss and metric computation
    - No gradient computation (torch.no_grad())
    - No weight updates
    
    The model is set to evaluation mode (model.eval()) which:
    - Disables dropout layers
    - Uses running statistics for batch normalization
    - Disables gradient computation for efficiency
    
    Args:
        model (nn.Module): Neural network to evaluate
        loss (nn.Module): Loss function
        metrics (List[nn.Module]): Metrics to track
        device (str): Computation device ('cpu' or 'cuda')
        verbose (bool): Show progress bar
    
    Note:
        - Uses torch.no_grad() context for efficiency
        - No gradients are computed or stored
        - Model parameters are not updated
        - Faster than training due to no backward pass
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: List[nn.Module],
        device: str = "cpu",
        verbose: bool = True
    ):
        """Initialize validation epoch with model, loss, and metrics."""
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
        Set model to evaluation mode.
        
        This disables:
        - Gradient computation
        - Dropout
        - Batch normalization training mode
        """
        self.model.eval()

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one validation step on a batch.
        
        Validation step sequence:
        1. Forward pass through model (no gradient computation)
        2. Compute loss
        3. Return loss and predictions
        
        Args:
            x (torch.Tensor): Input images, shape (B, C, D, H, W)
            y (torch.Tensor): Target masks, shape (B, 1, D, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, predictions)
                - loss: Scalar loss value
                - predictions: Model output, shape (B, num_classes, D, H, W)
                
        Note:
            - Wrapped in torch.no_grad() for memory efficiency
            - No backward pass or weight updates
            - Faster than training batch_update
        """
        # Disable gradient computation for efficiency
        with torch.no_grad():
            prediction = self.model.forward(x.float())
            loss = self.loss(prediction, y)

        return loss, prediction
      
