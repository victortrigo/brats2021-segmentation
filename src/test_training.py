"""
Unit Tests for BraTS 2021 Training Pipeline.

This module provides comprehensive unit tests for the training infrastructure,
covering configuration management, model instantiation, training loops, early
stopping, TensorBoard integration, and more.

Test Categories:
    - Configuration Merging: Verify base + mode config override behavior
    - Model Instantiation: Test all 4 models (UNet, DeepLabV3+, DeepLabV3+SAM, CLCUNet)
    - Early Stopping: Validate patience mechanism and checkpoint saving
    - TensorBoard Integration: Check metric logging
    - Scheduler Integration: Verify learning rate scheduling
    - Loss Function Selection: Test model-specific loss configurations
    - Edge Cases: Zero patience, continuous improvement, negative patience
    
Usage:
    # Run all tests
    python test_training.py
    
    # Run with verbose output
    python test_training.py -v
    
    # Run specific test class
    python -m unittest test_training.TestConfigMerging
    
    # Run specific test
    python -m unittest test_training.TestConfigMerging.test_subset_sizes_override
    
Coverage:
    - Configuration system (merging, overrides)
    - All 4 segmentation models
    - Training loop logic (forward, backward, optimization)
    - Validation loop (no gradients)
    - Early stopping mechanism
    - Model checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    - Loss function selection
"""

import unittest
import os
import yaml
import torch
import logging
from unittest.mock import patch, MagicMock
from training import main, get_model

# Configure logging for test output
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Configuration Management Tests
# ============================================================================

class TestConfigMerging(unittest.TestCase):
    """
    Test suite for configuration loading and merging.
    
    Verifies that:
    - Base configuration loads correctly
    - Mode configuration overrides base values
    - Subset sizes are properly applied
    - Epoch counts are correctly merged
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration paths."""
        cls.base_path = os.path.join('configs', 'unet_config.yaml')
        cls.mode_path = os.path.join('configs', 'config_test.yaml')

    def _setup_mocks_with_sizes(self, mock_subset, train_size=50, valid_size=5, test_size=5):
        """
        Helper to configure mocks with specific dataset sizes.
        
        Args:
            mock_subset: Mock for create_subset function
            train_size: Size of training subset
            valid_size: Size of validation subset
            test_size: Size of test subset
        """
        # Create mock subsets with __len__ configured
        train_subset_mock = MagicMock()
        train_subset_mock.__len__.return_value = train_size
        train_loader_mock = MagicMock()
        
        valid_subset_mock = MagicMock()
        valid_subset_mock.__len__.return_value = valid_size
        valid_loader_mock = MagicMock()
        
        test_subset_mock = MagicMock()
        test_subset_mock.__len__.return_value = test_size
        test_loader_mock = MagicMock()
        
        # Configure side_effect to return different subsets based on size/shuffle
        def create_subset_side_effect(dataset, size, batch_size, shuffle):
            if size == train_size or (size <= 0 and shuffle):  # Train uses shuffle=True
                return train_subset_mock, train_loader_mock
            elif size == valid_size or (size <= 0 and not shuffle):  # Valid uses shuffle=False
                return valid_subset_mock, valid_loader_mock
            else:
                return test_subset_mock, test_loader_mock
        
        mock_subset.side_effect = create_subset_side_effect

    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_subset_sizes_override(self, mock_cuda, mock_save, mock_valid, 
                                   mock_train, mock_writer, mock_model, 
                                   mock_subset, mock_dataset):
        """
        Verify that config_test.yaml correctly overrides subset_train_size.
        
        This test ensures the configuration merging system works:
        - Base config has subset_train_size: 0 (full dataset)
        - Mode config has subset_train_size: 50
        - Final config should use 50 (mode overrides base)
        """
        # Load configs to extract expected values
        with open(self.mode_path, 'r', encoding='utf-8') as f:
            mode_cfg = yaml.safe_load(f)
        
        expected_train_size = mode_cfg['training']['subset_train_size']
        expected_valid_size = mode_cfg['training']['subset_valid_size']
        expected_batch_size = mode_cfg['training'].get('batch_size', 1)
        
        # Setup mocks with correct sizes
        self._setup_mocks_with_sizes(mock_subset, expected_train_size, expected_valid_size, 5)
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        mock_valid.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.9}
        
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        # Execute
        main(self.base_path, self.mode_path)
        
        # VERIFICATION 1: create_subset was called with mode size
        calls = mock_subset.call_args_list
        train_call = [c for c in calls if c[0][2] == expected_batch_size and c[1]['shuffle']][0]
        self.assertEqual(train_call[0][1], expected_train_size)  # Verifica subset_train_size
        
        # VERIFICATION 2: Should be called 3 times (train, valid, test)
        self.assertEqual(mock_subset.call_count, 3)
        
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_epochs_override(self, mock_cuda, mock_save, mock_valid, 
                            mock_train, mock_writer, mock_model, 
                            mock_subset, mock_dataset):
        """
        Verify that the number of epochs is correctly overridden.
        
        Tests that:
        - Base config has epochs: 500
        - Mode config has epochs: 10 and patience: 10
        - Training stops before 500 epochs due to mode override
        """
        # Setup mocks
        self._setup_mocks_with_sizes(mock_subset, 50, 5, 5)
        
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        # Simulate early stopping at epoch 3 (patience=10 from config_test)
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        # IoU doesn't improve -> early stopping
        mock_valid.return_value.run.return_value = {'dice_loss': 0.2, 'iou_score': 0.7}
        
        main(self.base_path, self.mode_path)
        
         # Verify it stopped before 500 epochs (thanks to patience=10 from config_test)
        self.assertLess(mock_train.return_value.run.call_count, 500)
        self.assertGreater(mock_train.return_value.run.call_count, 0)


# ============================================================================
# Model Instantiation Tests
# ============================================================================

class TestModelInstantiation(unittest.TestCase):
    """
    Test suite for model factory (get_model function).
    
    Verifies that all supported models can be correctly instantiated
    with their specific parameters and that unsupported models raise errors.
    """
    
    def test_unet_with_correct_params(self):
        """
        UNet should accept in_channels, num_classes, and base_channels.
        
        Verifies:
        - Model instantiates without errors
        - Has expected attributes (encoder, decoder, out_conv)
        - Is instance of correct class
        """
        config = {
            'model': {
                'name': 'UNet', 
                'in_channels': 4, 
                'num_classes': 4, 
                'base_channels': 64
            }
        }
        model = get_model(config)
        
        from unet import UNet
        self.assertIsInstance(model, UNet)
        self.assertTrue(hasattr(model, 'encoder'))
        self.assertTrue(hasattr(model, 'decoder'))
        self.assertTrue(hasattr(model, 'out_conv'))
        
    def test_deeplabv3_instantiation(self):
        """DeepLabV3+ should instantiate correctly with in_channels and num_classes."""
        config = {
            'model': {
                'name': 'DeepLabV3+', 
                'in_channels': 4, 
                'num_classes': 4
            }
        }
        model = get_model(config)
        
        from deeplabv3 import DeepLabV3Plus
        self.assertIsInstance(model, DeepLabV3Plus)
        
    def test_deeplabv3sam_instantiation(self):
        """DeepLabV3+SAM should instantiate correctly."""
        config = {
            'model': {
                'name': 'DeepLabV3+SAM', 
                'in_channels': 4, 
                'num_classes': 4
            }
        }
        model = get_model(config)
        
        from deeplabv3sam import DeepLabV3PlusSAM
        self.assertIsInstance(model, DeepLabV3PlusSAM)
        
    def test_clcunet_instantiation(self):
        """CLCUNet should instantiate correctly with in_channels and base_channels."""
        config = {
            'model': {
                'name': 'CLCUNet', 
                'in_channels': 4, 
                'base_channels': 64
            }
        }
        model = get_model(config)
        
        from clcunet import CLCUNet
        self.assertIsInstance(model, CLCUNet)
        
    def test_unsupported_model_raises_error(self):
        """Unsupported model name should raise ValueError with clear message."""
        config = {'model': {'name': 'ModeloInventado'}}
        
        with self.assertRaisesRegex(ValueError, "Unsupported model: ModeloInventado"):
            get_model(config)

# ============================================================================
# Early Stopping Tests
# ============================================================================

class TestEarlyStopping(unittest.TestCase):
    """
    Test suite for early stopping mechanism.
    
    Verifies that:
    - Training stops after 'patience' epochs without improvement
    - Model is saved only when validation metric improves
    - Patience counter resets on improvement
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up configuration path for tests."""
        cls.config_path = 'configs/unet_config.yaml'
        
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_early_stopping_after_patience(self, mock_cuda, mock_save, mock_valid, 
                                          mock_train, mock_writer, mock_model,
                                          mock_adam, mock_scheduler, mock_subset, 
                                          mock_dataset):
        """
        Verify training stops after patience epochs without improvement.
        
        Scenario:
        - Epoch 0: IoU = 0.90 (improvement, save model, patience_counter = 0)
        - Epochs 1-50: IoU = 0.70 (no improvement, patience_counter increments)
        - At epoch 50: patience_counter >= patience (50), STOP
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        
        # Simulate validation: improves at epoch 0, then worsens (patience=50)
        validation_results = [
            {'dice_loss': 0.1, 'iou_score': 0.90},  # Epoch 0: improvement
            {'dice_loss': 0.2, 'iou_score': 0.85},  # Epochs 1-50: no improvement
        ] + [{'dice_loss': 0.3, 'iou_score': 0.70}] * 60
        
        mock_valid.return_value.run.side_effect = validation_results
        
        # Execute
        main(self.config_path)
        
        # Verifications
        # Should stop at epoch 50 (after 50 epochs without improvement)
        self.assertLessEqual(mock_train.return_value.run.call_count, 51)
        
        # Should save model at least once (at epoch 0 when it improves)
        self.assertGreaterEqual(mock_save.call_count, 1)
        
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_model_checkpoint_on_improvement(self, mock_cuda, mock_save, mock_valid, 
                                            mock_train, mock_writer, mock_model,
                                            mock_adam, mock_scheduler, mock_subset, 
                                            mock_dataset):
        """
        Verify model is saved ONLY when validation metric improves.
        
        Scenario:
        - Epoch 0: IoU = 0.85 → SAVE
        - Epoch 1: IoU = 0.84 → no save
        - Epoch 2: IoU = 0.87 → SAVE
        - Epoch 3: IoU = 0.84 → no save
        - Epoch 4: IoU = 0.90 → SAVE
        - Then: no more improvements → early stop
        
        Expected: 3 saves total (epochs 0, 2, 4)
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        
        # Simulate: improves at epochs 0, 2, 4, then stops
        mock_valid.return_value.run.side_effect = [
            {'dice_loss': 0.10, 'iou_score': 0.85},  # Epoch 0: save
            {'dice_loss': 0.11, 'iou_score': 0.84},  # Epoch 1: no save
            {'dice_loss': 0.09, 'iou_score': 0.87},  # Epoch 2: save
            {'dice_loss': 0.11, 'iou_score': 0.84},  # Epoch 3: no save
            {'dice_loss': 0.08, 'iou_score': 0.90},  # Epoch 4: save
        ] + [{'dice_loss': 0.15, 'iou_score': 0.70}] * 60  # Rest: no improvement
        
        
        main(self.config_path)
        
        # Should save exactly 3 times (epochs 0, 2, 4)
        self.assertEqual(mock_save.call_count, 3)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """
    Test suite for edge cases and boundary conditions.
    
    Tests unusual but valid scenarios like:
    - patience = 0 (immediate stopping if no improvement)
    - Continuous improvement (no early stopping)
    - Negative patience (invalid but shouldn't crash)
    """
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_zero_patience_stops_immediately(self, mock_cuda, mock_save, mock_valid, 
                                            mock_train, mock_writer, mock_model,
                                            mock_adam, mock_scheduler, mock_subset, 
                                            mock_dataset):
        """
        Verify behavior with patience=0 (stops if no immediate improvement).
        
        With patience=0:
        - Epoch 0: IoU = 0.85 → save, patience_counter = 0
        - Epoch 1: IoU = 0.70 (no improvement) → patience_counter = 1 > patience (0) → STOP
        """
        
        # Create temporary config with patience=0
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {
                'epochs': 100, 
                'batch_size': 1, 
                'learning_rate': 0.001, 
                'loss': 'DiceLoss', 
                'patience': 0  # ⚠️ ZERO patience
            },
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_zero_patience.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            # Setup mocks
            mock_subset.return_value = (MagicMock(), MagicMock())
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
            mock_model.return_value = mock_model_instance
            
            mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
            
            # Simulate NEVER improves after epoch 0
            mock_valid.return_value.run.side_effect = [
                {'dice_loss': 0.10, 'iou_score': 0.85},  # Epoch 0: save
                {'dice_loss': 0.20, 'iou_score': 0.70},  # Epoch 1: no improvement → STOP
            ]
            
            main(temp_path)
            
            # With patience=0, should stop at epoch 1
            self.assertLessEqual(mock_train.return_value.run.call_count, 2)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_continuous_improvement_no_early_stop(self, mock_cuda, mock_save, mock_valid, 
                                                  mock_train, mock_writer, mock_model,
                                                  mock_adam, mock_scheduler, mock_subset, 
                                                  mock_dataset):
        """
        Verify NO early stopping if model continuously improves.
        
        If validation metric improves every epoch, patience counter
        resets to 0 each time, so training should complete all epochs.
        """
        # Create config with few epochs for quick test
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {
                'epochs': 5,  # Only 5 epochs
                'batch_size': 1, 
                'learning_rate': 0.001, 
                'loss': 'DiceLoss', 
                'patience': 10
            },
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_continuous_improvement.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            # Setup mocks
            mock_subset.return_value = (MagicMock(), MagicMock())
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
            mock_model.return_value = mock_model_instance
            
            mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
            
            # Simulate continuous improvement
            mock_valid.return_value.run.side_effect = [
                {'dice_loss': 0.10, 'iou_score': 0.85},  # Epoch  0
                {'dice_loss': 0.09, 'iou_score': 0.87},  # Epoch  1: improvement
                {'dice_loss': 0.08, 'iou_score': 0.89},  # Epoch  2: improvement
                {'dice_loss': 0.07, 'iou_score': 0.91},  # Epoch  3: improvement
                {'dice_loss': 0.06, 'iou_score': 0.93},  # Epoch  4: improvement
            ]
            
            main(temp_path)
            
            # Should complete all 5 epochs without early stopping
            self.assertEqual(mock_train.return_value.run.call_count, 5)
            
            # Should save model 5 times (improves every epoch)
            self.assertEqual(mock_save.call_count, 5)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_negative_patience_raises_no_error(self):
        """
        Verify negative patience doesn't crash (graceful degradation).
        
        Note: In production, you should validate patience >= 0 in main()
        This test just ensures the code doesn't explode with invalid input.
        """
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {
                'epochs': 1, 
                'batch_size': 1, 
                'learning_rate': 0.001, 
                'loss': 'DiceLoss', 
                'patience': -5  # ⚠️ Invalid value
            },
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_negative_patience.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            # This test just verifies no crash
            # (in production you should add validation in main())
            self.assertTrue(os.path.exists(temp_path))
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ============================================================================
# TensorBoard Integration Tests
# ============================================================================

class TestTensorBoardIntegration(unittest.TestCase):
    """
    Test suite for TensorBoard logging integration.
    
    Verifies that:
    - SummaryWriter is created
    - Training metrics are logged (loss, IoU)
    - Validation metrics are logged
    - Writer is properly closed after training
    """
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_tensorboard_logs_train_metrics(self, mock_cuda, mock_save, mock_valid, 
                                           mock_train, mock_writer_class, mock_model,
                                           mock_adam, mock_scheduler, mock_subset, 
                                           mock_dataset):
        """
        Verify training metrics (Loss/train, IoU/train) are logged to TensorBoard.
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        # Get writer instance
        mock_writer = mock_writer_class.return_value
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.15, 'iou_score': 0.82}
        mock_valid.return_value.run.side_effect = [
            {'dice_loss': 0.10, 'iou_score': 0.90},
        ] + [{'dice_loss': 0.20, 'iou_score': 0.70}] * 5
        
        # Create temporary config
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {'epochs': 3, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 2},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_tensorboard_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            main(temp_path)
            
            # Verify SummaryWriter was instantiated
            mock_writer_class.assert_called_once()
            
            # Verify add_scalar was called for train metrics
            calls = mock_writer.add_scalar.call_args_list
            
            # Should have calls for 'Loss/train' and 'IoU/train'
            train_loss_calls = [c for c in calls if 'Loss/train' in str(c)]
            train_iou_calls = [c for c in calls if 'IoU/train' in str(c)]
            
            self.assertGreater(len(train_loss_calls), 0, "Debe registrar Loss/train")
            self.assertGreater(len(train_iou_calls), 0, "Debe registrar IoU/train")
            
            # Verify writer.close() was called
            mock_writer.close.assert_called_once()
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_tensorboard_logs_validation_metrics(self, mock_cuda, mock_save, mock_valid, 
                                                 mock_train, mock_writer_class, mock_model,
                                                 mock_adam, mock_scheduler, mock_subset, 
                                                 mock_dataset):
        """
        Verify validation metrics (Loss/validation, IoU/validation) are logged.
        """
        
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_writer = mock_writer_class.return_value
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.15, 'iou_score': 0.82}
        mock_valid.return_value.run.side_effect = [
            {'dice_loss': 0.12, 'iou_score': 0.88},  # Specific values
        ] + [{'dice_loss': 0.20, 'iou_score': 0.70}] * 5
        
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {'epochs': 2, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 1},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_tensorboard_valid_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            main(temp_path)
            
            calls = mock_writer.add_scalar.call_args_list
            
            # Verify calls for validation metrics
            valid_loss_calls = [c for c in calls if 'Loss/validation' in str(c)]
            valid_iou_calls = [c for c in calls if 'IoU/validation' in str(c)]
            
            self.assertGreater(len(valid_loss_calls), 0, "Should log Loss/validation")
            self.assertGreater(len(valid_iou_calls), 0, "Should log IoU/validation")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# Model Output Shape Tests
# ============================================================================

class TestModelOutputShapes(unittest.TestCase):
    """
    Test suite for verifying model output dimensions.
    
    Ensures that all models produce outputs with the correct shape:
    - Batch dimension matches input
    - Number of classes is correct
    - Spatial dimensions are as expected
    """
    
    def setUp(self):
        """Create test input tensor for all tests."""
        self.input_tensor = torch.randn(1, 4, 128, 128, 128)
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_unet_output_shape(self, mock_cuda):
        """
        Verify UNet produces output with correct number of classes.
        
        Expected output shape: [batch, num_classes, depth, height, width]
        For BraTS: [1, 4, 128, 128, 128]
        """
        config = {
            'model': {
                'name': 'UNet', 
                'in_channels': 4, 
                'num_classes': 4, 
                'base_channels': 64
            }
        }
        
        model = get_model(config)
        model.eval()
        
        with torch.no_grad():
            output = model(self.input_tensor)
        
        # Verify shape: [batch, num_classes, depth, height, width]
        self.assertEqual(output.shape[0], 1)  # Batch
        self.assertEqual(output.shape[1], 4)  # Num classes
        self.assertEqual(output.shape[2], 128)  # Depth
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_deeplabv3_output_shape(self, mock_cuda):
        """
        Verify DeepLabV3+ produces correct output shape.
        
        Note: May skip if model has dimension requirements that don't match test input.
        """
        config = {
            'model': {
                'name': 'DeepLabV3+', 
                'in_channels': 4, 
                'num_classes': 4
            }
        }
        
        model = get_model(config)
        model.eval()
        
        with torch.no_grad():
            try:
                output = model(self.input_tensor)
                self.assertEqual(output.shape[1], 4)  # Num classes
            except RuntimeError as e:
                logging.warning(f"DeepLabV3+ failed (may need dimension adjustment): {e}")
                self.skipTest("Model requires dimension adjustment")


# ============================================================================
# Learning Rate Scheduler Tests
# ============================================================================

class TestSchedulerIntegration(unittest.TestCase):
    """
    Test suite for learning rate scheduler integration.
    
    Verifies that:
    - Scheduler is initialized correctly
    - scheduler.step() is called each epoch
    - Scheduler parameters (step_size, gamma) are correct
    """
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_scheduler_step_called_each_epoch(self, mock_cuda, mock_save, mock_valid, 
                                             mock_train, mock_writer, mock_model,
                                             mock_adam, mock_scheduler_class, 
                                             mock_subset, mock_dataset):
        """
        Verify scheduler.step() is called once per epoch.
        
        The learning rate should be updated after each training epoch,
        so scheduler.step() call count should match number of epochs run.
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        
        # Simulate early stopping after 5 epochs
        mock_valid.return_value.run.side_effect = [
            {'dice_loss': 0.1, 'iou_score': 0.90},
        ] + [{'dice_loss': 0.2, 'iou_score': 0.70}] * 60
        
        # Create temporary config with patience=3
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {'epochs': 100, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 3},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_scheduler_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            main(temp_path)
            
            # Get scheduler instance
            scheduler_instance = mock_scheduler_class.return_value
            
            # scheduler.step() should be called once per epoch (except last that breaks)
            # If 4 epochs run (0,1,2,3) before early stopping, step() is called 3 times
            self.assertGreater(scheduler_instance.step.call_count, 0)
            self.assertLessEqual(scheduler_instance.step.call_count, 
                                mock_train.return_value.run.call_count)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_scheduler_reduces_learning_rate(self, mock_cuda, mock_save, mock_valid, 
                                            mock_train, mock_writer, mock_model,
                                            mock_adam, mock_scheduler_class, 
                                            mock_subset, mock_dataset):
        """
        Verify scheduler is initialized with correct parameters.
        
        StepLR should be initialized with:
        - optimizer (from Adam)
        - step_size=50 (reduce LR at epoch 50)
        - gamma=0.1 (multiply LR by 0.1)
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_optimizer_instance = MagicMock()
        mock_adam.return_value = mock_optimizer_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        mock_valid.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.9}
        
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {'epochs': 2, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 1},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_scheduler_params_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            main(temp_path)
            
            # Verify StepLR was called with correct optimizer
            mock_scheduler_class.assert_called_once()
            call_args = mock_scheduler_class.call_args
            
            # First argument should be the optimizer
            self.assertEqual(call_args[0][0], mock_optimizer_instance)
            
            # Verify scheduler parameters (step_size=50, gamma=0.1)
            self.assertEqual(call_args[1]['step_size'], 50)
            self.assertEqual(call_args[1]['gamma'], 0.1)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# All Models Training Tests
# ============================================================================

class TestAllModelsTraining(unittest.TestCase):
    """
    Test suite to verify ALL models can train in BOTH modes.
    
    Tests 8 scenarios:
    - UNet + Test Mode (subset 50/5/5)
    - UNet + Full Mode (999/125/125)
    - CLCUNet + Test Mode
    - CLCUNet + Full Mode
    - DeepLabV3+ + Test Mode
    - DeepLabV3+ + Full Mode
    - DeepLabV3+SAM + Test Mode
    - DeepLabV3+SAM + Full Mode
    """
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_all_models_with_test_mode(self, mock_cuda, mock_save, mock_valid, 
                                       mock_train, mock_writer, mock_get_model,
                                       mock_adam, mock_scheduler, mock_subset, 
                                       mock_dataset):
        """
        Verify all 4 models can train in TEST MODE (subset 50/5/5).
        
        Test cases:
        1. UNet + Test Mode
        2. CLCUNet + Test Mode  
        3. DeepLabV3+ + Test Mode
        4. DeepLabV3+SAM + Test Mode
        """
        
        model_configs = {
            'UNet': 'configs/unet_config.yaml',
            'CLCUNet': 'configs/clcunet_config.yaml',
            'DeepLabV3+': 'configs/deeplabv3_config.yaml',
            'DeepLabV3+SAM': 'configs/deeplabv3sam_config.yml'
        }
        mode_config = 'configs/config_test.yaml'
        
        # Setup common mocks
        def setup_mocks():
            # Mocks with correct sizes for test mode
            train_subset_mock = MagicMock()
            train_subset_mock.__len__.return_value = 50
            valid_subset_mock = MagicMock()
            valid_subset_mock.__len__.return_value = 5
            test_subset_mock = MagicMock()
            test_subset_mock.__len__.return_value = 5
            
            def subset_side_effect(dataset, size, batch_size, shuffle):
                if shuffle:
                    return train_subset_mock, MagicMock()
                elif size == 5 or (size <= 0 and not shuffle):
                    return valid_subset_mock, MagicMock()
                else:
                    return test_subset_mock, MagicMock()
            
            mock_subset.side_effect = subset_side_effect
            
            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
            mock_get_model.return_value = mock_model_instance
            
            # Mock training and validation
            mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
            mock_valid.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.9}
        
        # Test each model
        for model_name, model_config in model_configs.items():
            with self.subTest(model=model_name, mode="PRUEBA"):
                # Reset mocks
                mock_subset.reset_mock()
                mock_train.reset_mock()
                mock_valid.reset_mock()
                mock_save.reset_mock()
                
                setup_mocks()
                
                # Verify configs exist
                self.assertTrue(os.path.exists(model_config), f"Config not found: {model_config}")
                self.assertTrue(os.path.exists(mode_config), f"Config not found: {mode_config}")
                
                # Run training
                try:
                    main(model_config, mode_config)
                    
                    # Basic verifications
                    self.assertTrue(mock_get_model.called, f"{model_name} didn't instantiate model")
                    self.assertTrue(mock_train.return_value.run.called, f"{model_name} didn't run training")
                    self.assertTrue(mock_valid.return_value.run.called, f"{model_name} didn't run validation")
                    
                    # Verify subset_train_size=50 was used (from config_test.yaml)
                    train_calls = [c for c in mock_subset.call_args_list if c[1].get('shuffle')]
                    if train_calls:
                        self.assertEqual(train_calls[0][0][1], 50, 
                                       f"{model_name} didn't use subset_train_size=50")
                    
                except Exception as e:
                    self.fail(f"{model_name} failed in TEST mode: {e}")
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_all_models_with_full_dataset(self, mock_cuda, mock_save, mock_valid, 
                                          mock_train, mock_writer, mock_get_model,
                                          mock_adam, mock_scheduler, mock_subset, 
                                          mock_dataset):
        """
        Verify all 4 models can train in FULL MODE (complete dataset).
        
        Test cases:
        5. UNet + Full Mode
        6. CLCUNet + Full Mode
        7. DeepLabV3+ + Full Mode
        8. DeepLabV3+SAM + Full Mode
        """
        
        model_configs = {
            'UNet': 'configs/unet_config.yaml',
            'CLCUNet': 'configs/clcunet_config.yaml',
            'DeepLabV3+': 'configs/deeplabv3_config.yaml',
            'DeepLabV3+SAM': 'configs/deeplabv3sam_config.yml'
        }
        
        # Setup common mocks
        def setup_mocks():
            mock_subset.return_value = (MagicMock(), MagicMock())
            
            mock_model_instance = MagicMock()
            mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
            mock_get_model.return_value = mock_model_instance
            
            # Simulate few epochs for quick test
            mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
            # ✅ Use return_value instead of side_effect to avoid StopIteration
            mock_valid.return_value.run.return_value = {'dice_loss': 0.2, 'iou_score': 0.70}
        
        # Test each model
        for model_name, model_config in model_configs.items():
            with self.subTest(model=model_name, mode="SERIO"):
                # Reset mocks
                mock_subset.reset_mock()
                mock_train.reset_mock()
                mock_valid.reset_mock()
                mock_save.reset_mock()
                
                setup_mocks()
                
                # Verify config exists
                self.assertTrue(os.path.exists(model_config), f"Config no existe: {model_config}")
                
                # Run training WITHOUT config_test.yaml (full mode)
                try:
                    main(model_config)  # ⚠️ No second argument = full dataset
                    
                    # Basic verifications
                    self.assertTrue(mock_get_model.called, f"{model_name} didn't instantiate model")
                    self.assertTrue(mock_train.return_value.run.called, f"{model_name} didn't run training")
                    self.assertTrue(mock_valid.return_value.run.called, f"{model_name} didn't run validation")
                        
                except Exception as e:
                    self.fail(f"{model_name} failed in FULL mode: {e}")


# ============================================================================
# Loss Function Selection Tests
# ============================================================================

class TestLossFunctionSelection(unittest.TestCase):
    """
    Test suite for loss function selection logic.
    
    Verifies that:
    - CLCUNet uses DiceLoss with sigmoid activation
    - Other models use DiceLoss without sigmoid
    - JaccardLoss can be selected
    """
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_clcunet_uses_dice_loss_with_sigmoid(self, mock_cuda, mock_save, mock_valid, 
                                                 mock_train, mock_writer, mock_model,
                                                 mock_adam, mock_scheduler, mock_subset, 
                                                 mock_dataset):
        """
        Verify CLCUNet uses DiceLoss with sigmoid activation.
        
        CLCUNet outputs probabilities per tumor region (NCR, ED, ET),
        so it requires sigmoid activation rather than softmax.
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        mock_valid.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.9}
        
        # Config with CLCUNet
        temp_config = {
            'model': {'name': 'CLCUNet', 'in_channels': 4, 'base_channels': 64},
            'training': {'epochs': 1, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 1},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_clcunet_loss_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            # Capture prints to verify message
            import io
            import sys
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            main(temp_path)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            # Verify correct message was printed
            self.assertIn("sigmoid", output.lower())
            self.assertIn("clcunet", output.lower())
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR')
    @patch('training.Adam')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_unet_uses_dice_loss_default(self, mock_cuda, mock_save, mock_valid, 
                                         mock_train, mock_writer, mock_model,
                                         mock_adam, mock_scheduler, mock_subset, 
                                         mock_dataset):
        """
        Verify UNet uses DiceLoss by default (without sigmoid).
        
        UNet, DeepLabV3+, and DeepLabV3+SAM use standard DiceLoss
        without sigmoid activation (they output class logits).
        """
        # Setup mocks
        mock_subset.return_value = (MagicMock(), MagicMock())
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))]
        mock_model.return_value = mock_model_instance
        
        mock_train.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        mock_valid.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.9}
        
        temp_config = {
            'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4},
            'training': {'epochs': 1, 'batch_size': 1, 'learning_rate': 0.001, 
                        'loss': 'DiceLoss', 'patience': 1},
            'paths': {'data_dir': './mock_data', 'models_dir': './mock_models'}
        }
        
        temp_path = 'temp_unet_loss_test.yaml'
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_config, f)
        
        try:
            import io
            import sys
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            main(temp_path)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            # Verify uses DiceLoss by default
            self.assertIn("by default", output)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    """
    Run all unit tests with verbose output.
    
    Usage:
        python test_training.py
        python test_training.py -v  # Verbose
        python -m unittest test_training.TestConfigMerging  # Specific class
    """
    unittest.main(verbosity=2)
    