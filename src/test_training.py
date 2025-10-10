import unittest
import os
import yaml
import torch
from unittest.mock import patch, MagicMock
from training import main # Importa la función main de tu archivo

# Crear archivos YAML temporales para la prueba
def create_mock_configs(test_dir):
    os.makedirs(test_dir, exist_ok=True)
    
    base_content = """
model:
  name: UNet
paths:
  data_dir: ./mock_data
  models_dir: ./mock_models
training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.001
  loss: DiceLoss
  patience: 10
  subset_train_size: 0
  subset_valid_size: 0  
  subset_test_size: 0
"""
    mode_content = """
training:
  epochs: 3
  batch_size: 2
  subset_train_size: 10
  subset_valid_size: 5  
  subset_test_size: 5 
"""
    with open(os.path.join(test_dir, 'base.yaml'), 'w') as f:
        f.write(base_content)
    with open(os.path.join(test_dir, 'mode.yaml'), 'w') as f:
        f.write(mode_content)

class TestConfiguration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Configuración antes de todas las pruebas
        cls.test_dir = 'temp_test_config'
        create_mock_configs(cls.test_dir)
        cls.base_path = os.path.join(cls.test_dir, 'base.yaml')
        cls.mode_path = os.path.join(cls.test_dir, 'mode.yaml')

    @classmethod
    def tearDownClass(cls):
        # Limpieza después de todas las pruebas
        os.remove(cls.base_path)
        os.remove(cls.mode_path)
        os.rmdir(cls.test_dir)

    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_config_merging(self, mock_cuda, mock_save, mock_valid_epoch, mock_train_epoch, mock_writer, mock_get_model, mock_create_subset, mock_dataset):
        """Prueba que la configuración del modo sobrescribe la base."""
        
        # Mocks para saltarse la inicialización de datos y modelos
        mock_create_subset.return_value = (MagicMock(), MagicMock()) 
        
        # Mockeamos el bucle de entrenamiento para que se detenga rápidamente
        mock_train_epoch.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}
        # Hacemos que la validación solo corra una vez y luego falle para detener el bucle
        mock_valid_epoch.return_value.run.side_effect = [
            {'dice_loss': 0.1, 'iou_score': 0.9}, # Época 0 (guarda modelo)
            {'dice_loss': 0.2, 'iou_score': 0.8}, # Época 1 (no mejora, contador = 1)
            {'dice_loss': 0.3, 'iou_score': 0.7}  # Época 2 (no mejora, contador = 2 -> Early Stop)
        ]
        # ...
        # INSERTA ESTO ANTES DE LLAMAR A main()
        mock_model_instance = MagicMock()
        # ¡CRUCIAL! Simula que model.parameters() devuelve una lista con un parámetro mockeado.
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))] 
        mock_get_model.return_value = mock_model_instance
        # ...
        # Ejecutar main con las configuraciones
        main(self.base_path, self.mode_path)
        
        # 1. Verificar la sobrescritura de 'epochs'
        # El bucle de entrenamiento se debe haber detenido en la época 2 debido a Early Stopping
        # En la configuración base (100) y la de modo (3), la prueba está diseñada para terminar pronto.
        self.assertEqual(mock_train_epoch.return_value.run.call_count, 3) # Se ejecuta 3 veces (0, 1, 2)
        
        # 2. Verificar la sobrescritura de 'subset_train_size'
        # Verificamos que 'create_subset' fue llamado con el valor del modo (10)
        mock_create_subset.assert_any_call(
            mock_dataset.return_value, 
            10, # subset_train_size del modo.yaml
            2,  # BATCH_SIZE del modo.yaml
            shuffle=True
        )

# ... (más pruebas a continuación)
# ... (imports, setup y TestConfiguration)

from training import get_model, unet, deeplabv3, deeplabv3sam, clcunet

class TestModelInstancing(unittest.TestCase):

    def test_unet_instantiation(self):
        """Prueba la instanciación de UNet con sus parámetros."""
# Usa un config que coincida con los parámetros que UNet realmente acepta
        config = {'model': {'name': 'UNet', 'in_channels': 4, 'num_classes': 4, 'base_channels': 64}}   
        model = get_model(config)
        self.assertIsInstance(model, unet.UNet)
        # Verificar que los parámetros se pasan correctamente
        self.assertTrue(hasattr(model, 'out_conv')) 

    def test_deeplabv3plus_sam_instantiation(self):
        """Prueba la instanciación de DeepLabV3+SAM."""
        config = {'model': {'name': 'DeepLabV3+SAM', 'encoder_name': 'snet', 'encoder_weights': 'none'}}
        model = get_model(config)
        self.assertIsInstance(model, deeplabv3sam.DeepLabV3PlusSAM)

    def test_unsupported_model(self):
        """Prueba que un nombre de modelo no soportado lanza un ValueError."""
        config = {'model': {'name': 'CustomNet'}}
        with self.assertRaisesRegex(ValueError, "Modelo no soportado: CustomNet"):
            get_model(config)

# ... (más pruebas a continuación)
# ... (imports, setup, TestConfiguration, TestModelInstancing)

from training import main

class TestTrainingLoopLogic(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Crear un config.yaml mínimo para la prueba
        cls.test_dir = 'temp_loop_test'
        os.makedirs(cls.test_dir, exist_ok=True)
        config_content = """
model:
  name: UNet
  in_channels: 4      
  num_classes: 4      
paths:
  data_dir: ./mock_data
  models_dir: ./mock_models
training:
  epochs: 10
  batch_size: 4
  learning_rate: 0.001
  loss: DiceLoss
  patience: 3
  subset_train_size: 1 
  subset_valid_size: 1  
  subset_test_size: 1 
"""
        cls.config_path = os.path.join(cls.test_dir, 'config.yaml')
        with open(cls.config_path, 'w') as f:
            f.write(config_content)
            
        # Asegurarse de que el directorio de modelos existe para torch.save
        cls.models_dir = os.path.join(cls.test_dir, 'mock_models')
        os.makedirs(cls.models_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Limpieza
        os.remove(cls.config_path)
        os.rmdir(cls.models_dir)
        os.rmdir(cls.test_dir)

    @patch('training.Dataset')
    @patch('training.create_subset')
    @patch('training.StepLR') # Mocks la clase StepLR
    @patch('training.Adam') 
    @patch('training.get_model')
    @patch('training.SummaryWriter')
    @patch('training.TrainEpoch')
    @patch('training.ValidEpoch')
    @patch('training.torch.save')
    @patch('training.torch.cuda.is_available', return_value=False)
    def test_early_stopping_and_checkpoint(self, mock_cuda, mock_save, mock_valid_epoch, mock_train_epoch, mock_writer, mock_get_model, mock_adam, mock_scheduler_class, mock_create_subset, mock_dataset):
    # def test_early_stopping_and_checkpoint(self, mock_save, mock_valid_epoch, mock_train_epoch, mock_get_model, mock_adam, mock_create_subset):
        """
        Prueba la lógica de Early Stopping y el guardado del modelo (Checkpointing).
        Patience está configurada en 3.
        """
        # Mocks para saltarse la inicialización de datos y modelos
        mock_create_subset.return_value = (MagicMock(), MagicMock()) 
        mock_train_epoch.return_value.run.return_value = {'dice_loss': 0.1, 'iou_score': 0.8}

        mock_valid_epoch.return_value.run.side_effect = [
            {'dice_loss': 0.1, 'iou_score': 0.90}, 
            {'dice_loss': 0.1, 'iou_score': 0.92}, 
            {'dice_loss': 0.1, 'iou_score': 0.91}, 
            {'dice_loss': 0.1, 'iou_score': 0.91}, 
            {'dice_loss': 0.1, 'iou_score': 0.91}, # Aquí se detiene el bucle
        ]

        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.empty(0))] 
        mock_get_model.return_value = mock_model_instance

        main(self.config_path)
        
        # 1. Verificar Early Stopping
        self.assertEqual(mock_train_epoch.return_value.run.call_count, 5, "El bucle debe ejecutar 5 épocas (0, 1, 2, 3, 4) y parar en la 4.")
        
        # 2. Verificar Checkpointing (guardado del modelo)
        # El modelo solo se debe guardar en las épocas 0 y 1 (cuando IoU mejora)
        self.assertEqual(mock_save.call_count, 2, "El modelo debe guardarse solo dos veces (mejora en 0.90 y 0.92).")
        
        # 3. Verificar el uso del scheduler
        # Accedemos a la instancia del scheduler creada por el mock de la clase StepLR
        scheduler_instance = mock_scheduler_class.return_value 
        self.assertEqual(scheduler_instance.step.call_count, 4, "El scheduler debe ejecutarse 4 veces (una vez por época antes del break).")


        
# Código de prueba para ejecutarlo:
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# if __name__ == '__main__':
#     # Ejecuta el descubrimiento y ejecución de pruebas
#     unittest.main()