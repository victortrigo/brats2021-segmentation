import os
import sys
import torch
import unittest
from unittest.mock import patch
from torch.utils.data import Dataset as TorchDataset

sys.path.insert(0, os.path.abspath('src'))

class MockBratsDataset(TorchDataset):
    """
    Generates 4D random tensors (Channels, Depth, Height, Width) 
    to trick the 3D neural network during testing.
    """
    def __init__(self, *args, **kwargs):
        pass 

    def __len__(self):
        return 2 

    def __getitem__(self, idx):
        # BraTS 3D format: 4 channels, 16 depth slices, 64x64 resolution
        x_mock = torch.randn(4, 16, 64, 64) 
        y_mock = torch.randint(0, 2, (4, 16, 64, 64)).float() 
        return x_mock, y_mock


class TestHPO(unittest.TestCase):
    """
    Test suite for Phase 1 (Hyperparameter Optimization).
    """
    
    @patch('run_hpo.Dataset', new=MockBratsDataset)
    @patch('run_hpo.EPOCHS_HPO', 1)
    def test_fase1_hpo(self):
        """
        Tests the HPO execution with a mocked dataset.
        """
        print("\n" + "="*50)
        print("🧪 STARTING TEST: PHASE 1 (HPO with Mock Dataset)")
        print("="*50)
        
        import run_hpo
        
        try:
            run_hpo.run_hpo()
            print("✅ Phase 1 completed successfully. YAMLs created in output directory.")
        except Exception as e:
            self.fail(f"❌ Phase 1 failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()

# import os
# import yaml
# import optuna
# import random
# import time

# # --- 1. SIMULACIÓN DE ENTRENAMIENTO ---
# def mock_train_model(modelo, lr, weight_decay, batch_size, optimizer_name, use_scheduler):
#     """
#     Simula un entrenamiento que toma 1 segundo.
#     Devuelve un Dice Score aleatorio pero influenciado por los parámetros 
#     para simular que Optuna "aprende" algo.
#     """
#     print(f"  [Mock Train] Entrenando {modelo} con {optimizer_name} | LR: {lr:.5f} | Batch: {batch_size}")
#     time.sleep(0.5) # Simulamos que está procesando imágenes
    
#     # Simulamos un Dice Score (mientras más cerca el LR de 1e-3, "mejor" puntaje)
#     base_dice = 0.60
#     lr_penalty = abs(1e-3 - lr) * 100 
#     mock_dice = base_dice - lr_penalty + random.uniform(0, 0.1)
    
#     # Aseguramos que el Dice esté entre 0 y 1
#     return max(0.0, min(mock_dice, 1.0))

# # --- 2. FUNCIÓN OBJETIVO DE OPTUNA ---
# def objective(trial, modelo):
#     # Definimos el espacio de búsqueda (igual al que usaremos en la realidad)
#     batch_size = trial.suggest_categorical("batch_size", [1, 2])
#     optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
#     lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
#     use_scheduler = trial.suggest_categorical("use_cosine_scheduler", [True, False])
    
#     # Llamamos a nuestra función de entrenamiento falsa
#     val_dice = mock_train_model(
#         modelo=modelo,
#         lr=lr,
#         weight_decay=weight_decay,
#         batch_size=batch_size,
#         optimizer_name=optimizer_name,
#         use_scheduler=use_scheduler
#     )
    
#     return val_dice

# # --- 3. ORQUESTADOR PRINCIPAL ---
# def run_mock_hpo():
#     # Creamos la estructura de carpetas si no existe
#     output_dir = "experiment/phase1_hpo"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Solo probaremos 2 modelos por ahora para ir rápido
#     modelos_a_probar = ["unet", "deeplab_sam"]
    
#     for modelo in modelos_a_probar:
#         print(f"\n{'='*50}")
#         print(f"Iniciando Optuna Mock para modelo: {modelo.upper()}")
#         print(f"{'='*50}")
        
#         # Ocultamos los logs excesivos de Optuna para que la consola se vea limpia
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
        
#         # Creamos el estudio buscando MAXIMIZAR el Dice Score
#         study = optuna.create_study(direction="maximize", study_name=f"mock_{modelo}")
        
#         # Solo haremos 5 trials (intentos) para la prueba de humo
#         study.optimize(lambda trial: objective(trial, modelo), n_trials=5)
        
#         print(f"\n✅ Terminado {modelo.upper()}. Mejor Dice Score: {study.best_value:.4f}")
        
#         # --- GUARDAR RESULTADOS EN YAML ---
#         best_params = study.best_params
#         filepath = os.path.join(output_dir, f"best_params_{modelo}.yaml")
        
#         with open(filepath, 'w') as f:
#             yaml.dump(best_params, f, default_flow_style=False)
            
#         print(f"📁 YAML guardado en: {filepath}")

# if __name__ == "__main__":
#     run_mock_hpo()

