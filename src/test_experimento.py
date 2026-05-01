import os
import torch
from unittest.mock import patch
from torch.utils.data import Dataset as TorchDataset
import yaml

# 1. EL DATASET FALSO (MOCK)
class MockBratsDataset(TorchDataset):
    """Genera tensores aleatorios para engañar a la red neuronal."""
    def __init__(self, *args, **kwargs):
        pass # Ignoramos las rutas de los archivos

    def __len__(self):
        return 2 # Solo 2 imágenes para que sea rapidísimo

    def __getitem__(self, idx):
        # Forma típica BraTS: 4 canales, 64x64 de tamaño para no gastar RAM local
        x_mock = torch.randn(4, 64, 64) 
        y_mock = torch.randint(0, 2, (4, 64, 64)).float() 
        return x_mock, y_mock


# 2. TEST DE LA FASE 1 (HPO)
# Parcheamos el Dataset en run_hpo y le bajamos las épocas a 1
@patch('run_hpo.Dataset', new=MockBratsDataset)
@patch('run_hpo.EPOCHS_HPO', 1)
def test_fase1():
    print("\n" + "="*50)
    print("🧪 INICIANDO TEST: FASE 1 (HPO con Mock Dataset)")
    print("="*50)
    
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR) # Ocultar logs masivos
    
    import run_hpo
    
    # Para el test, vamos a reemplazar la función que orquesta los modelos
    # para que solo corra UNet y genere su YAML, ahorrando tiempo.
    original_modelos = ["UNet"]
    
    with patch('run_hpo.modelos', original_modelos, create=True):
        try:
            # Optuna correrá 20 trials pero de solo 1 época y 2 imágenes, tomará < 5 segundos
            run_hpo.run_hpo()
            print("✅ Fase 1 completada correctamente. YAML creado.")
        except Exception as e:
            print(f"❌ Error en Fase 1: {e}")


# 3. TEST DE LA FASE 2 (FACTORIAL)
# Parcheamos el Dataset en run_phase2_factorial, las épocas y las semillas
@patch('run_phase2_factorial.Dataset', new=MockBratsDataset)
@patch('run_phase2_factorial.EPOCHS_MAX', 1)
@patch('run_phase2_factorial.SEEDS', [42]) # Solo 1 semilla para el test
def test_fase2():
    print("\n" + "="*50)
    print("🧪 INICIANDO TEST: FASE 2 (Factorial con Mock Dataset)")
    print("="*50)
    
    # Nos aseguramos de que exista el YAML que Fase 2 intentará leer
    hpo_dir = "experiment/phase1_hpo"
    os.makedirs(hpo_dir, exist_ok=True)
    with open(f"{hpo_dir}/best_params_unet.yaml", "w") as f:
        yaml.dump({"batch_size": 2, "lr": 0.001, "optimizer": "Adam", "weight_decay": 1e-5, "use_cosine_scheduler": False}, f)

    import run_phase2_factorial
    try:
        # Corremos la fase 2 con use_amp en True (o False si lo prefieres)
        run_phase2_factorial.run_phase2(use_amp=True)
        print("✅ Fase 2 completada correctamente. CSV guardado.")
    except Exception as e:
        print(f"❌ Error en Fase 2: {e}")

if __name__ == "__main__":
    test_fase1()
    test_fase2()
    print("\n🏁 Test Finalizado. Si hay 2 checks verdes (✅), el código está perfecto.")