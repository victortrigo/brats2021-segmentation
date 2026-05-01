
import os
import yaml
import torch

import pandas as pd
from torch.utils.data import DataLoader

from dataset import Dataset
from training import run_experiment

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO (FASE 2)
# ==========================================
DATA_DIR = "./data/processed"               
SEEDS = [42, 123, 2026, 7, 999]  # Tus semillas estadísticas definitivas
HPO_DIR = "experiment/phase1_hpo" 
RESULTS_DIR = "experiment/phase2_factorial"  

# Aseguramos que el directorio de salida exista
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_full_dataloaders(batch_size: int):
    """
    Loads the complete dataset adapting to the optimal batch_size found by Optuna.
    
    Args:
        batch_size (int): The optimal batch size loaded from the YAML config.
        
    Returns:
        tuple: (train_loader, valid_loader) containing the complete dataset.
    """
    print(f"   🔄 Building DataLoaders with Batch Size: {batch_size}")
    CLASSES = ['background', 'NCR', 'ED', 'ET']

    train_ds = Dataset(os.path.join(DATA_DIR, 'X_train'), os.path.join(DATA_DIR, 'y_train'), CLASSES)
    valid_ds = Dataset(os.path.join(DATA_DIR, 'X_val'), os.path.join(DATA_DIR, 'y_val'), CLASSES)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, valid_loader


def main():
    """
    Main execution loop for Phase 2 (Factorial Training).
    
    This function:
    1. Iterates over all specified architectures.
    2. Dynamically loads their respective hyperparameters from Phase 1.
    3. Executes the training engine for multiple random seeds.
    4. Aggregates the best metrics (Fscore/IoU) into a structured CSV for ANOVA.
    """
    print("="*60)
    print("🚀 STARTING PHASE 2: FACTORIAL TRAINING")
    print("="*60)

    modelos = ["UNet", "CLCUNet", "DeepLabV3+", "DeepLabV3+SAM"]
    results_data = [] # Lista para acumular los resultados del CSV

    for model_name in modelos:
        # Limpiar el nombre para buscar el archivo correcto (DeepLabV3+ -> deeplabv3plus)
        safe_name = model_name.replace("+", "plus").lower()
        yaml_path = os.path.join(HPO_DIR, f"best_params_{safe_name}.yaml")
        
        # Verificar si Optuna dejó el archivo listo en la Fase 1
        if not os.path.exists(yaml_path):
            print(f"\n⚠️ ERROR: No YAML was found for {model_name}. Run Phase 1 first.")
            continue

        # Leer la receta de hiperparámetros ganadores
        with open(yaml_path, 'r') as f:
            best_params = yaml.safe_load(f)

        print(f"\n{'='*40}")
        print(f"📦 MODEL: {model_name}")
        print(f"📋 Recipe loaded: {best_params}")
        print(f"{'='*40}")

        # 1. EXTRAER BATCH SIZE Y CREAR DATALOADERS DINÁMICAMENTE
        # Usamos .get() con valor por defecto 2 como medida de seguridad
        batch_size = int(best_params.get('batch_size', 2))
        train_loader, valid_loader = get_full_dataloaders(batch_size)


        # 2. BUCLE DE SEMILLAS
        for seed in SEEDS:
            print(f"\n   🌱 Starting Trial -> Seed: {seed}")

            # Crear sub-carpeta exclusiva para los pesos y logs de este modelo/semilla
            run_dir = os.path.join(RESULTS_DIR, f"{safe_name}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            # 🔥 EJECUCIÓN DEL ENTRENAMIENTO REAL
            final_fscore, final_iou = run_experiment(
                model_name=model_name,
                seed=seed,
                best_params=best_params,
                run_dir=run_dir,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device='cuda:0'
            )

            # 3. REGISTRAR RESULTADOS
            # Estructuramos los datos con los factores listos para el Permutation ANOVA
            results_data.append({
                "Architecture": "DeepLab" if "DeepLab" in model_name else "UNet",
                "Uses_SAM": "Yes" if "SAM" in model_name or model_name == "CLCUNet" else "No",
                "Model_Full_Name": model_name,
                "Seed": seed,
                "Batch_Size": batch_size,
                "Best_Fscore": final_fscore,
                "Best_IoU": final_iou
            })
    
    # ==========================================
    # GUARDADO FINAL DEL CSV
    # ==========================================
    print("\n" + "="*60)
    print("💾 SAVING FINAL RESULTS TABLE")
    print("="*60)

    # Convertimos la lista de diccionarios en un DataFrame y exportamos
    df_results = pd.DataFrame(results_data)
    csv_path = os.path.join(RESULTS_DIR, "factorial_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print(f"✅ Experiment finished. Results exported to: {csv_path}\n")

if __name__ == "__main__":
    main()