import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from evaluate import evaluate_model  # Tu función matemática estrella
from training import get_model, set_seed

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO (FASE 3)
# ==========================================
DATA_DIR = "./data/processed"
RESULTS_DIR = "experiment/phase2_factorial"
EVAL_DIR = "experiment/phase3_evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

def get_test_dataloader(batch_size: int = 1):
    """
    Loads the unseen test dataset. 
    Batch size is strictly 1 to evaluate metrics per individual patient volume.
    """
    print("   🔄 Building Test DataLoader...")
    CLASSES = ['background', 'NCR', 'ED', 'ET']
    
    # Cargamos el Test Set (Datos que los modelos jamás han visto)
    test_ds = Dataset(os.path.join(DATA_DIR, 'X_test'), os.path.join(DATA_DIR, 'y_test'), CLASSES)
    
    # Para evaluación, SIEMPRE se apaga el shuffle
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return test_loader

def main():
    """
    Main execution loop for Phase 3 (Evaluation).
    Iterates through all 20 saved models, evaluates them on the test set,
    and generates the final ANOVA dataset with clinical metrics.
    """
    print("="*60)
    print("🚀 STARTING PHASE 3: CLINICAL EVALUATION")
    print("="*60)

    # 1. Cargar el Test Set
    test_loader = get_test_dataloader(batch_size=1)
    
    # Lista para acumular todas las métricas para tu Permutation ANOVA
    anova_data = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 2. Recorrer la carpeta de resultados de la Fase 2
    for folder_name in os.listdir(RESULTS_DIR):
        run_dir = os.path.join(RESULTS_DIR, folder_name)
        
        # Ignorar archivos sueltos (como el .csv), solo queremos las carpetas
        if not os.path.isdir(run_dir):
            continue

        # Extraer el nombre del modelo y la semilla desde el nombre de la carpeta
        # ej. "unet_seed42" -> safe_name = "unet", seed = "42"
        parts = folder_name.split("_seed")
        if len(parts) != 2:
            continue
            
        safe_name = parts[0]
        seed = int(parts[1])

        # 3. Buscar el archivo de pesos .pth exacto
        pth_file = f"{safe_name}_seed{seed}.pth"
        pth_path = os.path.join(run_dir, pth_file)

        if not os.path.exists(pth_path):
            print(f"⚠️ WARNING: No weights found for {pth_file}. Skipping...")
            continue

        print(f"\n{'='*40}")
        print(f"🩺 EVALUATING: {safe_name.upper()} | Seed: {seed}")
        print(f"{'='*40}")

        # 4. "Despertar" al modelo y cargarle lo que aprendió en la Fase 2
        set_seed(seed)
        
        # Revertimos el nombre seguro al nombre original (ej. deeplabv3plus -> DeepLabV3+)
        model_name = safe_name.replace("plus", "+").upper()
        if "SAM" in model_name: model_name = "DeepLabV3+SAM" # Corrección de mayúsculas
        if model_name == "CLCUNET": model_name = "CLCUNet"
        if model_name == "UNET": model_name = "UNet"

        # Instanciamos la red vacía
        model = get_model({"model": {"name": model_name}}).to(device)
        
        # Le inyectamos los pesos guardados
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval() # Modo evaluación estricto (apaga el Dropout, etc.)

        # ====================================================================
        # 5. CONEXIÓN CON TU EVALUATE.PY
        # ====================================================================
        # Aquí llamamos a tu función matemática.
        # Ajusta los parámetros según lo que evaluate_model() requiera en tu código real.
        
        print("   🧠 Running clinical metrics (Dice, HD95, Sens, Spec)...")
        
        # ASUMIMOS que evaluate_model te devuelve un diccionario con los resultados
        # Ej: {'dice_NCR': 0.8, 'dice_ED': 0.7, 'hd95_ET': 2.5, ...}
        patient_metrics = evaluate_model(
            model=model, 
            data_loader=test_loader, 
            device=device,
            model_name=model_name
        )
        
        # 6. Empaquetar los datos para la Estadística (ANOVA)
        # Transformamos el diccionario en filas estructuradas
        anova_data.append({
            "Architecture": "DeepLab" if "DeepLab" in model_name else "UNet",
            "Uses_SAM": "Yes" if "SAM" in model_name or model_name == "CLCUNet" else "No",
            "Model_Full_Name": model_name,
            "Seed": seed,
            **patient_metrics # Expande todas las métricas clínicas en nuevas columnas
        })
        
        print(f"   ✅ Metrics recorded for {model_name} (Seed {seed})")

    # ==========================================
    # GUARDADO DEL DATASET ANOVA (EL SANTO GRIAL DE LA TESIS)
    # ==========================================
    print("\n" + "="*60)
    print("💾 GENERATING FINAL CLINICAL DATASET FOR ANOVA")
    print("="*60)
    
    df_anova = pd.DataFrame(anova_data)
    csv_path = os.path.join(EVAL_DIR, "anova_clinical_dataset.csv")
    df_anova.to_csv(csv_path, index=False)
    
    print(f"✅ Phase 3 finished. Statistical dataset exported to: {csv_path}\n")

if __name__ == "__main__":
    main()