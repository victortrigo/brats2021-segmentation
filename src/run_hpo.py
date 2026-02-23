import os
import yaml
import optuna
import torch
from torch.cuda.amp import autocast, GradScaler

from dataset import Dataset, create_subset
from training import get_model
from metrics import DiceLoss, IoU

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO (FASE 1)
# ==========================================
DATA_DIR = "./data/processed" 
EPOCHS_HPO = 40               # Épocas máximas por intento
SUBSET_TRAIN = 100            # ~10% del total
SUBSET_VALID = 30             # ~24% del total

def get_dataloaders(batch_size):
    """Carga un subconjunto minúsculo de datos para iterar rápido."""
    CLASSES = ['background', 'NCR', 'ED', 'ET']
    train_ds = Dataset(os.path.join(DATA_DIR, 'X_train'), os.path.join(DATA_DIR, 'y_train'), CLASSES)
    valid_ds = Dataset(os.path.join(DATA_DIR, 'X_val'), os.path.join(DATA_DIR, 'y_val'), CLASSES)
    
    _, train_loader = create_subset(train_ds, SUBSET_TRAIN, batch_size)
    _, valid_loader = create_subset(valid_ds, SUBSET_VALID, batch_size)
    return train_loader, valid_loader

def objective(trial, model_name):
    # 1. Definir el espacio de búsqueda
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    use_scheduler = trial.suggest_categorical("use_cosine_scheduler", [True, False])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Cargar Dataloaders reales (pero pequeños)
    try:
        train_loader, valid_loader = get_dataloaders(batch_size)
    except Exception as e:
        print(f"Error cargando datos: {e}. Revisa la ruta en DATA_DIR.")
        return 0.0 # Castigo para que Optuna evite este error

    # 3. Inicializar tu Modelo usando tu factory get_model
    config_mock = {
        'model': {'name': model_name, 'in_channels': 4, 'num_classes': 4, 'base_channels': 64}
    }
    model = get_model(config_mock).to(device)
    
    # 4. Configurar Optimizador y Scheduler
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HPO)
        
    criterion = DiceLoss().to(device)
    metric_iou = IoU().to(device)
    scaler = GradScaler() # Para Mixed Precision (aceleración de GPU)
    
    best_iou = 0.0
    
    # 5. Bucle de Entrenamiento Rápido
    for epoch in range(EPOCHS_HPO):
        model.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Entrenamiento con Mixed Precision
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        if use_scheduler:
            scheduler.step()
            
        # Validación
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    pred = model(x)
                    val_iou += metric_iou(pred, y).item()
                    
        val_iou /= len(valid_loader) # Promedio del subset
        
        if val_iou > best_iou:
            best_iou = val_iou
            
        # 6. Reportar a Optuna y aplicar Pruning
        trial.report(val_iou, epoch)
        
        # Si el modelo rinde muy mal en las primeras épocas, se cancela para ahorrar tiempo
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_iou

def run_hpo():
    output_dir = "experiment/phase1_hpo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Estos son los nombres exactos definidos en tu archivo training.py
    modelos = ["UNet", "DeepLabV3+", "DeepLabV3+SAM", "CLCUNet"]
    
    for model_name in modelos:
        print(f"\n{'='*50}")
        print(f"Iniciando HPO REAL para modelo: {model_name}")
        print(f"{'='*50}")
        
        # Configuramos el podador: Esperar 10 épocas antes de empezar a matar procesos malos
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(direction="maximize", study_name=f"hpo_{model_name}", pruner=pruner)
        
        # Ejecutamos 20 combinaciones por modelo
        study.optimize(lambda trial: objective(trial, model_name), n_trials=20)
        
        print(f"\n✅ Terminado {model_name}. Mejor IoU encontrado: {study.best_value:.4f}")
        
        # Guardar en YAML
        best_params = study.best_params
        safe_name = model_name.replace("+", "plus").lower() # Limpiar nombre para el archivo
        filepath = os.path.join(output_dir, f"best_params_{safe_name}.yaml")
        
        with open(filepath, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
            
        print(f"📁 YAML guardado exitosamente en: {filepath}")

if __name__ == "__main__":
    # Asegúrate de que los logs de Optuna sean visibles para ver el progreso real
    optuna.logging.set_verbosity(optuna.logging.INFO)
    run_hpo()