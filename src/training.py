import os
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
import yaml
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset, create_subset

import clcunet
# import dataset
import deeplabv3
import deeplabv3sam
import unet

from metrics import Accuracy, DiceLoss, Fscore, IoU, JaccardLoss, Precision, Recall
from train import TrainEpoch, ValidEpoch


# Definición del dispositivo para el entrenamiento
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Instancia el modelo de red neuronal a partir de su nombre en la configuración.

    Args:
        config (Dict[str, Any]): Diccionario de configuración cargado desde YAML.

    Returns:
        torch.nn.Module: El modelo de segmentación instanciado.
    """
    model_name = config['model']['name']
    # Extrae todos los parámetros del modelo, excluyendo el nombre
    model_params = {k: v for k, v in config['model'].items() if k != 'name'}
    
    if model_name == "UNet":
        return unet.UNet(**model_params)
    elif model_name == "DeepLabV3+":
        return deeplabv3.DeepLabV3Plus(**model_params)
    elif model_name == "DeepLabV3+SAM":
        return deeplabv3sam.DeepLabV3PlusSAM(**model_params)
    elif model_name == "CLCUNet":
        return clcunet.CLCUNet(**model_params)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    

def main(config_path: str, mode_config_path: str = None):
    """
    Función principal para inicializar y ejecutar el ciclo de entrenamiento.

    Maneja la carga y fusión de configuraciones, la preparación de datos
    (incluyendo subsets), la inicialización de modelo/optimizador/pérdida,
    y el bucle de entrenamiento con Early Stopping y TensorBoard.
    
    Args:
        config_path (str): Ruta al archivo de configuración base (e.g., unet_config.yaml).
        mode_config_path (str, optional): Ruta al archivo de modo (e.g., config_test.yaml)
                                           para sobrescribir valores. Defaults a None.
    """
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if mode_config_path and os.path.exists(mode_config_path):
        with open(mode_config_path, 'r') as f_mode:
            mode_config = yaml.safe_load(f_mode)
        
        # Sobrescribe la sección 'training' de la configuración BASE con la del MODO.
        config['training'].update(mode_config.get('training', {}))
        
        print(f"ATENCIÓN: Sobrescribiendo parámetros de 'training' con {mode_config_path}")
    
    print(f"Usando dispositivo: {DEVICE}")

    # Carga de datos
    DATA_DIR = config['paths']['data_dir']
    CLASSES = ['background', 'NCR', 'ED', 'ET']
    BATCH_SIZE = config['training']['batch_size']

    # Obtener tamaños de subset (usa 0 si no están definidos, lo que significa dataset completo)
    subset_train_size = config['training'].get('subset_train_size', 0)
    subset_valid_size = config['training'].get('subset_valid_size', 0)
    subset_test_size = config['training'].get('subset_test_size', 0)


    x_train_dir = os.path.join(DATA_DIR, 'X_train')
    y_train_dir = os.path.join(DATA_DIR, 'y_train')
    x_valid_dir = os.path.join(DATA_DIR, 'X_val')
    y_valid_dir = os.path.join(DATA_DIR, 'y_val')
    x_test_dir = os.path.join(DATA_DIR, 'X_test')
    y_test_dir = os.path.join(DATA_DIR, 'y_test')

    # Instanciación de Datasets completos
    train_dataset_full = Dataset(x_train_dir, y_train_dir, CLASSES)
    valid_dataset_full = Dataset(x_valid_dir, y_valid_dir, CLASSES)
    test_dataset_full = Dataset(x_test_dir, y_test_dir, CLASSES)

    # Creación de DataLoaders usando la lógica de subsets
    subset_train, train_loader = create_subset(train_dataset_full, subset_train_size, BATCH_SIZE, shuffle=True)
    subset_valid, valid_loader = create_subset(valid_dataset_full, subset_valid_size, BATCH_SIZE, shuffle=False)
    # Se incluye el loader de test, aunque no se usa en el bucle de entrenamiento, es útil para el futuro.
    subset_test, test_loader = create_subset(test_dataset_full, subset_test_size, BATCH_SIZE, shuffle=False)

    # Mensaje de confirmación de modo
    if subset_train_size > 0 or subset_valid_size > 0:
        print("--- MODO PRUEBAS RÁPIDAS (SUBSET) ---")
        print(f"Train size: {len(subset_train)}, Valid size: {len(subset_valid)}, Test size: {len(subset_test)}")
    else:
        print(f"Modo Entrenamiento Completo. Train size: {len(subset_train)}, Valid size: {len(subset_valid)}")
    

    metrics = [
        IoU(threshold=0.5),
        Accuracy(threshold=0.5),
        Fscore(threshold=0.5),
        Recall(threshold=0.5),
        Precision(threshold=0.5),
    ]

    model = get_model(config)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Scheduler: Reduce el LR a 0.1 en la época 50
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1) # Reduce el LR a 0.1 en la época 50
    
    loss_name = config['training']['loss']

    if loss_name == "DiceLoss" and config['model']['name'] == "CLCUNet":
        # Lógica de pérdida específica para CLCUNet
        loss = DiceLoss(activation="sigmoid")
    elif loss_name == "JaccardLoss":
        loss = JaccardLoss()
    else:
        raise ValueError(f"Pérdida no soportada: {loss_name}")

    print(f"Iniciando el entrenamiento del modelo: {config['model']['name']}")

    # Inicialización de TensorBoard
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['model']['name']}_{now}"
    writer = SummaryWriter(os.path.join('runs', run_name))

    # Inicialización de las clases de entrenamiento
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # BUCLE PRINCIPAL DE ENTRENAMIENTO
    max_score = 0
    epochs = config['training']['epochs']

    PATIENCE = config['training']['patience']
    patience_counter = 0        

    for i in range(0, epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Registra las métricas en TensorBoard
        writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
        writer.add_scalar('IoU/train', train_logs['iou_score'], i)
        writer.add_scalar('Loss/validation', valid_logs['dice_loss'], i)
        writer.add_scalar('IoU/validation', valid_logs['iou_score'], i)

        # Lógica de Checkpoint y Early Stopping
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            model_path = os.path.join(config['paths']['models_dir'], f"{config['model']['name']}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'Modelo guardado en {model_path}')

            patience_counter = 0  # Reinicia el contador: hubo mejora
        else:
            patience_counter += 1 # Incrementa el contador: no hubo mejora
            if patience_counter >= PATIENCE:
                print(f'Early stopping en la época {i} después de {PATIENCE} épocas sin mejora.')
                break # Detiene el entrenamiento

        # Aplicar el paso del Learning Rate Scheduler
        scheduler.step()

    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Argumento obligatorio: la configuración base del modelo (e.g., unet_config.yaml)
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML.")
    # Argumento opcional: la configuración de modo (e.g., config_test.yaml)
    parser.add_argument("--mode", type=str, required=False, help="Ruta opcional al archivo de configuración de modo (e.g., config_test.yaml) para sobrescribir valores.")
    args = parser.parse_args()
    
    # IMPORTANTE: Se llama a main con el argumento mode para habilitar la fusión de configs.
    main(args.config)

# python training.py --config unet_config.yaml
# python training.py --config unet_config.yaml --mode config_test.yaml