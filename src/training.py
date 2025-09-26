import os
from datetime import datetime
from typing import Any, Dict

import torch
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import clcu_net
import dataset
import deeplabv3
import deeplabv3sam
import unet
from metrics import Accuracy, DiceLoss, Fscore, IoU, JaccardLoss, Precision, Recall
from train import TrainEpoch, ValidEpoch

# Definición del dispositivo para el entrenamiento
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Diccionarios de mapeo para optimizadores y funciones de pérdida
OPTIMIZERS = {
    "Adam": Adam,
    "SGD": SGD,
}

LOSSES = {
    "DiceLoss": DiceLoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "JaccardLoss": JaccardLoss,
}

def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Crea y devuelve el modelo basado en la configuración."""
    model_name = config['model']['name']
    model_params = {k: v for k, v in config['model'].items() if k != 'name'}
    
    if model_name == "UNet":
        return unet.UNet(**model_params)
    elif model_name == "DeepLabV3+":
        return deeplabv3.DeepLabV3Plus(**model_params)
    elif model_name == "DeepLabV3+SAM":
        return deeplabv3sam.DeepLabV3PlusSAM(**model_params)
    elif model_name == "CLCUNet":
        return clcu_net.CLCUNet(**model_params)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    

def main(config_path: str):
    """Función principal para el entrenamiento."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Carga de datos
    DATA_DIR = config['paths']['data_dir']
    CLASSES = ['background', 'NCR', 'ED', 'ET']

    x_train_dir = os.path.join(DATA_DIR, 'X_train')
    y_train_dir = os.path.join(DATA_DIR, 'y_train')
    x_valid_dir = os.path.join(DATA_DIR, 'X_val')
    y_valid_dir = os.path.join(DATA_DIR, 'y_val')

    train_dataset = dataset.Dataset(x_train_dir, y_train_dir, CLASSES)
    valid_dataset = dataset.Dataset(x_valid_dir, y_valid_dir, CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    metrics = [
        IoU(threshold=0.5),
        Accuracy(threshold=0.5),
        Fscore(threshold=0.5),
        Recall(threshold=0.5),
        Precision(threshold=0.5),
    ]

    # Carga del modelo
    model = get_model(config)

    # Instanciación dinámica de optimizador y pérdida
    optimizer_class = OPTIMIZERS[config['training']['optimizer']]
    optimizer = optimizer_class(model.parameters(), lr=config['training']['learning_rate'])
    
    loss_class = LOSSES[config['training']['loss']]
    loss = loss_class()

    print(f"Iniciando el entrenamiento del modelo: {config['model']['name']}")

    # Crea un nombre de corrida único al inicio de la ejecución
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['model']['name']}_{now}"

    # Crea el escritor de TensorBoard con el nombre único
    writer = SummaryWriter(os.path.join('runs', run_name))

    # Lógica de entrenamiento
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

    max_score = 0
    epochs = config['training']['epochs']

    for i in range(0, epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Registra las métricas en TensorBoard
        writer.add_scalar('Loss/train', train_logs['dice_loss'], i)
        writer.add_scalar('IoU/train', train_logs['iou_score'], i)
        writer.add_scalar('Loss/validation', valid_logs['dice_loss'], i)
        writer.add_scalar('IoU/validation', valid_logs['iou_score'], i)

        # Save the model with best iou score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            model_path = os.path.join(config['paths']['models_dir'], f"{config['model']['name']}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'Modelo guardado en {model_path}')

        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    main(args.config)