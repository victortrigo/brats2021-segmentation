import os
import torch
import nibabel as nib
import numpy as np
from typing import List, Optional, Tuple
from torch.utils.data import Dataset as BaseDataset

# Define las clases para la segmentación
CLASSES = ['background', 'NCR', 'ED', 'ET']

class Dataset(BaseDataset):
    """
    Clase de Dataset para cargar datos de BraTS.
    """
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        classes: List[str] = None,
        augmentation: Optional[callable] = None,
        preprocessing: Optional[callable] = None
    ) -> None:
        """
        Inicializa el dataset.

        Args:
            images_dir (str): Directorio raíz con las carpetas de imágenes de los pacientes.
            masks_dir (str): Directorio raíz con las carpetas de máscaras de los pacientes.
            classes (List[str], optional): Lista de nombres de clases a segmentar.
            augmentation (Optional[callable], optional): Función de aumento de datos.
            preprocessing (Optional[callable], optional): Función de preprocesamiento.
        """
        valid_modalities: List[str] = ['flair.nii.gz', 't1.nii.gz', 't1ce.nii.gz', 't2.nii.gz']

        # Primero, obtén todos los IDs de pacientes en cada directorio
        try:
            ids_x_temp = set([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
            ids_y_temp = set([d for d in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, d))])
        except FileNotFoundError as e:
            raise RuntimeError(f"Directorio no encontrado: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error al listar directorios: {str(e)}") from e

        # Encuentra los IDs que están en AMBOS directorios
        common_ids = sorted(list(ids_x_temp.intersection(ids_y_temp)))
        if not common_ids:
            raise RuntimeError("No se encontraron IDs comunes entre los directorios de imágenes y máscaras.")

        # Filtra la lista común para asegurarte de que cada carpeta tiene los archivos necesarios
        self.ids: List[str] = []
        for d in common_ids:
            try:
                imgs = os.listdir(os.path.join(images_dir, d))
                msks = os.listdir(os.path.join(masks_dir, d))
            except Exception as e:
                print(f"Advertencia: No se pudo acceder a la carpeta {d}: {e}")
                continue
            if any(f.lower().endswith(tuple(valid_modalities)) for f in imgs) and \
               any(f.lower().endswith('seg.nii.gz') for f in msks):
                self.ids.append(d)

        self.images_fps: List[str] = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps: List[str] = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.class_values: List[int] = [CLASSES.index(cls) for cls in classes] if classes else []

        self.augmentation: Optional[callable] = augmentation
        self.preprocessing: Optional[callable] = preprocessing

    def __len__(self) -> int:
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga y devuelve una muestra del dataset en el índice `i`.

        Args:
            i (int): Índice de la muestra a cargar.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Una tupla que contiene la imagen y la máscara.
        """
        folder_image: str = self.images_fps[i]
        folder_mask: str = self.masks_fps[i]
        
        # Cargar archivos de imagen y máscara
        files_image: List[str] = os.listdir(folder_image)
        files_mask: List[str] = os.listdir(folder_mask)

        image_data: List[np.ndarray] = []
        mask_data: List[np.ndarray] = []
        
        # Lógica para cargar las imágenes
        for file in files_image:
            file_path: str = os.path.join(folder_image, file)
            if file.endswith('flair.nii.gz') or file.endswith('t1.nii.gz') or \
               file.endswith('t1ce.nii.gz') or file.endswith('t2.nii.gz'):
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))
                image_data.append(img_new)
            # Nota: la máscara no debe cargarse del directorio de imágenes

        # Lógica para cargar las máscaras
        for file in files_mask:
            file_path: str = os.path.join(folder_mask, file)
            if file.endswith('seg.nii.gz'):
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))
                mask_data.append(img_new)

        image2: np.ndarray = np.asarray(image_data)
        mask2: np.ndarray = np.asarray(mask_data)

        # Normalizar etiquetas 
        mask2[mask2 == 4] = 3
        mask2 = mask2.astype(np.float32)

        # Convierte el array de NumPy a un tensor de PyTorch
        image_tensor = torch.from_numpy(image2)
        mask_tensor = torch.from_numpy(mask2).long()
        
        return image_tensor, mask_tensor

# Código de prueba
if __name__ == "__main__":
    # Usa la ruta relativa desde la raíz del proyecto
    data_base_path = 'data/processed'

    # Usa os.path.join para construir rutas de forma segura
    images_dir = os.path.join(data_base_path, 'X_train')
    masks_dir = os.path.join(data_base_path, 'y_train')
    
    classes = ['background', 'NCR', 'ED', 'ET']

    dataset = Dataset(images_dir, masks_dir, classes)

    try:
        img, msk = dataset[0]
        print(f"Total de muestras: {len(dataset)}")
        print(f"Tipo de dato de la imagen: {img.dtype}")
        print(f"Tipo de dato de la máscara: {msk.dtype}")
        print(f"Shape imagen: {img.shape}")
        print(f"Shape máscara: {msk.shape}")
        # Muestra los valores de píxeles de una sección de la imagen
        print("\nValores de píxeles de la imagen (slice 64, canal 0):")
        print(img[0, 64, 64:70, 64:70])

        print(f"Valores de la imagen (min/max): {img.min().item()} / {img.max().item()}")

        # Muestra los valores únicos y el conteo de etiquetas en la máscara
        unique, counts = np.unique(msk.numpy(), return_counts=True)
        label_counts = dict(zip(unique, counts))
        print("\nValores únicos y conteo en la máscara:")
        print(label_counts)
    except Exception as e:
        print(f"Error al cargar la muestra: {e}")
