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
        valid_modalities: List[str] = ['flair.nii', 't1.nii', 't1ce.nii', 't2.nii']
        
        self.ids_x: List[str] = sorted([
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d)) and
               any(f.lower().endswith(tuple(valid_modalities)) for f in os.listdir(os.path.join(images_dir, d)))
        ])
        
        self.ids_y: List[str] = sorted([
            d for d in os.listdir(masks_dir)
            if os.path.isdir(os.path.join(masks_dir, d)) and
               any(f.lower().endswith('seg.nii') for f in os.listdir(os.path.join(masks_dir, d)))
        ])

        self.images_fps: List[str] = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        self.masks_fps: List[str] = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]

        self.class_values: List[int] = [CLASSES.index(cls) for cls in classes] if classes else []

        self.augmentation: Optional[callable] = augmentation
        self.preprocessing: Optional[callable] = preprocessing

    def __len__(self) -> int:
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.ids_x)

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
            if file.endswith('flair.nii') or file.endswith('t1.nii') or \
               file.endswith('t1ce.nii') or file.endswith('t2.nii'):
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))
                image_data.append(img_new)
            # Nota: la máscara no debe cargarse del directorio de imágenes

        # Lógica para cargar las máscaras
        for file in files_mask:
            file_path: str = os.path.join(folder_mask, file)
            if file.endswith('seg.nii'):
                img = nib.load(file_path)
                img_new = np.array(img.get_fdata(caching='fill'))
                mask_data.append(img_new)

        image2: np.ndarray = np.asarray(image_data)
        mask2: np.ndarray = np.asarray(mask_data)

        # Normalizar etiquetas 
        mask2[mask2 == 4] = 3
        mask2 = mask2.astype(np.float32)
        
        return image2, mask2
