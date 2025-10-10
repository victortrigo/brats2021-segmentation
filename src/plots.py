import matplotlib.pyplot as plt

def plot_sample(sample, slice_index=64):
    """
    Grafica un ejemplo de imágenes y máscara.

    Args:
        sample (tuple): Un ejemplo del Dataset (imágenes, máscara).
        slice_index (int): Índice del corte (en el eje Z) a mostrar.
    """
    images, mask = sample
    titles = ['FLAIR', 'T1', 'T1ce', 'T2', 'Máscara']

    fig, axes = plt.subplots(1, 5, figsize=(20, 10))

    # Graficar imágenes (FLAIR, T1, T1ce, T2)
    for i in range(4):
        axes[i].imshow(images[i, :, :, slice_index], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')  # Elimina las marcas de los ejes

    # Graficar máscara
    axes[4].imshow(mask[0, :, :, slice_index], cmap='viridis')
    axes[4].set_title(titles[4])
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()