# Brain Tumor Segmentation (BraTS) Challenge 2021

Este repositorio contiene el entrenamiento de modelos de deep learning como U-Net y DeepLabv3+; ademas de añadir modificaciones en sus respectivas arquitecturas, incorporando mecanismos de atención como SAM (Segmented attention module).

## Contenido

- `data/` : Directorio para almacenar los datos del BraTS 2021.
- `models/` : Modelos entrenados.
- `notebooks/` : Notebooks para exploración.
- `src/` : Scripts de dataset, modelos, entrenamiento, evaluación y predicción.

## Clonar repositorio

```bash
git clone https://github.com/victortrigo/brats2021-segmentation.git
cd brats2021-segmentation
```

## Instalación del entorno

Se recomienda usar Conda:

```bash
conda env create -f environment.yml
conda activate brats2021
```

## Configuraciones

Existen 4 archivos de configuración en `configs/`:

- `unet_config.yaml` : Parámetros para entrenar la U-Net base.
- `clcunet_config.yaml` : U-Net con módulo de atención SAM.
- `deeplabv3_config.yaml` : DeepLabV3+ estándar.
- `deeplabv3sam_config.yaml` : DeepLabV3+ con SAM.
- `config_test.yaml`: Activar el modo de prueba rápida (ej. 50 épocas, subsets 50/5/5)

### Entrenamiento (dataset completo 999/125/125)

Para entrenar un modelo específico:

```bash
python src/training.py --config configs/unet_config.yaml
```

### Entrenamiento (dataset de prueba 50/5/5)

Para entrenar un modelo específico:

```bash
python src/training.py --config configs/deeplabv3_config.yaml --mode configs/config_test.yaml
```


## Evaluación

Existen 4 modelos entrenados, cada uno con su best model correspondiente en `models/`:

- `best_model_unet.pth`
- `best_model_clcunet.pth`
- `best_model_deeplabv3.pth`
- `best_model_deeplabv3sam.pth`

```bash
python src/evaluate.py --model models/best_model_unet.pth
```


## Visualización

TensorBoard para monitorear entrenamiento y métricas, dentro de `runs/`:

- `runs/unet/`
- `runs/clcunet/`
- `runs/deeplabv3/`
- `runs/deeplabv3sam/`

Para monitorear el entrenamiento de un modelo:

```bash
tensorboard --logdir runs/unet/
```

## Referencias

* [The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification](https://arxiv.org/abs/2107.02314)
* [Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features](https://www.nature.com/articles/sdata2017117)
* [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](https://ieeexplore.ieee.org/abstract/document/6975210)
* [Deep Learning for Image Segmentation with Python & Pytorch](https://www.udemy.com/course/deep-learning-for-semantic-segmentation-with-python-pytorh/)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
* [CLCU-Net: Cross-level connected U-shaped network with selective feature aggregation attention module for brain tumor segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0169260721002285)
* [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)
* [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://ieeexplore.ieee.org/abstract/document/7913730)
* [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html)

* [DeeplabV3+ Model with CBAM and CSPM Attention Mechanism for Navel Orange Defects Segmentation](https://openurl.ebsco.com/EPDB%3Agcd%3A1%3A12484113/detailv2?sid=ebsco%3Aplink%3Ascholar&id=ebsco%3Agcd%3A180179636&crl=c&link_origin=scholar.google.com)

