# Brain Tumor Segmentation (BraTS) Challenge 2021 using U-Net, DeepLabv3+ and Segmented Attention Module (SAM)

## Content

- `data/` : datset BraTS 2021
- `models/` : models trainers
- `src/` : Scripts de dataset, models, training, evaluate and prediction

## 1. Clone

```bash
git clone https://github.com/victortrigo/brats2021-segmentation.git
```

## 2. Enviroment

Use Conda

```bash
pip install -r requirements.txt
```

## Configurations

Configurations or hyperparameters `configs/`:

- `unet_config.yaml` : U-Net 
- `clcunet_config.yaml` : U-Net (CLCU-Net) whit SAM 
- `deeplabv3_config.yaml` : DeepLabV3+ 
- `deeplabv3sam_config.yaml` : DeepLabV3+ with SAM
- `config_test.yaml`: test mode (ej. 50 epocs, subsets 50/5/5)

## Train 
Dataset full (999/125/125) and Dataset test (50/5/5)


```bash
# Train TEST to 4 models
python main.py --mode test

# Train FULL to 4 models
python main.py --mode full

# Train only a spycific models 
python main.py --mode test --models unet clcunet
python main.py --mode full --models deeplabv3 deeplabv3sam
```


## Evaluation

4 trained models in `models/`:

- `best_model_unet.pth`
- `best_model_clcunet.pth`
- `best_model_deeplabv3.pth`
- `best_model_deeplabv3sam.pth`

```bash
pass
```


## Visualization

TensorBoard monitoring train and metrics in `runs/`:

- `runs/unet/`
- `runs/clcunet/`
- `runs/deeplabv3/`
- `runs/deeplabv3sam/`

Para monitorear el entrenamiento de un modelo:

```bash
tensorboard --logdir runs/unet/
```

## Architeture visualization

```bash
python src/unet.py
python src/deeplabv3.py
python src/clcunet.py
python src/deeplabv3sam.py
python src/backbone.py
python src/convs.py
python src/sam.py
python src/pooling.py
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

