# Brain Tumor Segmentation (BraTS) Challenge 2021
### Experimental Pipeline using U-Net, DeepLabV3+, CLCUNet, and SAM (Segmented Attention Module)

This repository contains a fully automated Deep Learning pipeline designed for statistical robustness in medical image segmentation. It features dynamic Hyperparameter Optimization (HPO), Factorial Design training across multiple seeds, and comprehensive clinical metric evaluation for Permutation ANOVA analysis.

## 📂 Project Structure

- `data/processed/` : Preprocessed BraTS 2021 dataset (X_train, y_train, X_val, X_test).
- `src/` : Source code including models, data loaders, training engines, and tests.
- `experiment/` : Auto-generated directory containing all pipeline outputs (YAMLs, weights, logs, and CSVs).


- `data/` : datset BraTS 2021
- `models/` : models trainers
- `src/` : Scripts de dataset, models, training, evaluate and prediction

## 🛠️ 1. Clone & Environment

```bash
git clone https://github.com/victortrigo/brats2021-segmentation.git
```

It is highly recommended to use Conda for environment management:

```bash
conda create -n brats2021 python=3.9
conda activate brats2021
pip install -r requirements.txt
```

## 2. The 3-Phase Experimental Pipeline

The entire experiment is orchestrated through main.py. This ensures reproducibility and clean execution.

### Phase 1: Hyperparameter Optimization (HPO)

Uses Optuna to find the best learning rate, weight decay, and batch size for each model.

```bash
python main.py --mode hpo
```

* Outputs: Optimal configuration files saved in experiment/phase1_hpo/best_params_[model].yaml.


### Phase 2: Factorial Training

Trains all models dynamically using the parameters found in Phase 1 across multiple statistical seeds (e.g., 42, 123, 2026, 7, 999).

```bash
python main.py --mode phase2
```

* Outputs: Trained weights (.pth), TensorBoard logs, and a general training summary in experiment/phase2_factorial/factorial_results.csv.


### Phase 3: Clinical Evaluation

Loads all trained models and evaluates them on the unseen test set, calculating per-class clinical metrics (Dice, HD95, Sensitivity, Specificity) for NCR, ED, and ET regions.

```bash
python main.py --mode phase3
```

* Outputs: The final statistical dataset ready for R/Python analysis in experiment/phase3_evaluation/anova_clinical_dataset.csv


### 🧹 Utilities: Reset Environment

To ensure a clean slate before a new major experiment, you can securely delete all previous runs (YAMLs, weights, and logs).

```bash
python main.py --mode reset
```


## 3. Running Unit Tests

This repository includes a robust testing suite using mock datasets to verify pipeline integrity without requiring heavy GPU computation.

Run the tests to ensure your environment is set up correctly:

```bash
python -m unittest src/test_hpo_mock.py
python -m unittest src/test_training.py
python -m unittest src/test_phase2_factorial.py
python -m unittest src/test_phase3_evaluate.py
```

## 4. Visualization (TensorBoard)

Track the Loss, IoU, and Dice Score (Fscore) curves in real-time during Phase 2 training. Since each seed has its own folder, you can compare multiple runs simultaneously.


```bash
tensorboard --logdir experiment/phase2_factorial
```

Then, open your browser at http://localhost:6006.



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

