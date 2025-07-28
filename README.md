# Leukemia Detection Project

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://claude.ai/chat/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive deep learning framework for leukemia detection in whole slide images (WSI), featuring both segmentation and classification models for automated analysis of bone marrow samples.

## 🎯 Motivation

Accurate cell segmentation and classification in microscopy images are fundamental for advancing biomedical research and clinical diagnostics. Tasks such as identifying pathological markers for disease diagnosis and quantifying cellular responses are paramount. Hematological analysis, for instance, relies heavily on morphological examination of blood and bone marrow cells for diagnosing and monitoring conditions like leukemia, where identifying and quantifying specific cell types (e.g., blast cells) is critical for patient management.

The emergence of digital pathology and high-throughput microscopy generates vast quantities of image data, creating both opportunities and challenges for automated analysis. While manual analysis by expert pathologists is the gold standard, it is labor-intensive, time-consuming, and subject to inter-observer variability. This project is motivated by the critical need for robust, accurate, and efficient automated methods for cellular image analysis, specifically to bridge the gap in effectively applying powerful foundation models like CellSAM to specialized biomedical imaging tasks.

## 🔬 Overview

This project implements state-of-the-art deep learning approaches for:

* **Cell Segmentation** : Automated detection and segmentation of individual cells using CellSAM
* **Cell Classification** : Binary classification of CD34+ and CD34- myeloblasts
* **Annotation Processing** : Tools for converting between different annotation formats
* **Data Pipeline** : Complete preprocessing pipeline from WSI to training-ready datasets

## 🚀 Features

* **WSI Processing** : Extract patches from whole slide images with configurable parameters
* **Annotation Handling** : Process QuPath annotations and GeoJSON files
* **Cell Segmentation** : Segment individual cells using using deep learning models
* **Cell Classification** : Classify cells as positive/negative for CD34 marker
* **Data Pipeline** : Complete pipeline from raw WSI to training-ready datasets
* **Advanced Preprocessing** : Color deconvolution, pseudo-RGB channels creation, patch extraction, and quality filtering
* **Visualization** : Interactive visualization tools for annotations and predictions
* Support for multiple staining types (H&E, IHC)
* Comprehensive evaluation metrics (Deepcell and Coco)
* Supports QuPath annotations and GeoJSON formats

## 📁 Project Structure

```
leukemia-detection/
├── src/                          # Core library code
│   ├── classification/           # Classification models and training
│   ├── segmentation/            # Segmentation models and training
│   ├── data/                    # Data processing modules
│   ├── common/                  # Shared constants and exceptions
│   └── visualization/           # Plotting and visualization tools
├── scripts/                     # Command-line utilities
├── configs/                     # Configuration files
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit and integration tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
└── README.md                    # This file
```

## 🛠️ Installation

### Quick Install (pip)

The easiest way to get started is to install directly from GitHub:

```bash
pip install git+https://github.com/danikvh/leukemia-detection.git
```


## 📊 Results

This section presents key quantitative results and illustrative examples from the project, demonstrating the performance of the CellSAM model under various conditions and preprocessing strategies.

### Visualizing Color Deconvolution

Effective preprocessing, especially color deconvolution for stained images, is crucial for CellSAM's performance. The figure below illustrates different preprocessing strategies applied to a representative TNBC-Extended dataset image:

![Project Logo](docs/images/color_deconvolution.png)

**Figure Description:** Visual comparison of preprocessing strategies applied to a representative TNBC-Extended dataset image. (a) Original RGB input. (b) Image preprocessed using a simplified approach, converting the input to be primarily represented in the Blue channel. (c) Image after color deconvolution, where only the Hematoxylin component is utilized and assigned to the Blue channel. (d) Image after color deconvolution, with the Hematoxylin component assigned to the Green channel and the DAB component to the Blue channel (IHC stain). (e) Image after color deconvolution, with the Hematoxylin component assigned to the Green channel and the Eosin component to the Blue channel (H&E stain). [Page 49]

### Key Quantitative Results

The following tables summarize the segmentation performance of the CellSAM model across different stages of the project, evaluated using metrics such as Precision, Recall, F1-score, Dice Similarity Coefficient, Jaccard Index, and Average Precision at 0.5 IoU (AP50).

#### Table 1: Performance Metrics of Pre-trained CellSAM Model

An initial evaluation of selected preprocessing strategies on a subset of the TNBC-Extended dataset using the pre-trained CellSAM model. The Nuclear-Only approach yielded the best performance among the tested configurations, outperforming even the default CellSAM preprocessing.

![img](docs/images/results_preeval.png)

#### Table 2: Segmentation Performance of CellSAM Fine-tuned (Nuclear-Only Preprocessing Pipeline)

Results from fine-tuning CellSAM on the TNBC-Extended dataset using the Nuclear-Only preprocessing pipeline. The Nuclear-Only preprocessing, when combined with a weighted OHEM approach during Stage 2 neck training, consistently yielded the best segmentation performance.

![img](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Dani/Documents/Personal/Informatica/MUIARFID/TFM/leukemia_detection/docs/images/results_nuclearonly.png)

#### Table 3: Segmentation Performance of CellSAM Fine-tuned (H&E-Specific Two-Channel Color Deconvolution)

Results from fine-tuning CellSAM on the TNBC-Extended dataset using H&E-specific two-channel color deconvolution. Complex data augmentation yielded markedly better results for this richer input representation.

![](docs/images/results_ihc.png)

#### Table 4: Segmentation Performance of CellSAM Fine-tuned (IHC-Specific Two-Channel Color Deconvolution)

Results from fine-tuning CellSAM on the TNBC-Extended dataset using IHC-specific two-channel color deconvolution. This approach generally outperformed the H&E-specific method and provided a stronger starting point after baseline fine-tuning.

![](docs/images/results_he.png)
