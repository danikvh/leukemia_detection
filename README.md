# Leukemia Detection Project

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://claude.ai/chat/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive deep learning framework for leukemia detection in whole slide images (WSI), featuring both segmentation and classification models for automated analysis of bone marrow samples.

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
