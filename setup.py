#!/usr/bin/env python3
"""
Setup script for the Leukemia Detection project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')

setup(
    name="leukemia-detection",
    version="0.1.0",
    author="Daniel Kyu Vicente Hungerbuhler",
    author_email="danivicen@gmail.com",
    description="Deep learning models for leukemia detection in whole slide images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danikvh/leukemia-detection",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include data files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "extract-patches=scripts.extract_patches:main",
            "convert-annotations=scripts.convert_annotations:main",
            "train-segmentation=src.segmentation.training.experiment_runner:main",
            "train-classification=src.classification.trainer:main",
            "leukemia-inference=scripts.inference:main",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discoverability
    keywords=[
        "leukemia", "detection", "deep-learning", "medical-imaging",
        "whole-slide-images", "pathology", "segmentation", "classification",
        "computer-vision", "healthcare", "ai"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/leukemia-detection/issues",
        "Source": "https://github.com/yourusername/leukemia-detection",
        "Documentation": "https://leukemia-detection.readthedocs.io/",
    },
)