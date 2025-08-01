# Data Processing Module

This module provides comprehensive tools for processing Whole Slide Images (WSI) and annotations for leukemia detection and cell analysis. It handles extraction of annotations from QuPath projects, patch extraction from WSI files, mask-to-annotation conversion, and GeoJSON processing.

## 🏗️ Architecture Overview

The data processing pipeline consists of several interconnected components:

```
WSI Files (.svs) + GeoJSON Annotations
           ↓
    AnnotationExtractor
           ↓
    Individual Cell Crops + ROI Patches
           ↓
    WSIProcessor (Patch Extraction)
           ↓
    MaskToAnnotationConverter
```

## 📁 Module Structure

```
data/
├── annotation_extractor.py    # Extract annotations and cell crops from WSI
├── config.py                  # Configuration settings
├── mask_converter.py          # Convert masks to GeoJSON annotations
└── wsi_processor.py          # WSI patch extraction and processing
```

## 🔧 Core Components

### AnnotationExtractor

Extracts annotations from WSI files and GeoJSON files, creating masks and individual cell crops for training.

**Key Features:**

* Extract cell crops with exact mask shapes or rectangular bounds
* Support for both Spanish and English labels ("positivo"/"negativo" → "positive"/"negative")
* ROI-based processing with overlay visualization
* Configurable cell crop padding and background values
* Automatic label CSV generation

**Usage:**

```python
from data.annotation_extractor import AnnotationExtractor

extractor = AnnotationExtractor()
extractor.extract_annotations(
    svs_path="path/to/slide.svs",
    geojson_path="path/to/annotations.geojson",
    output_dir="./output",
    image_name="slide_001",
    cell_crop_padding=10,
    use_mask_shape=True  # Extract cells with exact shape
)
```

### WSIProcessor

Processes Whole Slide Images for patch extraction and annotation handling.

**Key Features:**

* Grid-based or ROI-centered patch extraction
* Quality filtering (skip white/empty patches)
* Model-based mask generation support
* Metadata preservation in PNG files
* Multi-level pyramid support

**Usage:**

```python
from data.wsi_processor import WSIProcessor

processor = WSIProcessor()
processor.extract_patches(
    svs_file="path/to/slide.svs",
    output_dir="./patches",
    qp_img=qupath_image,
    patch_size=512,
    stride=384,
    generate_mask=True,
    use_rois=False
)
```

### MaskToAnnotationConverter

Converts binary masks back to GeoJSON annotations with overlap removal and border filtering.

**Key Features:**

* Convert instance masks to polygon annotations
* Intelligent overlap filtering (keeps larger polygons)
* Border margin filtering to avoid edge artifacts
* Spatial indexing for efficient processing
* Configurable quality thresholds

**Usage:**

```python
from data.mask_converter import MaskToAnnotationConverter

converter = MaskToAnnotationConverter()
converter.convert_masks_to_geojson(
    output_dir="./model_output",
    filename="predictions",
    classification_name="CellSAM Mask"
)
```

## ⚙️ Configuration

The `DataProcessingConfig` class centralizes all configuration parameters:

```python
from data.config import DataProcessingConfig

config = DataProcessingConfig()
config.patch_size = 512
config.stride = 384
config.border_margin_ratio = 0.1
config.overlap_threshold = 0.85
config.min_polygon_area = 10.0
config.white_threshold = 240
```

### Key Configuration Parameters

| Parameter               | Description                          | Default |
| ----------------------- | ------------------------------------ | ------- |
| `patch_size`          | Size of extracted patches            | 512     |
| `stride`              | Stride between patches               | 384     |
| `border_margin_ratio` | Margin ratio for border filtering    | 0.1     |
| `overlap_threshold`   | Threshold for overlap removal        | 0.85    |
| `min_polygon_area`    | Minimum area for valid polygons      | 10.0    |
| `white_threshold`     | Threshold for white patch filtering  | 240     |
| `bbox_threshold`      | Threshold for bounding box detection | 0.19    |

## 📊 Output Structure

The processing pipeline generates organized output directories:

```
output/
├── imgs/                     # ROI patches
├── masks_png/               # Binary masks (PNG)
├── masks_npy/               # Instance masks (NumPy)
├── overlay/                 # Visualization overlays
├── cells/                   # Individual cell crops
└── {image_name}_cell_labels.csv  # Cell labels
```

For WSI patch extraction:

```
patches/
├── images/                  # Extracted patches
├── overlay/                 # Patch overlays
└── masks/
    ├── img/                # Mask images
    └── data/               # Mask arrays (.npy)
```

## 🔍 Quality Control Features

### Patch Quality Filtering

* **White patch detection** : Skip patches with mean intensity > threshold
* **Variance filtering** : Remove uniform/low-contrast patches
* **Border filtering** : Exclude patches near slide edges

### Annotation Validation

* **Geometry validation** : Check polygon validity and closure
* **Area filtering** : Remove annotations below minimum area
* **Overlap detection** : Identify and resolve overlapping annotations
* **Coordinate validation** : Ensure valid numeric coordinates

### Data Integrity

* **Metadata preservation** : Store patch coordinates in PNG metadata
* **Label mapping** : Consistent Spanish-to-English label conversion
* **Error handling** : Robust error handling with detailed logging

## 📈 Performance Optimizations

### Memory Management

* **Lazy loading** : Load images only when needed
* **Batch processing** : Process multiple patches efficiently
* **Memory limits** : Configurable memory usage limits

### Spatial Operations

* **STRtree indexing** : Fast spatial queries for overlap detection
* **Vectorized operations** : NumPy-based array operations
* **Parallel processing** : Multi-processing support for large datasets

## 🛠️ Dependencies

```python
# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0
shapely>=1.8.0
pandas>=1.3.0
openslide-python>=1.1.2

# Optional dependencies
matplotlib>=3.4.0  # For visualization
torch>=1.9.0       # For model-based mask generation
```

## 🚀 Usage Examples

### Complete Pipeline Example

```python
from data.annotation_extractor import AnnotationExtractor
from data.wsi_processor import WSIProcessor
from data.config import DataProcessingConfig

# Configure processing
config = DataProcessingConfig()
config.patch_size = 512
config.stride = 384
config.use_mask_shape = True

# Extract annotations and cell crops
extractor = AnnotationExtractor(config)
extractor.extract_annotations(
    svs_path="slide.svs",
    geojson_path="annotations.geojson",
    output_dir="./training_data",
    image_name="slide_001",
    cell_crop_padding=15,
    background_value=255
)

# Process WSI for patch extraction
processor = WSIProcessor(config)
processor.extract_patches(
    svs_file="slide.svs",
    output_dir="./patches",
    qp_img=qupath_image,
    patch_size=512,
    generate_mask=False,
    use_rois=True
)
```

### Batch Processing Multiple Files

```python
import glob
from data.annotation_extractor import merge_cell_label_csvs

# Process multiple WSI files
svs_files = glob.glob("data/slides/*.svs")
for svs_file in svs_files:
    geojson_file = svs_file.replace(".svs", ".geojson")
    if os.path.exists(geojson_file):
        image_name = os.path.basename(svs_file).replace(".svs", "")
        extractor.extract_annotations(
            svs_path=svs_file,
            geojson_path=geojson_file,
            output_dir="./batch_output",
            image_name=image_name
        )

# Merge all cell labels into single CSV
merge_cell_label_csvs("./batch_output", "all_cell_labels.csv")
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenSlide Installation** : Ensure OpenSlide library is properly installed
2. **Memory Issues** : Reduce batch size or patch size for large WSI files
3. **Invalid Geometries** : Check GeoJSON format and polygon validity
4. **Missing Dependencies** : Install all required packages

### Logging Configuration

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📝 License

This module is part of a larger leukemia detection project. Please refer to the main project license for usage terms.

---

For more detailed information about specific functions and classes, refer to the inline documentation in each module.
