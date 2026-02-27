# Transfer Learning Enables Acute Stroke Severity Prediction in Lesion-Negative Patients Using Clinical MRI (Anonymous Submission)

This repository contains the preprocessing, training, fine-tuning, and inference code used for the experiments described in the associated conference submission.  
All identifiers have been anonymised for double-blind review.

---

## Overview

This repository provides:

- MRI preprocessing pipeline
- Model training and fine-tuning scripts
- Inference utilities
- Configuration files and random seeds used in the submission
- Fine-tuned model weights for all folds and repeats reported in the paper

The core deep learning framework (`aispycore`) was developed specifically for this study. It provides a CLI for model training, evaluation, and inference on 3D MRI data.

---

## Repository Structure

```
.
├── preprocess_soop.py              # MRI preprocessing script
├── aispycore/                      # Core deep learning package
│   └── cli/
│       └── train.py                # Model training CLI
├── model_training/
│   └── scripts/                    # Bash scripts, configs and seeds used in paper
├── trained_models/
│   └── finetuned_Dementia_SFCN/    # Fine-tuned weights for all folds and repeats
└── README.md
```

---

## Installation

### Requirements

- Python 3.9
- A CUDA-capable GPU (recommended)

### Python Environment

We recommend creating a conda environment:

```bash
conda create -n aispycore python=3.9.15
conda activate aispycore
```

### Dependencies

Install the required packages:

```bash
pip install tensorflow==2.6.2 \
            keras==2.6.0 \
            numpy==1.19.5 \
            pandas==1.4.4 \
            scikit-learn==1.6.1 \
            nibabel==5.1.0 \
            iterative-stratification==0.1.9
```

**pyment is also required.** The SFCN backbone architecture and pretrained Healthy Brain Age and Dementia model weights used in this work are based on the implementations provided by Leonardsen et al. (2024). Install pyment from the official repository before using `aispycore`:

```
`https://github.com/estenhl/pyment-public/tree/main`
```

**Citation:**  
`Leonardsen, E.H. et al.: Constructing personalized characterizations of structural brain aber-rations in patients with dementia using explainable artificial intelli-gence. npj Digit. Med. 7, 1, 110 (2024). https://doi.org/10.1038/s41746-024-01123-7.`

Then add `aispycore` and `pyment` to sys path in the predict.py and train.py scripts before importing aispy or pyment: 
```sys.path.insert(0, "/path/to/package/folder")```

### Verified Package Versions

| Package | Version |
|---|---|
| tensorflow | 2.6.2 |
| keras | 2.6.0 |
| numpy | 1.19.5 |
| pandas | 1.4.4 |
| scikit-learn | 1.6.1 |
| nibabel | 5.1.0 |
| pyment | 2.0 |
| Python | 3.9.15 |
| iterative-stratification | 0.1.9 |

---

## Data

> **Note:** No participant data is included in this repository. All datasets used are publicly available and must be downloaded independently.

### Stroke Dataset

Model training and evaluation were performed using data from the **Stroke Outcome Optimization Project (SOOP)**, a publicly available acute stroke clinical dataset. Raw T1-weighted MRI scans in native space must be downloaded from the official SOOP repository on OpenNeuro and placed locally before preprocessing.

**Dataset link:**  
`https://openneuro.org/datasets/ds004889/versions/1.1.2`

Lesion volume and regional lesion load values used in this study are available in tabular format from the SOOP Project Demo Repository:

**Lesion data link:**  
`https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo/tree/main`


**Dataset and Lesion data citation:**

`Absher, J. et al.: The stroke outcome optimization project: Acute ischemic strokes from a comprehensive stroke center. Sci Data. 11, 1, 839 (2024). https://doi.org/10.1038/s41597-024-03667-5.`

---

### Healthy Control Dataset

Healthy controls used in this study were obtained from the publicly available IXI dataset.

**IXI dataset link:**  
`https://brain-development.org/ixi-dataset/`

**IXI dataset citation:**  
`IXI Dataset – Brain Development, https://brain-development.org/ixi-dataset/, last accessed 2025/01/05.`

---

## Preprocessing

Preprocessing is performed using `preprocess_soop.py`, which handles skull stripping, registration, and any other processing steps applied prior to model input.

### Arguments

| Argument | Type | Description |
|---|---|---|
| `--batch_file` | `str` | Path to a plain text file where each line is an absolute path to a T1-weighted MRI scan in native space |
| `--max_images` | `int` | Maximum number of images to process before terminating the script (useful for debugging or batched cluster jobs) |
| `--num_threads` | `int` | Number of parallel threads to use for preprocessing |

### Usage

```bash
python preprocess_soop.py \
    --batch_file /path/to/scan_paths.txt \
    --max_images 100 \
    --num_threads 8
```

The `batch_file` should be formatted as follows, with one scan path per line:

```
/data/soop/sub-1/anat/sub-1_T1w.nii.gz
/data/soop/sub-2/anat/sub-2_T1w.nii.gz
...
```

---

## Data Preparation

After preprocessing, a dataset CSV must be created before running training or inference. This is done using `aispycore/data/prepare_mRS_data.py`, which joins preprocessed image paths with participant metadata and applies a binarisation threshold to the mRS outcome score. 

aispycore/data/prepare_mRS_data.py expects an input .tsv file containing participant demographic and clinical data. In the OpenNeuro repository this is named 'participants.tsv'. Additionally, lesion volume from the lesion data in the Project Demo is expected. This is contained in the 'artery_cleaned.tsv' file of the SOOP Demo repository.

### Arguments

| Argument | Type | Description |
|---|---|---|
| `--image_dir` | `str` | Directory containing the preprocessed T1w MRI files |
| `--participant_data_file` | `str` | Path to the participant metadata file (TSV), containing mRS scores and other clinical variables |
| `--participant_ids_file` | `str` | Path to a plain text file listing the participant IDs to include, one per line (e.g. a filtered subset passing quality control) |
| `--output_csv` | `str` | Path where the output CSV will be written |
| `--threshold` | `int` | mRS binarisation threshold. Participants with mRS ≥ threshold are assigned label `1` (poor outcome); those below are assigned label `0` (good outcome) |

### Usage
```bash
python prepare_mRS_data.py \
    --image_dir /path/to/preprocessed_images/ \
    --participant_data_file /path/to/SOOP_Participants.tsv \
    --participant_ids_file /path/to/included_participant_ids.txt \
    --output_csv /path/to/dataset.csv \
    --threshold 1
```

The output CSV contains one row per participant and includes columns for participant ID, image path, and binarised mRS label. This file is passed directly to `train.py` and the inference scripts via the `--csv_file` argument.


## Model Training

Training is performed using the `aispycore` CLI:

```bash
python aispycore/cli/train.py --config <config.yaml> [options]
```

Example bash scripts, the exact configuration files, and the random seeds used for all experiments in the paper are provided in `model_training/scripts/`. **File paths within these scripts will need to be updated to reflect your local environment before running.**

Refer to the scripts directory for the full set of arguments. Key options include specifying the dataset CSV, output directory, model architecture, and the repeat/fold index for nested cross-validation.

---

## Fine-tuning and Inference

Scripts for running fine-tuning and inference across all repeats and folds are provided in `model_training/scripts/`, alongside the configuration files and seeds used to produce the results reported in the paper.

As with training, **file paths in these scripts must be updated** to match your local data and output directories before running.

---

## Pretrained and Fine-tuned Weights

### Pretrained Backbone

The SFCN backbone pretrained on brain age (used for correlational analysis in the current study) and dementia diagnosis (used as initialisation for fine-tuning) is sourced from the pyment package (Leonardsen et al., 2024):

```
`https://github.com/estenhl/pyment-public/tree/main`
```

### Fine-tuned Weights

Fine-tuned model weights for every fold and repeat reported in the paper are provided in this repository under:

```
trained_models/finetuned_Dementia_SFCN/
```

Weights are organised by repeat and fold, matching the directory structure produced by `train.py`. These can be used directly with the inference scripts in `model_training/scripts/` to reproduce the results in the paper.

---
