# Movie Recommandation System
## Project Overview
This repository contains the implementation and experimental evaluation for the **Movie Recommandation System**. 
The goal of this project is to train, evaluate, and compare different data mining / machine learning methods on web-related datasets using standard evaluation metrics.

---

## Group Information
- **Group number:** 11
- **Members:**
  - *Nguyen Viet Anh – 20225434*
  - *Hoang Trung Khai – 20225502*
  - *Trinh Duy Phong – 20220065*
  - *Luu Thien Viet Cuong – 20225477*
  - *Do Dinh Hoang – 20225445*

---
## Dataset
The original dataset used in this project is **MovieLens-1M**.  
After preprocessing, different dataset variants are created depending on the model architecture:

- **DCNv3 dataset (with user and item metadata):**  
  [DCNv3 Dataset](https://drive.google.com/drive/folders/1S69tMoM9Aq1DB1B3cgVcINJM-lj_Uv9D)

- **DeepFM and MLP dataset (ratings-only):**  
  [DeepFM / MLP Dataset](https://drive.google.com/drive/folders/1lIXD206ilaXIwK-NET_v1fOuPy8ij96P?usp=sharing)

All datasets are **preprocessed and already split into training, validation (dev), and test sets**, enabling direct use for model training and evaluation.

## How to train
To train a model, run:

```bash
python train.py \
  --model_type <model_type> \
  --dataset_path <ratings_path> \
  --users_path <users_path> \
  --movies_path <movies_path> \
  --checkpoint_path <checkpoint_path> \
  --lr 0.0005
```
Arguments:
- --model_type: Model architecture to train (mlp/deepfm/dcnv3)
- --dataset_path: Path to the ratings dataset
- --users_path: Path to the user metadata (optional for some models)
- --movies_path: Path to the item metadata (optional for some models)
- --checkpoint_path: Path to save the trained model
- --lr: Learning rate

## Evaluating a Model
After training, evaluate the saved checkpoint using:

```bash
python evaluate.py \
  --model_type <model_type> \
  --dataset_path <ratings_path> \
  --users_path <users_path> \
  --movies_path <movies_path> \
  --checkpoint_path <checkpoint_path> \
  --output_dir <output_dir> \
  --topk 10
```
Arguments:
- --checkpoint_path: Path to the trained model checkpoint
- --output_dir: Directory to save evaluation results
- --topk: Top-K recommendation setting

## Documentation
All documentation and experiment-related materials that are **not directly involved in code execution** are organized in the `docs/` directory.

The directory includes:

- **Evaluation Metrics (`metrics_used.md`)**  
  A markdown file describing all evaluation metrics used in the experiments, along with brief explanations and usage contexts.

- **Performance Results (`performance/`)**  
  A folder storing quantitative results of the evaluated methods.

- **Training Progress (`training_progress/`)**      
  A folder containing visualizations generated during training, including:
  - Training loss curves
  - Validation (dev) loss curves

- **Final Experiment Report (`.pdf`)**  
  A single PDF file containing the complete experiment report, including:
  - Problem formulation
  - Dataset description and preprocessing
  - Evaluated methods
  - Evaluation metrics
  - Experimental results
  - Hyperparameter settings
