# Freezing of Gait Detection Pipeline (Parkinson’s Disease)

This repository contains a complete pipeline for detecting Freezing of Gait (FOG) using tri-axial accelerometer data from wearable sensors. The pipeline includes preprocessing, augmentation, CNN model training, and model evaluation.

---

## 📌 Project Overview

- **Sliding Window Segmentation** of raw accelerometer signals
- **Gramian Angular Field (GAF) Conversion** of time-series windows
- **Balanced Dataset Creation** via upsampling, downsampling, or hybrid
- **Multi-branch CNN Training** for FOG classification
- **Evaluation & Visualization** of classification results across models

---

## 🔧 Setup

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/FreezingOfGaitPipeline.git
cd FreezingOfGaitPipeline
pip install -r requirements.txt
```

### 2. Folder Structure
- Place raw accelerometer `.txt` files in a directory like `data/raw/`
- Outputs will be saved in user-defined locations during script execution

---

## 🧩 Pipeline Stages

### 1. Sliding Window Segmentation
```bash
python preprocessing/slidingWindows.py --data_dir data/raw --output_dir data/windows --write_summary
```

### 2. Convert to GAF Images
```bash
python preprocessing/slidingWindowGAFimages.py --pickle_dir data/windows --output_dir data/gaf
```

### 3. Generate Augmented Dataset (Choose One)

- **Upsampling** (for FOG minority class):
```bash
python preprocessing/upSamplingFogEvents.py --source_dir data/gaf --output_dir data/gaf_upsampled --augmentations_per_sample 2
```

- **Downsampling** (to match FOG prevalence):
```bash
python preprocessing/downSamplingNonFogEvents.py --input_dir data/gaf --output_dir data/gaf_downsampled --train_split 0.8
```

- **Hybrid Dataset** (60% non-FOG, 40% FOG):
```bash
python preprocessing/slidingWindowGAFImages_Hybrid.py --input_dir data/gaf --output_dir data/gaf_hybrid --ratio_nonfog 0.6 --ratio_fog 0.4 --train_split 0.85
```

---

## 🧠 CNN Model Training

### For GAF Images (downsampled or hybrid):
```bash
python training/CNNModel4.py --data_dir data/gaf_hybrid --save_dir outputs/hybrid_model --epochs 60 --batch_size 32 --model_name fog_cnn_model.h5
```

### For Preprocessed `.npz` Files (upsampled):
```bash
python training/CNNModel5.py --data_dir data/npz --save_dir outputs/upsampled_model --model_name fog_cnn_model_streamed.h5
```

---

## 📊 Model Evaluation

Evaluate all models using shared validation structure:
```bash
python evaluation/CNNModelEvaluation.py --config_file eval_config.json --output_dir outputs/evaluation
```

### Sample `eval_config.json`:
```json
{
  "upsampled": {
    "model_path": "outputs/upsampled_model/fog_cnn_model_streamed.h5",
    "val_dir": "data/gaf_upsampled/valid"
  },
  "downsampled": {
    "model_path": "outputs/downsampled_model/fog_cnn_model.h5",
    "val_dir": "data/gaf_downsampled/valid"
  },
  "hybrid": {
    "model_path": "outputs/hybrid_model/fog_cnn_model.h5",
    "val_dir": "data/gaf_hybrid/valid"
  }
}
```

---

## ✅ Output Includes

- `classification_report.txt`
- `confusion_matrix.npy` + plotted heatmap
- `roc_curve.png`, `precision_recall_curve.png`
- Bootstrapped confidence intervals
- Misclassified image grid

---

## 👩‍💻 Author

**Genelle Jenkins**  
M.S. Biomedical Informatics & Data Science  
Capstone Project, Spring 2025
