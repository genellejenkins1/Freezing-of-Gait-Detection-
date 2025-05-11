"""
CNNModelEvaluation.py
----------------------
Evaluates multiple trained CNN models using shared validation sets. Outputs metrics
and visualizations including ROC, PR curves, confusion matrices, and bootstrapped CIs.

Usage:
    python CNNModelEvaluation.py --config_file eval_config.json --output_dir model_evaluation_results

Author: Genelle Jenkins
Version: 1.0
"""

import os
import json
import random
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score, f1_score
)
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

# Constants
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 192, 192, 3
FOG_LABEL_MAPPING = {'1': 1, '0': 0}
BATCH_SIZE = 32

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# === Data Generator ===
class MultiBranchDataGenerator(Sequence):
    def __init__(self, image_dir, batch_size=BATCH_SIZE):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_paths, self.labels = self._load_image_paths_and_labels()
        self.on_epoch_end()

    def _load_image_paths_and_labels(self):
        image_paths, labels = [], []
        for session in os.listdir(self.image_dir):
            session_path = os.path.join(self.image_dir, session)
            if not os.path.isdir(session_path):
                continue
            for axis in ["ankle", "hip", "thigh"]:
                axis_path = os.path.join(session_path, axis)
                if not os.path.isdir(axis_path):
                    continue
                for file in sorted(os.listdir(axis_path)):
                    if file.endswith(".png"):
                        label_str = file.split("_")[-1].split(".")[0]
                        label = FOG_LABEL_MAPPING.get(label_str)
                        if label is not None:
                            image_paths.append(os.path.join(axis_path, file))
                            labels.append(label)
        return image_paths, labels

    def __len__(self):
        return max(1, len(self.image_paths) // (self.batch_size * 3))

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            j = idx * self.batch_size * 3 + i * 3
            if j + 2 >= len(self.image_paths):
                continue
            x = self._load_image(self.image_paths[j])
            y = self._load_image(self.image_paths[j + 1])
            z = self._load_image(self.image_paths[j + 2])
            batch_x.append([x, y, z])
            batch_y.append(self.labels[j])
        return {
            'input_x': tf.convert_to_tensor([x[0] for x in batch_x], dtype=tf.float32),
            'input_y': tf.convert_to_tensor([x[1] for x in batch_x], dtype=tf.float32),
            'input_z': tf.convert_to_tensor([x[2] for x in batch_x], dtype=tf.float32)
        }, tf.convert_to_tensor(batch_y, dtype=tf.float32)

    def _load_image(self, path):
        img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        return img_to_array(img) / 255.0

    def on_epoch_end(self):
        zipped = list(zip(self.image_paths, self.labels))
        random.shuffle(zipped)
        self.image_paths, self.labels = zip(*zipped)


# === Evaluation Utilities ===
def save_plot(fig, path): fig.savefig(path); plt.close(fig)

def plot_confusion_matrix(cm, out_dir, label):
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-FOG", "FOG"], yticklabels=["Non-FOG", "FOG"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{label} Confusion Matrix")
    save_plot(fig, os.path.join(out_dir, f"{label}_ConfusionMatrix.png"))

def plot_roc(y_true, y_pred, out_dir, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_val = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{label} ROC Curve")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend()
    save_plot(fig, os.path.join(out_dir, f"{label}_ROC.png"))

def plot_precision_recall(y_true, y_pred, out_dir, label):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    fig = plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{label} Precision-Recall Curve"); plt.legend()
    save_plot(fig, os.path.join(out_dir, f"{label}_PR_Curve.png"))

def misclassified_images(generator, y_true, y_pred, out_dir, label):
    wrong_indices = [i for i, (t, p) in enumerate(zip(y_true, np.round(y_pred))) if t != p]
    if not wrong_indices:
        return
    sample_wrong = random.sample(wrong_indices, min(9, len(wrong_indices)))
    fig = plt.figure(figsize=(10, 10))
    for i, idx in enumerate(sample_wrong):
        path = generator.image_paths[idx * 3]
        img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {int(y_true[idx])}, Pred: {int(np.round(y_pred[idx]))}")
        plt.axis("off")
    plt.suptitle(f"{label} Misclassified GAF Images")
    plt.tight_layout()
    save_plot(fig, os.path.join(out_dir, f"{label}_Misclassified.png"))

def bootstrap_ci(y_true, y_pred, metric_fn, n=1000):
    scores = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        scores.append(metric_fn(np.array(y_true)[idx], np.round(np.array(y_pred)[idx])))
    return np.percentile(scores, [2.5, 97.5])


# === Evaluation Loop ===
def evaluate_models(eval_config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, paths in eval_config.items():
        label = name.replace("_", " ").title()
        val_dir = paths["val_dir"]
        model_path = paths["model_path"]
        out_dir = os.path.join(output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        logging.info(f"🔍 Evaluating: {label}")
        model = load_model(model_path, compile=False)
        generator = MultiBranchDataGenerator(val_dir)

        y_true, y_pred = [], []
        for i in range(len(generator)):
            x_batch, y_batch = generator[i]
            preds = model.predict(x_batch, verbose=0).flatten()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        class_counts = Counter(y_true)
        logging.info(f"{label} class distribution: {class_counts}")

        report_txt = classification_report(y_true, np.round(y_pred), target_names=["Non-FOG", "FOG"])
        with open(os.path.join(out_dir, "classification_report.txt"), 'w') as f:
            f.write(report_txt)

        cm = confusion_matrix(y_true, np.round(y_pred))
        plot_confusion_matrix(cm, out_dir, label)
        plot_roc(y_true, y_pred, out_dir, label)
        plot_precision_recall(y_true, y_pred, out_dir, label)
        misclassified_images(generator, y_true, y_pred, out_dir, label)

        acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
        f1_ci = bootstrap_ci(y_true, y_pred, f1_score)
        logging.info(f"{label} Accuracy CI: {acc_ci}")
        logging.info(f"{label} F1 CI: {f1_ci}")
        with open(os.path.join(out_dir, "confidence_intervals.txt"), 'w') as f:
            f.write(f"Accuracy 95% CI: {acc_ci}\nF1-score 95% CI: {f1_ci}\n")


# === Main Entry ===
def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple CNN models on shared validation sets")
    parser.add_argument("--config_file", required=True, help="Path to JSON with model paths and val dirs")
    parser.add_argument("--output_dir", required=True, help="Directory to save all evaluation outputs")
    args = parser.parse_args()

    with open(args.config_file) as f:
        eval_config = json.load(f)

    evaluate_models(eval_config, args.output_dir)

if __name__ == "__main__":
    main()
