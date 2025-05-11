"""
slidingWindowGAFimages.py
--------------------------
Converts sliding window .pkl files into Gramian Angular Field (GAF) images
for each axis (X, Y, Z), stored in a mirrored folder structure.

Usage:
    python slidingWindowGAFimages.py --pickle_dir path/to/slidingWindows --output_dir path/to/gafImages

Author: Genelle Jenkins
Version: 1.0
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import logging

# GAF and image dimensions
IMG_HEIGHT, IMG_WIDTH = 192, 450
GAF_TRANSFORMER = GramianAngularField(image_size=IMG_HEIGHT, method="summation")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def process_pickle_to_gaf(pickle_path, output_dir, file_prefix, label_code):
    """
    Converts a single .pkl window into three GAF images (X, Y, Z) and saves them.
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        if data.shape[0] != 192 or data.shape[1] < 3:
            logging.warning(f"Skipping {pickle_path} due to shape {data.shape}")
            return

        axes = data[:, :3].T  # Split into x, y, z

        for axis_name, axis_data in zip(['x', 'y', 'z'], axes):
            gaf = GAF_TRANSFORMER.fit_transform(axis_data.reshape(1, -1))[0]
            filename = f"{file_prefix}_{axis_name}_{label_code}.png"
            save_path = os.path.join(output_dir, filename)

            plt.figure(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100))
            plt.imshow(gaf, cmap='viridis', aspect='auto')
            plt.axis('off')
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

    except Exception as e:
        logging.error(f"Error processing {pickle_path}: {e}")

def process_all_pickles(pickle_dir, output_dir):
    """
    Traverses patient and accelerometer directories, processes each .pkl file.
    """
    for patient in os.listdir(pickle_dir):
        patient_path = os.path.join(pickle_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for accel in ["ankle", "hip", "thigh"]:
            accel_path = os.path.join(patient_path, accel)
            if not os.path.isdir(accel_path):
                continue

            output_accel_path = os.path.join(output_dir, patient, accel)
            os.makedirs(output_accel_path, exist_ok=True)

            for file in sorted(os.listdir(accel_path)):
                if not file.endswith(".pkl") or "summary" in file or "_incomplete" in file:
                    continue

                label_code = 0 if "non_fog" in file else 1 if "fog" in file else None
                if label_code is None:
                    logging.warning(f"Unknown label in {file}, skipping")
                    continue

                window_id = file.split('_')[1]
                prefix = f"{patient}_{accel}_w{window_id}"
                process_pickle_to_gaf(os.path.join(accel_path, file), output_accel_path, prefix, label_code)

def main():
    parser = argparse.ArgumentParser(description="Convert sliding window .pkl files to GAF images.")
    parser.add_argument('--pickle_dir', required=True, help='Path to folder containing sliding window .pkl files')
    parser.add_argument('--output_dir', required=True, help='Where to save the GAF images')
    args = parser.parse_args()

    process_all_pickles(args.pickle_dir, args.output_dir)

if __name__ == "__main__":
    main()

