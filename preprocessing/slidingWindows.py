"""
slidingWindows.py
-----------------
Extracts non-overlapping windows of accelerometer data labeled as FOG or non-FOG
based on FOG annotations. Windows are saved as .pkl files and optionally logged
in summary text files.

Usage:
    python slidingWindows.py --data_dir path/to/raw_dataset --output_dir path/to/save_windows --write_summary

Author: Genelle Jenkins
Version: 1.0
"""

import os
import numpy as np
import pickle
import argparse
import logging

# Constants
WINDOW_SIZE = 192  # 3 seconds at 64 Hz
FOG_THRESHOLD = 0.5  # If >50% of window contains FOG, label it as FOG

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_and_clean_data(file_path):
    """
    Loads the raw data and removes rows with non-experimental markers (label == 0).
    """
    data = np.loadtxt(file_path)
    cleaned_data = data[data[:, -1] != 0]
    return cleaned_data

def create_windows(data, accel_columns):
    """
    Creates non-overlapping sliding windows from accelerometer columns
    and labels them based on FOG annotation.
    """
    windows = []

    for start in range(0, len(data), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        window_data = data[start:end]

        if len(window_data) < WINDOW_SIZE:
            is_complete = False
        else:
            is_complete = True

        accel_data = window_data[:, accel_columns]
        fog_annotation = window_data[:, -1]

        combined = np.column_stack((accel_data, fog_annotation))

        fog_ratio = np.mean(fog_annotation == 2)
        non_fog_ratio = np.mean(fog_annotation == 1)

        if fog_ratio == 1.0 or fog_ratio > FOG_THRESHOLD:
            label = "fog"
        elif non_fog_ratio == 1.0:
            label = "non_fog"
        elif non_fog_ratio > FOG_THRESHOLD:
            continue
        else:
            continue

        windows.append((combined, label, is_complete))

    return windows

def save_windows(windows, output_dir, accelerometer, write_summary):
    """
    Saves each window as a .pkl file and optionally writes a summary report.
    """
    os.makedirs(output_dir, exist_ok=True)
    fog_count = 0
    non_fog_count = 0
    incomplete = []

    for i, (window, label, is_complete) in enumerate(windows):
        filename = f"window_{i}_{label}"
        if not is_complete:
            filename += "_incomplete"
            incomplete.append(filename)

        with open(os.path.join(output_dir, f"{filename}.pkl"), 'wb') as f:
            pickle.dump(window, f)

        if label == "fog":
            fog_count += 1
        else:
            non_fog_count += 1

    if write_summary:
        summary_file = os.path.join(output_dir, f"{accelerometer}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Summary for {accelerometer}:\n")
            f.write(f"Total FOG windows: {fog_count}\n")
            f.write(f"Total non-FOG windows: {non_fog_count}\n")
            if incomplete:
                f.write("Incomplete windows:\n")
                for name in incomplete:
                    f.write(f"{name}\n")

def process_file(file_path, output_dir, write_summary):
    """
    Handles all accelerometers for a single patient-trial file.
    """
    logging.info(f"Processing {os.path.basename(file_path)}")
    data = load_and_clean_data(file_path)
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    file_output_dir = os.path.join(output_dir, file_id)
    os.makedirs(file_output_dir, exist_ok=True)

    accelerometers = {
        "ankle": [1, 2, 3],
        "thigh": [4, 5, 6],
        "hip": [7, 8, 9]
    }

    for name, cols in accelerometers.items():
        accel_output_dir = os.path.join(file_output_dir, name)
        windows = create_windows(data, cols)
        save_windows(windows, accel_output_dir, name, write_summary)

def main():
    parser = argparse.ArgumentParser(description="Segment accelerometer data into labeled windows")
    parser.add_argument("--data_dir", required=True, help="Path to the folder with raw .txt files")
    parser.add_argument("--output_dir", required=True, help="Where to save the windowed .pkl files")
    parser.add_argument("--write_summary", action="store_true", help="If set, writes a summary text file per sensor")
    args = parser.parse_args()

    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.txt')]
    for file_name in all_files:
        file_path = os.path.join(args.data_dir, file_name)
        process_file(file_path, args.output_dir, args.write_summary)

if __name__ == "__main__":
    main()
