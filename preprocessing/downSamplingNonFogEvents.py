"""
downSamplingNonFogEvents.py
----------------------------
Reduces the number of non-FOG GAF triplets to match the FOG count, creating a 
balanced dataset for training and evaluation. Maintains triplet integrity and
preserves original folder structure.

Usage:
    python downSamplingNonFogEvents.py --input_dir path/to/GAFImages --output_dir path/to/DownsampledSet --train_split 0.8

Author: Genelle Jenkins
Version: 1.0
"""

import os
import shutil
import argparse
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def collect_triplet_paths(gaf_dir):
    """
    Collect all valid FOG and non-FOG triplet base paths (prefixes).
    """
    fog, nonfog = [], []
    for root, _, files in os.walk(gaf_dir):
        x_files = [f for f in files if f.endswith("_x_1.png") or f.endswith("_x_0.png")]
        for f in x_files:
            label = "1" if f.endswith("_1.png") else "0"
            prefix = f.replace(f"_x_{label}.png", "")
            base_path = os.path.join(root, prefix)
            if all(os.path.exists(f"{base_path}_{axis}_{label}.png") for axis in ['x', 'y', 'z']):
                (fog if label == "1" else nonfog).append(base_path)
    return fog, nonfog

def split_and_copy_triplets(triplet_list, output_base, split_ratio, label_str):
    """
    Splits triplet list into train/valid and copies to output_base maintaining structure.
    """
    random.shuffle(triplet_list)
    cut = int(len(triplet_list) * split_ratio)
    train, valid = triplet_list[:cut], triplet_list[cut:]

    for subset, name in zip([train, valid], ['train', 'valid']):
        for prefix in subset:
            for axis in ['x', 'y', 'z']:
                src = f"{prefix}_{axis}_{label_str}.png"
                rel_path = os.path.relpath(prefix, start=input_dir)
                dest = os.path.join(output_base, name, rel_path + f"_{axis}_{label_str}.png")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)

def main():
    parser = argparse.ArgumentParser(description="Downsample non-FOG triplets to match FOG count.")
    parser.add_argument("--input_dir", required=True, help="Path to original GAF images")
    parser.add_argument("--output_dir", required=True, help="Where to save downsampled dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    args = parser.parse_args()

    global input_dir
    input_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Gather all triplets
    fog_triplets, nonfog_triplets = collect_triplet_paths(input_dir)
    if not fog_triplets:
        logging.warning(" No FOG triplets found!")
    if not nonfog_triplets:
        logging.warning(" No non-FOG triplets found!")

    logging.info(f"Found {len(fog_triplets)} FOG and {len(nonfog_triplets)} non-FOG triplets.")

    # Step 2: Downsample non-FOG
    target_nonfog = min(len(fog_triplets), len(nonfog_triplets))
    nonfog_triplets_down = random.sample(nonfog_triplets, target_nonfog)
    logging.info(f"Using {len(fog_triplets)} FOG and {len(nonfog_triplets_down)} downsampled non-FOG.")

    # Step 3: Copy into train/valid folders
    split_and_copy_triplets(fog_triplets, args.output_dir, args.train_split, "1")
    split_and_copy_triplets(nonfog_triplets_down, args.output_dir, args.train_split, "0")

    logging.info(" Downsampling complete.")

if __name__ == "__main__":
    main()

