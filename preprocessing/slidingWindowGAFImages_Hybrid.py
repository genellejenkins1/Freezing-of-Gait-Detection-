"""
slidingWindowGAFImages_Hybrid.py
--------------------------------
Creates a hybrid dataset with a specified class ratio (e.g., 60% non-FOG, 40% FOG),
split into train/valid folders, preserving triplet integrity and folder structure.

Usage:
    python slidingWindowGAFImages_Hybrid.py --input_dir path/to/GAFImages --output_dir path/to/HybridSet --ratio_nonfog 0.6 --ratio_fog 0.4 --train_split 0.85

Author: Genelle Jenkins
Version: 1.0
"""

import os
import shutil
import random
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def collect_triplet_paths(base_dir):
    """
    Finds valid FOG and non-FOG triplet prefixes under base_dir.
    """
    fog, nonfog = [], []
    for root, _, files in os.walk(base_dir):
        x_files = [f for f in files if f.endswith("_x_1.png") or f.endswith("_x_0.png")]
        for f in x_files:
            label = "1" if f.endswith("_1.png") else "0"
            prefix = f.replace(f"_x_{label}.png", "")
            base_path = os.path.join(root, prefix)
            if all(os.path.exists(f"{base_path}_{axis}_{label}.png") for axis in ['x', 'y', 'z']):
                (fog if label == "1" else nonfog).append(base_path)
    return fog, nonfog

def split_data(items, ratio):
    """
    Splits list into train and validation sets.
    """
    random.shuffle(items)
    split = int(len(items) * ratio)
    return items[:split], items[split:]

def copy_triplets(prefixes, split, out_dir, base_dir):
    """
    Copies full triplet images to out_dir/train or out_dir/valid while keeping structure.
    """
    for prefix in prefixes:
        label = "1" if "_1" in prefix else "0"
        for axis in ['x', 'y', 'z']:
            src = f"{prefix}_{axis}_{label}.png"
            rel = os.path.relpath(os.path.dirname(prefix), base_dir)
            dest = os.path.join(out_dir, split, rel, os.path.basename(src))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)

def main():
    parser = argparse.ArgumentParser(description="Create a hybrid dataset with target FOG/non-FOG ratio.")
    parser.add_argument("--input_dir", required=True, help="Base GAF image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for hybrid dataset")
    parser.add_argument("--ratio_nonfog", type=float, default=0.6, help="Proportion of non-FOG samples")
    parser.add_argument("--ratio_fog", type=float, default=0.4, help="Proportion of FOG samples")
    parser.add_argument("--train_split", type=float, default=0.85, help="Train split ratio (default: 0.85)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_dir = args.input_dir

    # Step 1: Collect available triplets
    fog, nonfog = collect_triplet_paths(base_dir)
    logging.info(f"Found {len(nonfog)} non-FOG and {len(fog)} FOG triplets.")

    # Step 2: Determine sample counts
    max_size = min(len(fog) / args.ratio_fog, len(nonfog) / args.ratio_nonfog)
    target_fog = int(max_size * args.ratio_fog)
    target_nonfog = int(max_size * args.ratio_nonfog)

    sampled_fog = fog if len(fog) <= target_fog else random.sample(fog, target_fog)
    sampled_nonfog = random.sample(nonfog, target_nonfog)

    logging.info(f"Sampling {len(sampled_nonfog)} non-FOG and {len(sampled_fog)} FOG triplets.")

    # Step 3: Train/Valid split
    train_fog, valid_fog = split_data(sampled_fog, args.train_split)
    train_nonfog, valid_nonfog = split_data(sampled_nonfog, args.train_split)

    # Step 4: Copy all samples to final structure
    copy_triplets(train_fog, "train", args.output_dir, base_dir)
    copy_triplets(train_nonfog, "train", args.output_dir, base_dir)
    copy_triplets(valid_fog, "valid", args.output_dir, base_dir)
    copy_triplets(valid_nonfog, "valid", args.output_dir, base_dir)

    logging.info(f" Hybrid dataset created at: {args.output_dir}")
    logging.info(f"   ➤ Train: {len(train_fog) + len(train_nonfog)} samples")
    logging.info(f"   ➤ Valid: {len(valid_fog) + len(valid_nonfog)} samples")

if __name__ == "__main__":
    main()

