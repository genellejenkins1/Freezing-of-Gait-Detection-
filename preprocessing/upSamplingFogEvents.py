"""
upSamplingFogEvents.py
-----------------------
Performs data augmentation on GAF triplet images for FOG events to balance
the dataset against more abundant non-FOG events. Saves results in structured
train/valid directories for model input.
    
Usage:
    python upSamplingFogEvents.py --source_dir path/to/GAFImages --output_dir path/to/UpsampledSet --augmentations_per_sample 2

Author: Genelle Jenkins
Version: 1.0
"""

import os
import shutil
import random
import argparse
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Image augmentation settings
image_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def get_triplet_prefixes(base_path):
    """
    Scans directory to return lists of full triplet prefixes for FOG and non-FOG.
    """
    fog, nonfog = [], []
    for root, _, files in os.walk(base_path):
        x_files = [f for f in files if f.endswith("_x_1.png") or f.endswith("_x_0.png")]
        for f in x_files:
            label = "1" if f.endswith("_1.png") else "0"
            prefix = f.replace(f"_x_{label}.png", "")
            triplet_base = os.path.join(root, prefix)
            if all(os.path.exists(f"{triplet_base}_{axis}_{label}.png") for axis in ['x', 'y', 'z']):
                (fog if label == "1" else nonfog).append(triplet_base)
    return fog, nonfog

def augment_triplet(prefix_path, num_augments):
    """
    Applies augmentation `num_augments` times to a FOG triplet.
    """
    for i in range(num_augments):
        for axis in ['x', 'y', 'z']:
            src = f"{prefix_path}_{axis}_1.png"
            img = load_img(src)
            img_array = np.expand_dims(img_to_array(img), axis=0)
            aug_img = next(image_datagen.flow(img_array, batch_size=1))[0].astype('uint8')
            aug_path = f"{prefix_path}_aug_{i}_{axis}_1.png"
            array_to_img(aug_img).save(aug_path)

def copy_triplets(prefixes, split, base_output, is_augmented=False):
    """
    Copies original or augmented triplets into train/valid structure.
    """
    copied = 0
    for prefix in prefixes:
        label = "1" if "_1" in prefix else "0"
        valid = all(os.path.exists(f"{prefix}_{axis}_{label}.png") for axis in ['x', 'y', 'z'])
        if not valid:
            continue

        for axis in ['x', 'y', 'z']:
            src = f"{prefix}_{axis}_{label}.png"
            rel = os.path.relpath(os.path.dirname(prefix), start=raw_path)
            dest = os.path.join(base_output, split, rel, os.path.basename(src))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
        copied += 1
    logging.info(f"📂 Copied {copied} triplets to {split}/.")

def main():
    parser = argparse.ArgumentParser(description="Augment FOG GAF images to balance dataset.")
    parser.add_argument("--source_dir", required=True, help="Path to original GAF image directory")
    parser.add_argument("--output_dir", required=True, help="Where to save upsampled dataset")
    parser.add_argument("--augmentations_per_sample", type=int, default=2, help="How many augmentations per FOG sample")
    args = parser.parse_args()

    global raw_path
    raw_path = os.path.join(args.output_dir, "raw")
    train_path = os.path.join(args.output_dir, "train")
    valid_path = os.path.join(args.output_dir, "valid")

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)

    # Step 1: Copy original GAF data
    if not os.path.exists(os.path.join(raw_path, "S01R01")):
        logging.info(" Copying original GAF files to workspace...")
        shutil.copytree(args.source_dir, raw_path, dirs_exist_ok=True)
    else:
        logging.info(" Raw copy already exists. Skipping...")

    # Step 2: Identify complete triplets
    fog, nonfog = get_triplet_prefixes(raw_path)
    logging.info(f"💡 Found {len(fog)} FOG and {len(nonfog)} non-FOG triplets.")

    # Step 3: Validation split (30%)
    val_count = int(len(fog) * 0.3)
    valid_fog = random.sample(fog, val_count)
    valid_nonfog = random.sample(nonfog, val_count)

    fog_train = [p for p in fog if p not in valid_fog]
    nonfog_train = [p for p in nonfog if p not in valid_nonfog]

    # Step 4: Augment FOG training samples
    max_real_fog = len(nonfog_train) // (1 + args.augmentations_per_sample)
    if len(fog_train) > max_real_fog:
        fog_train = random.sample(fog_train, max_real_fog)

    augmented = []
    for prefix in fog_train:
        for i in range(args.augmentations_per_sample):
            aug_prefix = f"{prefix}_aug_{i}"
            augmented.append(aug_prefix)
        augment_triplet(prefix, args.augmentations_per_sample)

    # Step 5: Copy into train/valid folders
    copy_triplets(fog_train + augmented, "train", args.output_dir, is_augmented=True)
    copy_triplets(nonfog_train, "train", args.output_dir)
    copy_triplets(valid_fog, "valid", args.output_dir)
    copy_triplets(valid_nonfog, "valid", args.output_dir)

    logging.info("Augmentation and split complete.")

if __name__ == "__main__":
    main()
