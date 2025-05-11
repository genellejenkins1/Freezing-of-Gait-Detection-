"""
CNNModel4.py
------------
Trains a multi-branch CNN on triplet GAF images from ankle, hip, and thigh axes.
Intended for use with downsampled or hybrid datasets stored in train/valid folders.

Usage:
    python CNNModel4.py --data_dir path/to/slidingWindowGAFImages_Downsampled --save_dir path/to/save --epochs 60 --model_name fog_cnn_model.h5

Author: Genelle Jenkins
Version: 1.0
"""

import os
import json
import random
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Constants
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 192, 192, 3
FOG_LABEL_MAPPING = {'1': 1, '0': 0}

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# === Model Definition ===
def create_branch(input_shape, name):
    input_layer = Input(shape=input_shape, name=name)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    return input_layer, x

def build_model():
    input_x, branch_x = create_branch((IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='input_x')
    input_y, branch_y = create_branch((IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='input_y')
    input_z, branch_z = create_branch((IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='input_z')

    merged = concatenate([branch_x, branch_y, branch_z])
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(merged)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_x, input_y, input_z], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# === Data Generator ===
class MultiBranchDataGenerator(Sequence):
    def __init__(self, image_dir, batch_size, augment=False):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.augment = augment
        self.image_paths, self.labels = self._load_image_paths_and_labels()
        self.datagen = ImageDataGenerator(rescale=1 / 255.0)
        self.on_epoch_end()

    def _load_image_paths_and_labels(self):
        paths, labels = [], []
        for session in os.listdir(self.image_dir):
            session_path = os.path.join(self.image_dir, session)
            if not os.path.isdir(session_path):
                continue
            for accel in ["ankle", "hip", "thigh"]:
                accel_path = os.path.join(session_path, accel)
                if not os.path.isdir(accel_path):
                    continue
                for file in sorted(os.listdir(accel_path)):
                    if file.endswith(".png"):
                        label_str = file.split("_")[-1].split(".")[0]
                        label = FOG_LABEL_MAPPING.get(label_str)
                        if label is not None:
                            paths.append(os.path.join(accel_path, file))
                            labels.append(label)
        return paths, labels

    def __len__(self):
        return max(1, len(self.image_paths) // (self.batch_size * 3))

    def __getitem__(self, index):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            idx = index * self.batch_size * 3 + i * 3
            if idx + 2 >= len(self.image_paths):
                continue
            x = self._load_image(self.image_paths[idx])
            y = self._load_image(self.image_paths[idx + 1])
            z = self._load_image(self.image_paths[idx + 2])
            batch_x.append([x, y, z])
            batch_y.append(self.labels[idx])

        return {
            'input_x': tf.convert_to_tensor([x[0] for x in batch_x]),
            'input_y': tf.convert_to_tensor([x[1] for x in batch_x]),
            'input_z': tf.convert_to_tensor([x[2] for x in batch_x])
        }, tf.convert_to_tensor(batch_y, dtype=tf.float32)

    def _load_image(self, path):
        img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        return self.datagen.standardize(img_to_array(img))

    def on_epoch_end(self):
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)


# === Training Routine ===
def train_model(data_dir, save_dir, epochs, batch_size, model_name):
    os.makedirs(save_dir, exist_ok=True)
    model = build_model()
    model.summary()

    train_generator = MultiBranchDataGenerator(os.path.join(data_dir, 'train'), batch_size=batch_size)
    val_generator = MultiBranchDataGenerator(os.path.join(data_dir, 'valid'), batch_size=batch_size)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save model + history
    model.save(os.path.join(save_dir, model_name))
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f)

    # Evaluation
    preds, true_labels = [], []
    for i in range(len(val_generator)):
        x, y = val_generator[i]
        p = model.predict(x).flatten()
        preds.extend(np.round(p))
        true_labels.extend(y.numpy())

    report = classification_report(true_labels, preds, target_names=["Non-FOG", "FOG"], output_dict=True)
    cm = confusion_matrix(true_labels, preds)
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)

    # ROC
    fpr, tpr, _ = roc_curve(true_labels, preds)
    auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()


# === Entry Point ===
def main():
    parser = argparse.ArgumentParser(description="Train CNN on GAF triplet images.")
    parser.add_argument("--data_dir", required=True, help="Path to train/valid GAF images")
    parser.add_argument("--save_dir", required=True, help="Directory to save model + outputs")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_name", default="fog_cnn_model.h5", help="Filename for saved model")
    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.epochs, args.batch_size, args.model_name)


if __name__ == "__main__":
    main()

