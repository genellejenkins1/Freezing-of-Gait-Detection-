"""
CNNModel5.py
------------
Trains a multi-branch CNN using streamed .npz chunks for memory-efficient training.
Intended for the upsampled dataset stored in preprocessed NumPy format.

Usage:
    python CNNModel5.py --data_dir path/to/preprocessed_numpy --save_dir path/to/save_dir --model_name fog_cnn_model_streamed.h5

Author: Genelle Jenkins
Version: 1.0
"""

import os
import glob
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Constants
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 192, 192, 3

# Logging setup
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# === Data Generator ===
def chunk_generator(chunk_files):
    for chunk_path in chunk_files:
        with np.load(chunk_path) as data:
            for x_img, y_img, z_img, label in zip(data['x'], data['y'], data['z'], data['labels']):
                yield (x_img, y_img, z_img), label

def create_dataset(pattern, batch_size):
    chunk_files = sorted(glob.glob(pattern))
    total_samples = sum([np.load(f)['labels'].shape[0] for f in chunk_files])

    output_signature = (
        (tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, CHANNELS), tf.float32),
         tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, CHANNELS), tf.float32),
         tf.TensorSpec((IMG_HEIGHT, IMG_WIDTH, CHANNELS), tf.float32)),
        tf.TensorSpec((), tf.int64)
    )

    ds = tf.data.Dataset.from_generator(
        lambda: chunk_generator(chunk_files),
        output_signature=output_signature
    )

    ds = ds.map(lambda inputs, y: (
        {'input_x': inputs[0], 'input_y': inputs[1], 'input_z': inputs[2]},
        tf.expand_dims(tf.cast(y, tf.float32), axis=-1)
    )).shuffle(1000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, total_samples


# === Training Routine ===
def train_model(data_dir, save_dir, model_name, batch_size=32, epochs=60):
    os.makedirs(save_dir, exist_ok=True)

    train_ds, train_samples = create_dataset(os.path.join(data_dir, "train_chunk_*.npz"), batch_size)
    val_ds, val_samples = create_dataset(os.path.join(data_dir, "valid_chunk_*.npz"), batch_size)
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    model = build_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Save model + history
    model.save(os.path.join(save_dir, model_name))
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f)

    # Evaluation
    preds, true_labels = [], []
    for x_batch, y_batch in val_ds.take(validation_steps):
        batch_preds = model.predict(x_batch).flatten()
        preds.extend(np.round(batch_preds))
        true_labels.extend(y_batch.numpy().flatten())

    report = classification_report(true_labels, preds, target_names=["Non-FOG", "FOG"], output_dict=True)
    cm = confusion_matrix(true_labels, preds)
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)

    # ROC Curve
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
    parser = argparse.ArgumentParser(description="Train streaming CNN using preprocessed .npz GAF triplets")
    parser.add_argument("--data_dir", required=True, help="Path to directory with .npz chunk files")
    parser.add_argument("--save_dir", required=True, help="Directory to save model and metrics")
    parser.add_argument("--model_name", default="fog_cnn_model_streamed.h5", help="Model output filename")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.model_name, args.batch_size, args.epochs)

if __name__ == "__main__":
    main()
