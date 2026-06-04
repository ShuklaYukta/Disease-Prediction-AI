"""
train_xray.py — CNN model for Chest X-Ray Pneumonia Detection
===============================================================

DATASET SETUP (do this first):
1. Download from Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. Extract so your folder looks like this:

   dataset/
   └── pneumonia/
       ├── train/
       │   ├── NORMAL/       ← normal X-ray images
       │   └── PNEUMONIA/    ← pneumonia X-ray images
       ├── val/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── test/
           ├── NORMAL/
           └── PNEUMONIA/

RUN:
   python train_xray.py

OUTPUT FILES:
   xray_model.h5          ← saved model (used by app.py)
   xray_model_info.pkl    ← class labels and image size info
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Check TensorFlow ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow not found. Run: pip install tensorflow")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (150, 150)   # resize all images to this
BATCH_SIZE  = 32
EPOCHS      = 20           # early stopping will cut this short if needed
TRAIN_DIR   = "dataset/pneumonia/train"
VAL_DIR     = "dataset/pneumonia/val"
TEST_DIR    = "dataset/pneumonia/test"

# ── Validate dataset paths ────────────────────────────────────────────────────
for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(path):
        print(f"❌ Folder not found: {path}")
        print("   Please download and extract the dataset as shown in the instructions above.")
        exit(1)

print("\n🚀 Starting X-Ray Model Training...")
print(f"   Image size  : {IMG_SIZE}")
print(f"   Batch size  : {BATCH_SIZE}")
print(f"   Max epochs  : {EPOCHS}")

# ── Data generators ───────────────────────────────────────────────────────────
# Training: augment images to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Validation & Test: only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",   # NORMAL=0, PNEUMONIA=1
    color_mode="rgb",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,
)

test_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,
)

class_labels = {v: k for k, v in train_gen.class_indices.items()}
print(f"\n📊 Class mapping: {class_labels}")
print(f"   Training samples   : {train_gen.samples}")
print(f"   Validation samples : {val_gen.samples}")
print(f"   Test samples       : {test_gen.samples}")

# ── Build CNN model ───────────────────────────────────────────────────────────
def build_model(input_shape=(150, 150, 3)):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Classifier head
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # binary output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        "xray_model_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🏋️ Training started...\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate on test set ──────────────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"   Test Accuracy : {test_acc:.2%}")
print(f"   Test Loss     : {test_loss:.4f}")

# ── Save model and metadata ───────────────────────────────────────────────────
model.save("xray_model.h5")
joblib.dump(
    {
        "class_labels": class_labels,   # {0: 'NORMAL', 1: 'PNEUMONIA'}
        "img_size": IMG_SIZE,
        "test_accuracy": test_acc,
    },
    "xray_model_info.pkl",
)

print("\n✅ Training complete!")
print(f"   Model saved    : xray_model.h5")
print(f"   Info saved     : xray_model_info.pkl")
print(f"   Test accuracy  : {test_acc:.2%}")
print("\n👉 Next: update app.py X-ray section to load xray_model.h5")