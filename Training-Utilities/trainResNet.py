import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "split_output"

IMG_SIZE = 224
BATCH_SIZE = 16

# Phase 1: frozen backbone
EPOCHS_FROZEN = 20

# Phase 2: fine-tuning
EPOCHS_FINETUNE = 15
FINETUNE_LR = 1e-5

# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    Path(DATASET_PATH) / "train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    Path(DATASET_PATH) / "val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    Path(DATASET_PATH) / "test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================
# IMPORTANT:
# Horizontal flipping is intentionally excluded because many
# classes are directional mirror pairs:
#   - Left Turn / Right Turn
#   - Keep Left / Keep Right
#   - One Way Left / One Way Right
#
# Flipping would generate visually valid images with incorrect labels.

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.03),      # ~ +/-11 degrees
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================================
# MODEL
# ============================================================

base_model = EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Phase 1: freeze entire backbone
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

# Correct preprocessing for EfficientNetV2
x = preprocess_input(x)

# training=False keeps BatchNorm layers in inference mode
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ============================================================
# CALLBACKS
# ============================================================

def make_callbacks(checkpoint_path, patience=5):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-7
    )

    return [early_stop, checkpoint, reduce_lr]

# ============================================================
# PHASE 1 — TRAIN FROZEN BACKBONE
# ============================================================

print("\n=== Phase 1: Training classifier head ===")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FROZEN,
    callbacks=make_callbacks(
        "best_phase1.keras",
        patience=5
    )
)

# ============================================================
# PHASE 2 — FINE-TUNING
# ============================================================
# Unfreeze only the final EfficientNet block.
#
# BatchNorm layers remain frozen because updating BN statistics
# on very small datasets often destabilizes transfer learning.

print("\n=== Phase 2: Fine-tuning final backbone layers ===")

base_model.trainable = True

# Freeze BatchNorm layers
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Unfreeze only deeper layers
fine_tune = False

for layer in base_model.layers:
    if "top_conv" in layer.name:
        fine_tune = True

    layer.trainable = fine_tune and not isinstance(
        layer,
        layers.BatchNormalization
    )

trainable_count = sum(
    1 for l in base_model.layers if l.trainable
)

print(
    f"Trainable backbone layers: "
    f"{trainable_count} / {len(base_model.layers)}"
)

model.compile(
    optimizer=keras.optimizers.Adam(FINETUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FINETUNE,
    callbacks=make_callbacks(
        "best_phase2.keras",
        patience=4
    )
)

# ============================================================
# EVALUATE
# ============================================================

print("\n=== Evaluating on test set ===")

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================

model.save("efficientnetv2b0_classifier.keras")

print(
    "\nModel saved as "
    "'efficientnetv2b0_classifier.keras'"
)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved class_names.json")