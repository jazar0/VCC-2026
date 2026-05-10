import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B2, EfficientNetV2S, ResNet50V2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "split_output"

IMG_SIZE = 384
BATCH_SIZE = 16      # reduced from 32 — 384px images use ~4x more VRAM per batch
EPOCHS_FROZEN = 30       # Phase 1: early stopping will cut this short if needed
EPOCHS_FINETUNE = 20     # Phase 2: same
FINETUNE_LR = 1e-5       # Much lower LR for fine-tuning
UNFREEZE_LAYERS = 30     # How many top EfficientNet layers to unfreeze

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
val_dataset   = val_dataset.prefetch(AUTOTUNE)
test_dataset  = test_dataset.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================
# NOTE: RandomFlip("horizontal") is intentionally excluded.
# Many classes are mirror-image pairs (Left/Right Turn, One Way Left/Right,
# Left/Right Curve, etc). Flipping a "Left Turn Only" sign produces an image
# indistinguishable from "Right Turn Only" — but the label doesn't change.
# This was actively teaching the model to confuse left vs right variants.

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.2),             
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),      # sign not always centered in frame
    layers.RandomBrightness(0.3),            # simulate lighting changes
    layers.RandomContrast(0.3),              # simulate shadows and glare
], name="data_augmentation")

# ============================================================
# MODEL
# ============================================================

base_model = ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False   # frozen for Phase 1

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ============================================================
# PHASE 1 — TRAIN WITH FROZEN BACKBONE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Uniform class weights — forces equal penalty for mistakes on every class
# regardless of how many images each class has in each batch.
class_weight = {i: 1.0 for i in range(num_classes)}

# ============================================================
# CALLBACKS
# ============================================================
# EarlyStopping: stops training when val_loss hasn't improved for
# PATIENCE epochs, then restores the best weights seen so far.
# With only 25 images/class this is important — the model can
# memorise the training set quickly and val loss starts climbing.
#
# ModelCheckpoint: also saves the best weights to disk mid-training
# as a safety net in case of a crash.

def make_callbacks(checkpoint_path, patience=7):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,  # rolls back to best epoch automatically
        verbose=1
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    return [early_stop, checkpoint]

# ============================================================
# PHASE 1 — TRAIN WITH FROZEN BACKBONE
# ============================================================

print("\n=== Phase 1: Training with frozen backbone ===")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight,
    callbacks=make_callbacks("best_phase1.keras", patience=10)
)

# ============================================================
# PHASE 2 — FINE-TUNE TOP LAYERS OF BACKBONE
# ============================================================
# Unfreeze the top N layers of EfficientNet and train at a much
# lower learning rate so we adapt the backbone to our domain
# without destroying the ImageNet weights.

print(f"\n=== Phase 2: Fine-tuning top {UNFREEZE_LAYERS} backbone layers ===")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable backbone layers: {trainable_count} / {len(base_model.layers)}")

model.compile(
    optimizer=keras.optimizers.Adam(FINETUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Lower patience for fine-tuning — changes are smaller so overfitting
# tends to creep in more gradually; 5 epochs is enough to detect it.
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight,
    callbacks=make_callbacks("best_phase2.keras", patience=7)
)

# ============================================================
# EVALUATE
# ============================================================

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ============================================================
# SAVE
# ============================================================

model.save("resnet50v2_classifier.keras")
print("\nModel saved as resnet50v2_classifier.keras")

with open("class_names.json", "w") as f:
    json.dump(class_names, f)