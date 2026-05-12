"""
EfficientNetV2-S Street Sign Classification - Single Training Script
Prompts for training strategy, trains model, saves it.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_ROOT = "Dataset_split"
MODELS_DIR = "models"
CLASS_NAMES_FILE = "class_names.json"

INPUT_SIZE = 384
BATCH_SIZE = 256
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 8

# Augmentation ranges (color only, no flips/rotations)
BRIGHTNESS_RANGE = 0.15
SATURATION_RANGE = 0.20
CONTRAST_RANGE = 0.15

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING & AUGMENTATION
# ============================================================================

def load_class_names():
    """Load class names from JSON."""
    with open(CLASS_NAMES_FILE, "r") as f:
        return sorted(json.load(f))


def augment_image(image):
    """Apply color augmentation (brightness, saturation, contrast)."""
    image = tf.image.adjust_brightness(image, tf.random.uniform([], -BRIGHTNESS_RANGE, BRIGHTNESS_RANGE))
    image = tf.image.adjust_saturation(image, tf.random.uniform([], 1 - SATURATION_RANGE, 1 + SATURATION_RANGE))
    image = tf.image.adjust_contrast(image, tf.random.uniform([], 1 - CONTRAST_RANGE, 1 + CONTRAST_RANGE))
    return tf.clip_by_value(image, 0.0, 1.0)


def load_and_preprocess_image(path, label, augment=False):
    """Load, resize, normalize, and optionally augment image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
    image = image / 255.0
    
    if augment:
        image = augment_image(image)
    
    return image, label


def create_dataset(directory, class_names, batch_size, augment=False):
    """Create tf.data.Dataset from directory."""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for image_file in sorted(os.listdir(class_dir)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, image_file))
                labels.append(class_idx)
    
    print(f"  Loaded {len(image_paths)} images from {directory}")
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if augment:
        dataset = dataset.shuffle(buffer_size=min(1000, len(image_paths)))
    
    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(class_names, strategy="freeze_early"):
    """Build EfficientNetV2-S model with specified strategy."""
    
    # Load pretrained base
    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
    )
    
    # Apply strategy
    if strategy == "full":
        print("  Strategy: FULL FINE-TUNING (all layers trainable)")
        base_model.trainable = True
        learning_rate = 1e-4
    elif strategy == "progressive":
        print("  Strategy: PROGRESSIVE UNFREEZING (start frozen, unfreeze gradually)")
        base_model.trainable = False
        learning_rate = 1e-3
    else:  # freeze_early
        print("  Strategy: FREEZE EARLY LAYERS (307 frozen, 206 trainable)")
        freeze_until = int(len(base_model.layers) * 0.6)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True
        learning_rate = 1e-3
    
    # Build custom head
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = base_model(inputs, training=(strategy == "full"))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    
    model = keras.Model(inputs, outputs, name=f"efficientnetv2_s_{strategy}")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model, learning_rate


# ============================================================================
# TRAINING
# ============================================================================

class ProgressiveUnfreezeCallback(keras.callbacks.Callback):
    """Progressively unfreeze layers during training."""
    
    def __init__(self, base_model, unfreeze_interval=10):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_interval = unfreeze_interval
        self.total_layers = len(base_model.layers)
        self.last_unfreeze_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch - self.last_unfreeze_epoch) >= self.unfreeze_interval and epoch > 0:
            blocks_to_unfreeze = (epoch // self.unfreeze_interval)
            layers_to_keep_frozen = max(20, self.total_layers - (blocks_to_unfreeze * 15))
            
            for layer in self.base_model.layers[:layers_to_keep_frozen]:
                layer.trainable = False
            for layer in self.base_model.layers[layers_to_keep_frozen:]:
                layer.trainable = True
            
            trainable_count = sum(1 for l in self.base_model.layers if l.trainable)
            frozen_count = sum(1 for l in self.base_model.layers if not l.trainable)
            
            print(f"\n  📍 Epoch {epoch}: Unfreezing layers | Frozen: {frozen_count}, Trainable: {trainable_count}")
            
            current_lr = float(self.model.optimizer.learning_rate)
            new_lr = current_lr * 0.9
            self.model.optimizer.learning_rate = new_lr
            print(f"    Reduced LR to: {new_lr:.2e}\n")
            
            self.last_unfreeze_epoch = epoch


def train_model(strategy="freeze_early"):
    """Train model with specified strategy."""
    
    print("\n" + "="*70)
    print(f"TRAINING: EfficientNetV2-S Street Sign Classification")
    print("="*70)
    
    # Load class names
    class_names = load_class_names()
    print(f"\nLoading {len(class_names)} classes...")
    
    # Determine batch size based on strategy
    # Full fine-tuning with all trainable layers requires less memory per sample
    if strategy == "full":
        batch_size = 64
        num_epochs = 100
    else:
        batch_size = BATCH_SIZE
        num_epochs = EPOCHS
    
    # Load datasets
    print("\nLoading datasets:")
    train_ds = create_dataset(os.path.join(DATA_ROOT, "train"), class_names, batch_size, augment=True)
    val_ds = create_dataset(os.path.join(DATA_ROOT, "val"), class_names, batch_size, augment=False)
    test_ds = create_dataset(os.path.join(DATA_ROOT, "test"), class_names, batch_size, augment=False)
    
    # Build model
    print("\nBuilding model:")
    model, learning_rate = build_model(class_names, strategy)
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Trainable params: {model.count_params():,}")
    
    # Create callbacks
    callbacks = [
        # Monitor val_loss: more sensitive than val_accuracy (catches overfitting
        # earlier since loss degrades before accuracy visibly plateaus).
        # min_delta=1e-4 prevents noise-level "improvements" from resetting the counter.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        # Checkpoint on val_loss for consistency with early stopping.
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f"efficientnetv2_s_{strategy}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        # Match monitor to val_loss; tighten patience so LR drops before
        # early stopping fires, giving the model one real chance to recover.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_delta=1e-4,
            mode="min",
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # Add progressive unfreezing callback if needed
    if strategy == "progressive":
        callbacks.append(ProgressiveUnfreezeCallback(model.layers[2].layers[0], unfreeze_interval=10))
    
    # Train
    print(f"\n{'='*70}")
    print(f"Starting training: {strategy.upper()}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*70}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*70}")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"efficientnetv2_s_{strategy}_final.keras")
    model.save(model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        "model_name": f"EfficientNetV2-S ({strategy})",
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "class_names": class_names,
    }
    metadata_path = os.path.join(MODELS_DIR, f"efficientnetv2_s_{strategy}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    
    return test_accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EfficientNetV2-S Street Sign Classification Training")
    print("="*70)
    print("\nChoose training strategy:\n")
    print("  1) FREEZE EARLY LAYERS (Fast, ~45 min - 1 hour)")
    print("     - Freeze ~60% of early layers, train later layers + head")
    print("     - Good accuracy, fastest training")
    print("     - Learning rate: 1e-3")
    print()
    print("  2) PROGRESSIVE UNFREEZING (Balanced, ~1.5-2 hours)")
    print("     - Start with frozen backbone, train head, unfreeze every 10 epochs")
    print("     - Very high accuracy, good balance")
    print("     - Learning rate: 1e-3 (reduces as layers unfreeze)")
    print()
    print("  3) FULL FINE-TUNING (Best accuracy, ~2-3 hours)")
    print("     - All layers trainable with conservative learning rate")
    print("     - Highest accuracy, slowest training")
    print("     - Learning rate: 1e-4")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    strategies = {
        "1": "freeze_early",
        "2": "progressive",
        "3": "full",
    }
    
    if choice not in strategies:
        print("Invalid choice. Using strategy 2 (progressive).")
        strategy = "progressive"
    else:
        strategy = strategies[choice]
    
    # Enable mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"\n✓ Mixed precision enabled: {policy.name}\n")
    
    # Train
    accuracy = train_model(strategy)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Model saved to: {MODELS_DIR}/efficientnetv2_s_{strategy}_final.keras")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()