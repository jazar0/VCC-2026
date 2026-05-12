"""
Simple Evaluation Script - Test trained models on test set
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_ROOT = "/home/dev/Desktop/VCC2026/Training-Utilities/Dataset_split"
MODELS_DIR = "/home/dev/Desktop/VCC2026/Training-Utilities/models"
CLASS_NAMES_FILE = "/home/dev/Desktop/VCC2026/Training-Utilities/class_names.json"

INPUT_SIZE = 384
BATCH_SIZE = 256


# ============================================================================
# DATA LOADING
# ============================================================================

def load_class_names():
    """Load class names from JSON."""
    with open(CLASS_NAMES_FILE, "r") as f:
        return sorted(json.load(f))


def load_and_preprocess_image(path, label):
    """Load, resize, normalize image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
    image = image / 255.0
    return image, label


def create_test_dataset(class_names):
    """Create test dataset."""
    image_paths = []
    labels = []
    
    test_dir = os.path.join(DATA_ROOT, "test")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for image_file in sorted(os.listdir(class_dir)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, image_file))
                labels.append(class_idx)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), len(image_paths)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_path, strategy, test_ds, class_names):
    """Evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {strategy}")
    print(f"{'='*70}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    
    print("Running inference on test set...")
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(pred_classes)
        all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n✓ Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n{'-'*70}")
    print("PER-CLASS METRICS")
    print(f"{'-'*70}")
    
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Print nicely formatted
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:35s} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    
    # Summary
    print(f"\n{'-'*70}")
    print("SUMMARY METRICS")
    print(f"{'-'*70}")
    print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall:    {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-Score:  {report['macro avg']['f1-score']:.4f}")
    
    # Weakest classes
    print(f"\n{'-'*70}")
    print("WEAKEST CLASSES (lowest recall)")
    print(f"{'-'*70}")
    recalls = [report[class_name]['recall'] for class_name in class_names]
    weakest_indices = np.argsort(recalls)[:5]
    for idx in weakest_indices:
        print(f"  {class_names[idx]:35s}: {recalls[idx]:.3f}")
    
    return accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MODEL EVALUATION - Test Set Assessment")
    print("="*70)
    
    # Load class names and test dataset
    class_names = load_class_names()
    print(f"\n✓ Loaded {len(class_names)} classes")
    
    print("\nLoading test dataset...")
    test_ds, num_test = create_test_dataset(class_names)
    print(f"✓ Loaded {num_test} test images")
    
    # Find available models
    strategies = {
        "freeze_early": "EfficientNetV2-S (Freeze Early Layers)",
        "progressive": "EfficientNetV2-S (Progressive Unfreezing)",
        "full": "EfficientNetV2-S (Full Fine-Tuning)",
    }
    
    results = {}
    
    for strategy, description in strategies.items():
        model_path = os.path.join(MODELS_DIR, f"efficientnetv2_s_{strategy}_final.keras")
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Model not found: {model_path}")
            continue
        
        accuracy = evaluate_model(model_path, description, test_ds, class_names)
        results[strategy] = accuracy
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for strategy, description in strategies.items():
        if strategy in results:
            accuracy = results[strategy]
            print(f"{description:50s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1])
        print(f"\n✓ Best model: {strategies[best_strategy[0]]} ({best_strategy[1]:.4f})")
    
    print()


if __name__ == "__main__":
    main()
