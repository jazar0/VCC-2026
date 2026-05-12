"""
EfficientNetV2-S Street Sign Inference - Camera Feed
Loads trained model and performs inference on webcam frames.
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
CLASS_NAMES_FILE = "class_names.json"
INPUT_SIZE = 384
CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# INITIALIZATION
# ============================================================================

def discover_models():
    """Scan the models directory and return a list of available .keras model files."""
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return []

    models = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".keras")
    ])

    return models


def choose_model():
    """List available models and prompt the user to pick one. Returns the full path."""
    models = discover_models()

    if not models:
        print("No .keras models found in:", MODELS_DIR)
        return None

    print(f"\nAvailable models in: {MODELS_DIR}")
    print("-" * 60)
    for i, name in enumerate(models, start=1):
        print(f"  {i}) {name}")
    print("-" * 60)

    while True:
        raw = input(f"\nEnter choice (1-{len(models)}) [default: 1]: ").strip()

        if raw == "":
            choice = 1
        else:
            try:
                choice = int(raw)
            except ValueError:
                print(f"  Please enter a number between 1 and {len(models)}.")
                continue

        if 1 <= choice <= len(models):
            selected = models[choice - 1]
            return os.path.join(MODELS_DIR, selected), selected
        else:
            print(f"  Please enter a number between 1 and {len(models)}.")


def load_model(model_path):
    """Load trained model from an explicit path."""
    print(f"\n✓ Loading model: {model_path}")
    return keras.models.load_model(model_path)


def load_class_names():
    """Load class names from JSON."""
    with open(CLASS_NAMES_FILE, "r") as f:
        return sorted(json.load(f))


# ============================================================================
# INFERENCE
# ============================================================================

def apply_zoom(frame, zoom_pct):
    """
    Crop the frame to a centered region based on zoom level, then resize
    back to the original dimensions so the output is always the same shape.

    zoom_pct: integer 0-100 from the trackbar.
      0   -> no zoom (full frame)
      100 -> maximum zoom (innermost 20% of each dimension)
    """
    if zoom_pct == 0:
        return frame

    h, w = frame.shape[:2]

    # Map 0-100 → crop factor 1.0 → 0.2  (never go below 20% of the frame)
    min_crop = 0.20
    crop_factor = 1.0 - (zoom_pct / 100.0) * (1.0 - min_crop)

    crop_h = int(h * crop_factor)
    crop_w = int(w * crop_factor)

    # Center the crop
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    y2 = y1 + crop_h
    x2 = x1 + crop_w

    cropped = frame[y1:y2, x1:x2]

    # Resize back to original dimensions, preserving the caller's expected shape
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(frame):
    """Preprocess frame for model input."""
    # Resize to model input size
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    batch = np.expand_dims(normalized, axis=0)

    return batch, resized


def predict_frame(model, frame, class_names):
    """Run inference on frame."""
    batch, resized = preprocess_frame(frame)

    # Get predictions
    predictions = model.predict(batch, verbose=0)

    # Get top prediction
    pred_class_idx = np.argmax(predictions[0])
    pred_confidence = predictions[0][pred_class_idx]
    pred_class_name = class_names[pred_class_idx]

    # Get top 3
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3 = [(class_names[i], predictions[0][i]) for i in top_3_indices]

    return pred_class_name, pred_confidence, top_3, resized


def draw_text_with_bg(frame, text, pos, font_scale, thickness, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Draw text with a semi-transparent background box for better contrast."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - text_height - 6), (x + text_width + 4, y + baseline + 2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    cv2.putText(frame, text, pos, font, font_scale, text_color, thickness)
    return frame


def draw_predictions(frame, class_name, confidence, top_3):
    """Draw predictions on frame."""
    # Main prediction
    text = f"{class_name}: {confidence:.2%}"
    draw_text_with_bg(frame, text, (10, 40), 1, 2, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Top 3
    draw_text_with_bg(frame, "Top 3:", (10, 100), 0.7, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))
    for i, (cls, conf) in enumerate(top_3):
        text = f"{i+1}. {cls}: {conf:.2%}"
        draw_text_with_bg(frame, text, (10, 130 + i*30), 0.6, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Instructions
    draw_text_with_bg(frame, "Press 'q' to quit, 's' to save frame", (10, frame.shape[0] - 20),
                     0.5, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    return frame


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EfficientNetV2-S Street Sign Inference - Camera Feed")
    print("="*70)

    # Load class names
    class_names = load_class_names()
    print(f"\n✓ Loaded {len(class_names)} classes")

    # Discover and choose model
    result = choose_model()
    if result is None:
        return
    model_path, model_name = result

    # Load model
    model = load_model(model_path)
    print(f"✓ Model loaded successfully")

    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("✓ Camera opened")
    print("\nStarting inference...")
    print("  Press 'q' to quit")
    print("  Press 's' to save frame")
    print("  Use the Zoom slider to zoom in/out")
    print()

    window_name = f"Street Sign Detection - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.createTrackbar("Zoom %", window_name, 0, 100, lambda _: None)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        # Flip horizontally (optional, for mirror effect)
        # frame = cv2.flip(frame, 1)

        # Apply zoom from trackbar
        zoom_pct = cv2.getTrackbarPos("Zoom %", window_name)
        frame = apply_zoom(frame, zoom_pct)

        # Get predictions
        class_name, confidence, top_3, resized = predict_frame(model, frame, class_names)

        # Draw on frame
        output_frame = frame.copy()
        output_frame = draw_predictions(output_frame, class_name, confidence, top_3)

        # Draw zoom level on frame
        cv2.putText(output_frame, f"Zoom: {zoom_pct}%", (10, output_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Display
        cv2.imshow(window_name, output_frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            frame_count += 1
            filename = f"street_sign_{class_name}_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved frame: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()