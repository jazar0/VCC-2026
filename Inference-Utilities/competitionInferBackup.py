"""
EfficientNetV2-S Street Sign Inference - Camera Feed
Loads trained model and performs inference on webcam frames.

Transition robustness:
  - Motion gate:      frames where the mean absolute pixel difference vs. the
                      previous frame exceeds MOTION_THRESHOLD are skipped and
                      not fed into the prediction buffer. This catches slide
                      transitions regardless of whether the incoming slide is
                      sharp or augmented (e.g. intentional Gaussian blur).
                      Unlike a sharpness/Laplacian gate, it does NOT penalise
                      valid blurry competition images, which are static and
                      therefore produce a near-zero diff score.
  - Temporal buffer:  raw softmax probability vectors are accumulated in a
                      fixed-length deque. The committed prediction is the
                      argmax of the *summed* probability vectors, so a single
                      high-confidence frame contributes more than several
                      low-confidence ones (better than majority voting).
  - Stability gate:   the committed class only changes after it has been the
                      winner for MIN_STABLE_FRAMES consecutive frames, preventing
                      a brief clean frame mid-transition from triggering TTS.
"""

import json
import os
import queue
import threading
from collections import deque

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

# --- Temporal smoothing ---
# Number of recent frames whose softmax vectors are summed to pick a class.
# Larger = more stable but slower to react to a genuine sign change.
# At ~10 FPS inference, 8 frames ~ 0.8 s of history.
BUFFER_SIZE = 8

# The aggregated (summed) confidence of the winning class must exceed this
# fraction of the total probability mass in the buffer before we commit.
# Keeps the system silent/unchanged during ambiguous transitions.
# Range 0-1; 0.4 means the winner holds >=40% of the pooled probability.
COMMIT_CONFIDENCE = 0.40

# How many consecutive frames the aggregated winner must stay the same
# before we accept it as the new committed class and fire TTS.
MIN_STABLE_FRAMES = 4

# --- Motion / transition rejection ---
# Mean absolute pixel difference (0-255 scale) between the current frame and
# the previous one. Slide transitions cause a large, sudden diff spike.
# Intentionally blurred/augmented competition images are STATIC, so their
# diff stays near zero and they pass through correctly.
# Raise if genuine transitions still sneak into the buffer;
# lower if the first frames of a new slide are being skipped too long.
MOTION_THRESHOLD = 18.0


# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================

class TTSWorker:
    """
    Single background thread that owns one pyttsx3 engine for its lifetime.
    Calling speak() drops any queued/in-progress speech and starts the new one
    immediately, preventing crashes from spawning multiple engines.
    """

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self._engine = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Request new speech. Discards any pending item not yet started."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
        except Exception:
            self._engine = None

        while True:
            text = self._queue.get()
            if text is None:
                break
            self._say(text)

    def _say(self, text):
        if self._engine is not None:
            try:
                self._engine.stop()
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                try:
                    import pyttsx3
                    self._engine = pyttsx3.init()
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass
        else:
            import platform
            system = platform.system()
            if system == "Darwin":
                os.system(f'say "{text}"')
            elif system == "Linux":
                os.system(f'espeak "{text}" 2>/dev/null || spd-say "{text}" 2>/dev/null')
            elif system == "Windows":
                os.system(
                    f'powershell -Command "Add-Type -AssemblyName System.Speech; '
                    f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
                )

    def shutdown(self):
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


_tts = TTSWorker()


def speak(text):
    _tts.speak(text)


# ============================================================================
# TEMPORAL PREDICTION SMOOTHER
# ============================================================================

class PredictionSmoother:
    """
    Aggregates raw softmax probability vectors over a sliding window of frames
    and emits a stable committed class only when confidence and consistency
    criteria are both met.

    Why sum probability vectors instead of majority-voting on class indices?
      A frame with 0.95 confidence votes much more strongly than one with 0.34
      confidence -- both would count as one vote under majority voting, but the
      probability sum weights them correctly.
    """

    def __init__(self, num_classes, buffer_size=BUFFER_SIZE,
                 commit_confidence=COMMIT_CONFIDENCE,
                 min_stable_frames=MIN_STABLE_FRAMES):
        self.num_classes = num_classes
        self.buffer = deque(maxlen=buffer_size)
        self.commit_confidence = commit_confidence
        self.min_stable_frames = min_stable_frames

        self._candidate = None
        self._candidate_streak = 0
        self.committed_class = None
        self.committed_conf = 0.0

    def update(self, probs: np.ndarray):
        """
        Feed one softmax probability vector.
        Returns (committed_class_idx, normalised_confidence, candidate_class_idx).
        committed_class_idx is None until the buffer fills and stability fires.
        """
        self.buffer.append(probs)

        summed = np.sum(self.buffer, axis=0)
        normed = summed / summed.sum()

        candidate_idx  = int(np.argmax(normed))
        candidate_conf = float(normed[candidate_idx])

        if candidate_idx == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = candidate_idx
            self._candidate_streak = 1

        if (candidate_conf >= self.commit_confidence
                and self._candidate_streak >= self.min_stable_frames):
            self.committed_class = candidate_idx
            self.committed_conf  = candidate_conf

        return self.committed_class, self.committed_conf, candidate_idx

    def reset(self):
        self.buffer.clear()
        self._candidate = None
        self._candidate_streak = 0
        self.committed_class = None
        self.committed_conf = 0.0

    @property
    def fill(self):
        return len(self.buffer) / self.buffer.maxlen


# ============================================================================
# INITIALIZATION
# ============================================================================

def discover_models():
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return []
    return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")])


def choose_model():
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
        print(f"  Please enter a number between 1 and {len(models)}.")


def load_model(model_path):
    print(f"\n✓ Loading model: {model_path}")
    return keras.models.load_model(model_path)


def load_class_names():
    with open(CLASS_NAMES_FILE, "r") as f:
        return sorted(json.load(f))


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

def is_transitioning(frame, prev_gray, threshold=MOTION_THRESHOLD):
    """
    Returns (transitioning: bool, motion_score: float).

    Computes the mean absolute pixel difference between the current frame and
    the previous one (both grayscale). A slide change causes a large, abrupt
    spike in this score. A static sign -- even one with intentional Gaussian
    blur, noise, or other augmentation applied by the competition -- stays near
    zero because the content is not moving between frames.

    This is the correct gate for competition use. A sharpness-based (Laplacian)
    gate would incorrectly reject valid blurry input images; this one does not.

    prev_gray may be None on the very first frame: returns (False, 0.0) then.
    """
    if prev_gray is None:
        return False, 0.0
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = float(cv2.absdiff(gray, prev_gray).mean())
    return score > threshold, score


def apply_zoom(frame, zoom_pct):
    if zoom_pct == 0:
        return frame

    h, w = frame.shape[:2]
    min_crop    = 0.20
    crop_factor = 1.0 - (zoom_pct / 100.0) * (1.0 - min_crop)
    crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(frame):
    resized    = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    batch      = np.expand_dims(normalized, axis=0)
    return batch, resized


def predict_frame(model, frame):
    """Returns the full softmax probability vector (shape: num_classes,)."""
    batch, _ = preprocess_frame(frame)
    predictions = model.predict(batch, verbose=0)
    return predictions[0]


# ============================================================================
# HUD DRAWING
# ============================================================================

def draw_text_with_bg(frame, text, pos, font_scale, thickness,
                      bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, pos, font, font_scale, text_color, thickness)
    return frame


def draw_hud(frame, committed_name, committed_conf, candidate_name,
             motion_score, in_transition, smoother, zoom_pct):
    """Draw all overlays onto the frame in-place."""
    h, w = frame.shape[:2]

    # Committed class -- top left
    label_color  = (255, 255, 255) if committed_name else (100, 100, 100)
    display_name = committed_name if committed_name else "Waiting..."
    draw_text_with_bg(frame, display_name, (10, 40), 1.5, 2,
                      bg_color=(0, 0, 0), text_color=label_color)

    if committed_name:
        draw_text_with_bg(frame, f"Committed: {committed_conf:.1%}", (10, 75), 0.65, 1,
                          bg_color=(0, 0, 0), text_color=(0, 220, 100))

    # Live candidate (pre-smoothing)
    draw_text_with_bg(frame, f"Live: {candidate_name}", (10, 105), 0.6, 1,
                      bg_color=(0, 0, 0), text_color=(180, 180, 255))

    # Buffer fill bar -- top right
    bar_w, bar_h = 120, 12
    bar_x = w - bar_w - 10
    bar_y = 10
    fill_px = int(smoother.fill * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_px, bar_y + bar_h), (0, 200, 255), -1)
    draw_text_with_bg(frame, f"Buffer {len(smoother.buffer)}/{smoother.buffer.maxlen}",
                      (bar_x, bar_y + bar_h + 18), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=(0, 200, 255))

    # Motion indicator
    motion_color = (0, 60, 255) if in_transition else (0, 200, 80)
    motion_label = f"MOTION {motion_score:.1f}" if in_transition else f"Still  {motion_score:.1f}"
    draw_text_with_bg(frame, motion_label, (bar_x, bar_y + bar_h + 40), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=motion_color)

    # Stability streak
    draw_text_with_bg(frame,
                      f"Stable {smoother._candidate_streak}/{smoother.min_stable_frames}",
                      (bar_x, bar_y + bar_h + 60), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=(200, 200, 100))

    # Zoom + instructions -- bottom
    cv2.putText(frame, f"Zoom: {zoom_pct}%", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    draw_text_with_bg(frame, "q quit  s save  r reset buffer",
                      (10, h - 20), 0.5, 1,
                      bg_color=(0, 0, 0), text_color=(255, 255, 255))

    return frame


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EfficientNetV2-S Street Sign Inference - Camera Feed")
    print("="*70)
    print(f"\nSmoothing config:")
    print(f"  Buffer size      : {BUFFER_SIZE} frames")
    print(f"  Commit confidence: {COMMIT_CONFIDENCE:.0%}")
    print(f"  Stability frames : {MIN_STABLE_FRAMES}")
    print(f"  Motion threshold : {MOTION_THRESHOLD}")

    class_names = load_class_names()
    print(f"\n✓ Loaded {len(class_names)} classes")

    result = choose_model()
    if result is None:
        return
    model_path, model_name = result

    model = load_model(model_path)
    print("✓ Model loaded successfully")

    print("\nOpening camera...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    print("✓ Camera opened")
    print("\nStarting inference...")
    print("  q -> quit   s -> save frame   r -> reset prediction buffer\n")

    window_name = f"Street Sign Detection - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.createTrackbar("Zoom %", window_name, 0, 100, lambda _: None)

    smoother    = PredictionSmoother(len(class_names))
    last_spoken = None
    frame_count = 0
    prev_gray   = None      # grayscale previous frame for motion detection

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        zoom_pct = cv2.getTrackbarPos("Zoom %", window_name)
        frame    = apply_zoom(frame, zoom_pct)

        # ── Motion gate ────────────────────────────────────────────────────
        in_transition, motion_score = is_transitioning(frame, prev_gray)

        # Always update prev_gray -- even skipped frames become the reference
        # so the next frame's diff correctly reflects the actual change.
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if in_transition:
            # Show HUD with current committed state but skip buffer update.
            candidate_name = class_names[smoother._candidate] if smoother._candidate is not None else "--"
            committed_name = class_names[smoother.committed_class] if smoother.committed_class is not None else None
            output_frame   = frame.copy()
            draw_hud(output_frame, committed_name, smoother.committed_conf,
                     candidate_name, motion_score, True, smoother, zoom_pct)
            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                smoother.reset()
                last_spoken = None
                print("  Buffer reset.")
            continue

        # ── Inference ──────────────────────────────────────────────────────
        probs = predict_frame(model, frame)

        # ── Temporal smoothing ─────────────────────────────────────────────
        committed_idx, committed_conf, candidate_idx = smoother.update(probs)

        committed_name = class_names[committed_idx] if committed_idx is not None else None
        candidate_name = class_names[candidate_idx]

        # ── TTS: fire only when committed class changes ────────────────────
        if committed_name is not None and committed_name != last_spoken:
            speak(committed_name)
            last_spoken = committed_name

        # ── Draw HUD ───────────────────────────────────────────────────────
        output_frame = frame.copy()
        draw_hud(output_frame, committed_name, committed_conf,
                 candidate_name, motion_score, False, smoother, zoom_pct)

        cv2.imshow(window_name, output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            frame_count += 1
            label    = committed_name or candidate_name
            filename = f"street_sign_{label}_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('r'):
            smoother.reset()
            last_spoken = None
            print("  Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    _tts.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()