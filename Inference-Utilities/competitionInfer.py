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

ROI selection:
  - Press 'd' to enter draw mode. The feed freezes on the current frame and
    you drag a rectangle with the mouse. The rectangle is locked to the camera's
    aspect ratio as you drag, expanding from the anchor corner.
  - Release the mouse (or press Escape / 'd' again) to confirm or cancel.
  - Press 'c' to clear the ROI back to the full frame.
  - The active ROI is always shown as a rectangle overlay on the live feed.
  - Inference and motion detection both operate on the cropped ROI region.
"""

import json
import os
import queue
import sys
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
BUFFER_SIZE = 8
COMMIT_CONFIDENCE = 0.40
MIN_STABLE_FRAMES = 4

# --- Motion / transition rejection ---
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
# ROI SELECTOR
# ============================================================================

class ROISelector:
    """
    Interactive, aspect-ratio-locked ROI drawn directly on the live feed.

    Usage
    -----
    Press 'd' to toggle draw mode. While in draw mode the feed is frozen on
    the frame captured at entry. Click and drag to define the ROI; the
    rectangle is constrained to the camera's aspect ratio (w:h) as you drag,
    always growing outward from the anchor corner. Releasing the mouse commits
    the selection. Pressing 'd' or Escape while in draw mode cancels and
    restores the previous ROI. Press 'c' at any time to reset to full frame.

    The active ROI is (x1, y1, x2, y2) in raw camera pixel coordinates.
    apply(frame) crops that region and resizes it to the full frame shape so
    downstream code sees a normal-sized image.
    """

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._aspect = frame_w / frame_h           # locked aspect ratio

        # Active ROI -- None means full frame
        self._roi = None                            # (x1, y1, x2, y2)

        # Draw-mode state
        self.drawing = False
        self._frozen_frame = None                   # frame captured on 'd' press
        self._anchor = None                         # (ax, ay) mouse-down point
        self._cursor = None                         # current mouse position
        self._prev_roi = None                       # saved ROI to restore on cancel

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enter_draw_mode(self, current_frame):
        """Call when the user presses 'd' to start drawing."""
        if self.drawing:
            self._cancel()
            return
        self.drawing = True
        self._frozen_frame = current_frame.copy()
        self._prev_roi = self._roi
        self._anchor = None
        self._cursor = None

    def cancel_draw_mode(self):
        """Call on Escape or second 'd' press to abort without committing."""
        self._cancel()

    def clear(self):
        """Reset ROI to full frame."""
        self._roi = None
        self.drawing = False

    def handle_mouse(self, event, x, y, flags, param):
        """Pass directly to cv2.setMouseCallback."""
        if not self.drawing:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._anchor = (x, y)
            self._cursor = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._anchor is not None:
            self._cursor = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._anchor is not None:
            self._cursor = (x, y)
            roi = self._compute_roi(self._anchor, self._cursor)
            if roi is not None:
                self._roi = roi
            self.drawing = False
            self._anchor = None

    def apply(self, frame):
        """
        Return the ROI region of frame, resized back to the full frame
        dimensions. If no ROI is set, returns frame unchanged.
        """
        if self._roi is None:
            return frame
        x1, y1, x2, y2 = self._roi
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (self.frame_w, self.frame_h), interpolation=cv2.INTER_LINEAR)

    def draw_overlay(self, frame):
        """
        Draw the ROI rectangle (and the in-progress drag rectangle if
        currently drawing) onto frame in-place. Also dims the area outside
        the ROI so the operator can clearly see what will be fed to the model.
        """
        h, w = frame.shape[:2]

        if self.drawing and self._anchor is not None and self._cursor is not None:
            roi = self._compute_roi(self._anchor, self._cursor)
            if roi is not None:
                self._draw_dim_and_rect(frame, roi, color=(0, 220, 255), label="Drawing ROI")
            # Show instruction while drawing
            cv2.putText(frame, "Release to confirm  |  Esc to cancel",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        elif self.drawing:
            # Entered draw mode but mouse not yet pressed
            cv2.putText(frame, "Click and drag to draw ROI",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        elif self._roi is not None:
            self._draw_dim_and_rect(frame, self._roi, color=(0, 255, 100), label="ROI")

    @property
    def roi(self):
        """Active ROI as (x1, y1, x2, y2), or None for full frame."""
        return self._roi

    @property
    def frozen_frame(self):
        return self._frozen_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel(self):
        self.drawing = False
        self._roi = self._prev_roi
        self._anchor = None
        self._cursor = None
        self._frozen_frame = None

    def _compute_roi(self, anchor, cursor):
        """
        Compute an aspect-ratio-locked rectangle from anchor to cursor.
        The rectangle always maintains frame_w:frame_h proportions and is
        clamped to the frame boundary. Returns (x1, y1, x2, y2) or None if
        the resulting rectangle is too small to be useful.
        """
        ax, ay = anchor
        cx, cy = cursor

        dx = cx - ax
        dy = cy - ay

        # Choose the dominant axis and derive the other from the aspect ratio.
        # We pick whichever raw delta is larger (in aspect-corrected units).
        if abs(dx) == 0 and abs(dy) == 0:
            return None

        # Candidate sizes from each axis
        w_from_dx = abs(dx)
        h_from_dx = w_from_dx / self._aspect

        h_from_dy = abs(dy)
        w_from_dy = h_from_dy * self._aspect

        # Use the larger area
        if w_from_dx * h_from_dx >= w_from_dy * h_from_dy:
            rw, rh = w_from_dx, h_from_dx
        else:
            rw, rh = w_from_dy, h_from_dy

        rw, rh = int(round(rw)), int(round(rh))

        # Expand from anchor toward cursor (preserve drag direction)
        x1 = ax if dx >= 0 else ax - rw
        y1 = ay if dy >= 0 else ay - rh
        x2 = x1 + rw
        y2 = y1 + rh

        # Clamp to frame
        x1 = max(0, min(x1, self.frame_w - 1))
        y1 = max(0, min(y1, self.frame_h - 1))
        x2 = max(0, min(x2, self.frame_w))
        y2 = max(0, min(y2, self.frame_h))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        return (x1, y1, x2, y2)

    def _draw_dim_and_rect(self, frame, roi, color, label):
        """Dim everything outside roi, then draw a labelled rectangle."""
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]

        # Semi-transparent dark mask outside ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        dim = frame.copy()
        dim[mask == 0] = (dim[mask == 0] * 0.35).astype(np.uint8)
        cv2.addWeighted(dim, 1.0, frame, 0.0, 0, frame)

        # Rectangle border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner markers for clarity
        corner_len = 12
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 3)

        # Label above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        roi_w = x2 - x1
        roi_h = y2 - y1
        text = f"{label}  {roi_w}x{roi_h}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        tx = x1
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), font, 0.5, color, 1)


# ============================================================================
# TEMPORAL PREDICTION SMOOTHER
# ============================================================================

class PredictionSmoother:
    """
    Aggregates raw softmax probability vectors over a sliding window and emits
    a stable committed class only when confidence and consistency criteria are
    both met. Summing probability vectors weights high-confidence frames more
    heavily than low-confidence ones (superior to majority voting).
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
    Computes mean absolute pixel difference vs. previous grayscale frame.
    Slide transitions spike; static augmented images (blur etc.) stay near 0.
    prev_gray=None on first frame returns (False, 0.0).
    """
    if prev_gray is None:
        return False, 0.0
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = float(cv2.absdiff(gray, prev_gray).mean())
    return score > threshold, score


def preprocess_frame(frame):
    resized    = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_frame(model, frame):
    """Returns the full softmax probability vector (shape: num_classes,)."""
    return model.predict(preprocess_frame(frame), verbose=0)[0]


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


def draw_big_class_label(frame, text, text_color=(255, 255, 255)):
    """
    Draw the class label as large as possible, centred horizontally,
    positioned in the upper 60 % of the frame.  The font scale is computed
    so the text spans ~90 % of the frame width (capped so it never exceeds
    60 % of the frame height).
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(frame.shape[1] / 120))   # scale stroke with width

    # Binary-search for the largest scale where the text fits within 90 % of w
    # and 60 % of h.
    target_w  = frame.shape[1] * 0.90
    max_h     = frame.shape[0] * 0.60
    lo, hi    = 0.5, 50.0
    for _ in range(20):
        mid = (lo + hi) / 2
        (tw, th), bl = cv2.getTextSize(text, font, mid, thickness)
        if tw <= target_w and th <= max_h:
            lo = mid
        else:
            hi = mid
    font_scale = lo

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]

    # Centre horizontally; place baseline at ~50 % of frame height
    x = (w - tw) // 2
    y = int(h * 0.50)

    # Semi-transparent background pill
    pad_x, pad_y = int(tw * 0.03) + 8, int(th * 0.10) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad_x, y - th - pad_y),
                  (x + tw + pad_x, y + baseline + pad_y),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness,
                cv2.LINE_AA)


def draw_hud(frame, committed_name, committed_conf, candidate_name,
             motion_score, in_transition, smoother, roi_selector):
    h, w = frame.shape[:2]

    # Committed class -- giant centred label
    label_color  = (255, 255, 255) if committed_name else (100, 100, 100)
    display_name = committed_name if committed_name else "Waiting..."
    draw_big_class_label(frame, display_name, text_color=label_color)

    # Small secondary info -- top left, below the big label
    info_y_start = int(h * 0.50) + 24   # just below the big text baseline

    if committed_name:
        draw_text_with_bg(frame, f"Committed: {committed_conf:.1%}",
                          (10, info_y_start), 0.65, 1,
                          bg_color=(0, 0, 0), text_color=(0, 220, 100))

    draw_text_with_bg(frame, f"Live: {candidate_name}",
                      (10, info_y_start + 28), 0.6, 1,
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

    motion_color = (0, 60, 255) if in_transition else (0, 200, 80)
    motion_label = f"MOTION {motion_score:.1f}" if in_transition else f"Still  {motion_score:.1f}"
    draw_text_with_bg(frame, motion_label, (bar_x, bar_y + bar_h + 40), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=motion_color)

    draw_text_with_bg(frame,
                      f"Stable {smoother._candidate_streak}/{smoother.min_stable_frames}",
                      (bar_x, bar_y + bar_h + 60), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=(200, 200, 100))

    # ROI indicator -- below stability
    roi_label = "ROI: full frame" if roi_selector.roi is None else "ROI: custom"
    roi_color = (160, 160, 160) if roi_selector.roi is None else (0, 255, 100)
    draw_text_with_bg(frame, roi_label, (bar_x, bar_y + bar_h + 80), 0.45, 1,
                      bg_color=(0, 0, 0), text_color=roi_color)

    # Instructions -- bottom left (don't collide with ROI draw prompt)
    if not roi_selector.drawing:
        draw_text_with_bg(frame, "q quit  s save  r reset buf  d draw ROI  c clear ROI",
                          (10, h - 20), 0.5, 1,
                          bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # ROI rectangle / dim overlay (drawn last so it's on top)
    roi_selector.draw_overlay(frame)

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
    print(f"\nROI controls:")
    print(f"  d  -- enter draw mode (feed freezes; click-drag to set ROI)")
    print(f"  c  -- clear ROI back to full frame")
    print(f"  Esc -- cancel draw mode without committing")

    class_names = load_class_names()
    print(f"\n✓ Loaded {len(class_names)} classes")

    result = choose_model()
    if result is None:
        return
    model_path, model_name = result

    model = load_model(model_path)
    print("✓ Model loaded successfully")

    print("\nOpening camera...")


    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Open with V4L2 backend explicitly — required for FOURCC to be respected
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    # Set FOURCC *before* resolution/FPS — order matters on V4L2
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set last so it doesn't get reset

    # Verify what actually got negotiated
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fmt = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Format : {actual_fmt}  ({actual_w}x{actual_h} @ {actual_fps:.0f}fps)")
    if actual_fmt != "MJPG":
        print("  WARNING: Camera did not accept MJPG — still on YUYV, expect low FPS")


    #print(sys.argv[1])
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # Read a frame to get actual camera dimensions (may differ from defaults)
    ret, probe = cap.read()
    if not ret:
        print("Could not read from camera")
        return
    cam_h, cam_w = probe.shape[:2]
    print(f"✓ Camera opened  ({cam_w}x{cam_h})")
    print("\nStarting inference...")
    print("  q -> quit   s -> save   r -> reset buffer   d -> draw ROI   c -> clear ROI\n")

    window_name = f"Street Sign Detection - {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, int(960 * cam_h / cam_w))

    roi_selector = ROISelector(cam_w, cam_h)
    cv2.setMouseCallback(window_name, roi_selector.handle_mouse)

    smoother    = PredictionSmoother(len(class_names))
    last_spoken = None
    frame_count = 0
    prev_gray   = None      # previous ROI-cropped grayscale frame

    while True:
        # While in draw mode, freeze the feed and show the frozen frame with
        # the in-progress drag rectangle; don't run inference.
        if roi_selector.drawing:
            display = roi_selector.frozen_frame.copy()
            roi_selector.draw_overlay(display)
            # Draw minimal HUD so the operator still sees committed class
            committed_name = class_names[smoother.committed_class] if smoother.committed_class is not None else None
            if committed_name:
                draw_big_class_label(display, committed_name)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d') or key == 27:   # 27 = Escape
                roi_selector.cancel_draw_mode()
                print("  ROI draw cancelled.")
            continue

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Apply ROI: crop-and-resize to full frame dimensions.
        # Both inference and motion detection run on this region.
        roi_frame = roi_selector.apply(frame)

        # ── Motion gate (operates on ROI region) ──────────────────────────
        in_transition, motion_score = is_transitioning(roi_frame, prev_gray)
        prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        if in_transition:
            candidate_name = class_names[smoother._candidate] if smoother._candidate is not None else "--"
            committed_name = class_names[smoother.committed_class] if smoother.committed_class is not None else None
            output_frame   = frame.copy()
            draw_hud(output_frame, committed_name, smoother.committed_conf,
                     candidate_name, motion_score, True, smoother, roi_selector)
            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                smoother.reset(); last_spoken = None; print("  Buffer reset.")
            elif key == ord('d'):
                roi_selector.enter_draw_mode(frame)
            elif key == ord('c'):
                roi_selector.clear(); smoother.reset(); last_spoken = None
                print("  ROI cleared.")
            continue

        # ── Inference (on ROI region) ──────────────────────────────────────
        probs = predict_frame(model, roi_frame)

        # ── Temporal smoothing ─────────────────────────────────────────────
        committed_idx, committed_conf, candidate_idx = smoother.update(probs)

        committed_name = class_names[committed_idx] if committed_idx is not None else None
        candidate_name = class_names[candidate_idx]

        if committed_name is not None and committed_name != last_spoken:
            speak(committed_name)
            last_spoken = committed_name

        # ── Draw HUD on the raw (full) frame so the ROI rect is visible ───
        output_frame = frame.copy()
        draw_hud(output_frame, committed_name, committed_conf,
                 candidate_name, motion_score, False, smoother, roi_selector)

        cv2.imshow(window_name, output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            frame_count += 1
            label    = committed_name or candidate_name
            filename = f"street_sign_{label}_{frame_count}.jpg"
            cv2.imwrite(filename, roi_frame)   # save the ROI crop, not the full frame
            print(f"✓ Saved: {filename}")
        elif key == ord('r'):
            smoother.reset(); last_spoken = None; print("  Buffer reset.")
        elif key == ord('d'):
            roi_selector.enter_draw_mode(frame)
            print("  Draw mode: click and drag to set ROI.")
        elif key == ord('c'):
            roi_selector.clear(); smoother.reset(); last_spoken = None
            prev_gray = None
            print("  ROI cleared.")

    cap.release()
    cv2.destroyAllWindows()
    _tts.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()