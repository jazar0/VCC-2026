"""
inferTest.py
EfficientNetV2-S Street Sign Inference - Augmentation Test Mode

Primary monitor  : live camera feed + model prediction overlay
Secondary monitor: continuously refreshing augmented L1-Image for the
                   current class, so the model is always looking at a
                   known sign.

The filename of each L1-Image is the class name, so expected vs actual
output is known.  The user manually advances classes with SPACE / ENTER;
there is no automatic interval switching.

Controls (camera window):
  SPACE / ENTER  - next class
  B              - previous class
  Q / ESC        - quit
  S              - save current camera frame
"""

import json
import multiprocessing as mp
import os
import random
import struct
import sys
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
CLASS_NAMES_FILE = "class_names.json"
INPUT_SIZE = 384
CONFIDENCE_THRESHOLD = 0.5

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# How many augmented frames are shown per second on the secondary monitor
AUGMENT_FPS = 4

# ============================================================================
# MODEL LOADING  (same pattern as infer.py)
# ============================================================================

def discover_models():
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Models directory not found: {MODELS_DIR}")
        return []
    return sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".keras"))


def choose_model():
    models = discover_models()
    if not models:
        print("❌ No .keras models found in:", MODELS_DIR)
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


def load_model_keras(model_path):
    from tensorflow import keras
    print(f"\n✓ Loading model: {model_path}")
    return keras.models.load_model(model_path)


def load_class_names():
    with open(CLASS_NAMES_FILE, "r") as f:
        return sorted(json.load(f))

# ============================================================================
# L1-IMAGE DISCOVERY
# ============================================================================

def discover_l1_images():
    """Return dict: class_name -> absolute path for every L1-Image found."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "L1-Images")

    if not os.path.isdir(folder):
        print(f"❌ L1-Images directory not found: {folder}")
        return {}, folder

    mapping = {}
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTS:
            cls = os.path.splitext(fname)[0].strip()
            mapping[cls] = os.path.join(folder, fname)

    return mapping, folder

# ============================================================================
# AUGMENTATIONS  (copied verbatim from augmentcollect.py)
# ============================================================================

def do_zoom(img):
    w, h = img.size
    f = random.uniform(0.5, 1.8)
    nw, nh = int(w*f), int(h*f)
    img = img.resize((nw, nh), Image.LANCZOS)
    if f > 1:
        img = img.crop(((nw-w)//2, (nh-h)//2, (nw-w)//2+w, (nh-h)//2+h))
    else:
        out = Image.new(img.mode, (w, h), (255, 255, 255))
        out.paste(img, ((w-nw)//2, (h-nh)//2))
        img = out
    return img, f"zoom {'in' if f>1 else 'out'} x{f:.2f}"

def do_rotation(img):
    a = random.uniform(-40, 40)
    return img.rotate(a, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255)), f"rotation {a:+.1f}deg"

def do_shear(img):
    s = random.uniform(-0.4, 0.4)
    w, h = img.size
    img = img.transform((w, h), Image.AFFINE, (1, s, -s*h/2, 0, 1, 0),
                        resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return img, f"shear {s:+.2f}"

def do_scale(img):
    sx, sy = random.uniform(0.6, 1.5), random.uniform(0.6, 1.5)
    w, h = img.size
    nw, nh = int(w*sx), int(h*sy)
    resized = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new(img.mode, (w, h), (255, 255, 255))
    ox, oy = max(0, (w-nw)//2), max(0, (h-nh)//2)
    cw, ch = min(nw, w), min(nh, h)
    cx0, cy0 = max(0, (nw-w)//2), max(0, (nh-h)//2)
    out.paste(resized.crop((cx0, cy0, cx0+cw, cy0+ch)), (ox, oy))
    return out, f"scale sx={sx:.2f} sy={sy:.2f}"

def do_aspect_ratio(img):
    w, h = img.size
    ar = random.uniform(0.6, 1.6)
    new_w = int(w)
    new_h = int(h / ar)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new(img.mode, (w, h), (255, 255, 255))
    y_offset = max(0, (h - new_h) // 2)
    out.paste(resized, (0, y_offset))
    return out, f"aspect ratio x{ar:.2f}"

def do_noise(img):
    sigma = random.uniform(10, 40)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr), f"gaussian noise s={sigma:.1f}"

def do_occlusion(img):
    w, h = img.size
    num = random.randint(1, 4)
    arr = np.array(img)
    for _ in range(num):
        bw = random.randint(w//12, w//4)
        bh = random.randint(h//12, h//4)
        bx = random.randint(0, w-bw)
        by = random.randint(0, h-bh)
        arr[by:by+bh, bx:bx+bw] = random.choice([(0,0,0),(255,255,255),(128,128,128)])
    return Image.fromarray(arr), f"occlusion x{num} patches"

def do_brightness(img):
    f = random.uniform(0.4, 1.8)
    return ImageEnhance.Brightness(img).enhance(f), f"brightness x{f:.2f}"

def do_saturation(img):
    f = random.uniform(0.3, 1.8)
    return ImageEnhance.Color(img).enhance(f), f"saturation x{f:.2f}"

def do_contrast(img):
    f = random.uniform(0.5, 1.8)
    return ImageEnhance.Contrast(img).enhance(f), f"contrast x{f:.2f}"

def do_color_tint(img):
    r, g, b = [random.randint(-40, 40) for _ in range(3)]
    arr = np.array(img, dtype=np.int16)
    arr[:,:,0] = np.clip(arr[:,:,0]+r, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]+g, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]+b, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), f"tint R{r:+d} G{g:+d} B{b:+d}"

def do_blur(img):
    radius = random.uniform(0.5, 4.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), f"blur r={radius:.1f}px"

def random_augmentations(pil_img):
    pool = [
        (do_zoom,           0.70),
        (do_rotation,       0.55),
        (do_shear,          0.65),
        (do_scale,          0.70),
        (do_aspect_ratio,   0.60),
        (do_noise,          0.65),
        (do_occlusion,      0.40),
        (do_brightness,     0.75),
        (do_saturation,     0.70),
        (do_contrast,       0.75),
        (do_color_tint,     0.65),
        (do_blur,           0.50),
    ]
    chosen = [fn for fn, p in pool if random.random() < p]
    if not chosen:
        chosen = [random.choice([fn for fn, _ in pool])]
    random.shuffle(chosen)
    chosen = chosen[:7]
    img, descs = pil_img.copy(), []
    for fn in chosen:
        img, desc = fn(img)
        descs.append(desc)
    return img, descs

# ============================================================================
# SHARED MEMORY  (same protocol as augmentcollect.py)
# ============================================================================

SHM_HEADER = 9

def shm_write(shm, rgb_bytes, w, h):
    buf = shm.buf
    buf[1:5] = struct.pack("<I", w)
    buf[5:9] = struct.pack("<I", h)
    nbytes = w * h * 3
    buf[SHM_HEADER:SHM_HEADER + nbytes] = rgb_bytes
    buf[0] = 1

def shm_read(shm):
    buf = shm.buf
    if buf[0] != 1:
        return None
    buf[0] = 0
    w = struct.unpack("<I", bytes(buf[1:5]))[0]
    h = struct.unpack("<I", bytes(buf[5:9]))[0]
    nbytes = w * h * 3
    data = bytes(buf[SHM_HEADER:SHM_HEADER + nbytes])
    return w, h, data

# ============================================================================
# MONITOR DETECTION  (same SDL2 approach as augmentcollect.py)
# ============================================================================

def _find_pygame_sdl2():
    import glob
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        hits = glob.glob(os.path.join(sp, "pygame.libs", "libSDL2-2*.so*"))
        if hits:
            return hits[0]
    from ctypes.util import find_library
    return find_library("SDL2") or "libSDL2.so"

def get_monitor_bounds():
    import ctypes

    class SDL_Rect(ctypes.Structure):
        _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                    ("w", ctypes.c_int), ("h", ctypes.c_int)]
    try:
        sdl = ctypes.CDLL(_find_pygame_sdl2())
        sdl.SDL_Init(0x00000020)
        n = sdl.SDL_GetNumVideoDisplays()
        out = []
        for i in range(n):
            r = SDL_Rect()
            sdl.SDL_GetDisplayBounds(i, ctypes.byref(r))
            out.append((r.x, r.y, r.w, r.h))
        sdl.SDL_Quit()
        return out
    except Exception:
        info = pygame.display.Info()
        return [(0, 0, info.current_w, info.current_h)]

# ============================================================================
# SECONDARY MONITOR  - pygame viewer process
# ============================================================================

def viewer_process(shm_name, vw, vh, stop_event):
    """
    Runs on the secondary monitor.
    Reads augmented frames from shared memory and renders them full-screen
    with a label overlay showing:
      - current class name (expected)
      - augmentation tags
      - class index / total
    An additional text area at the bottom is reserved for the prediction
    result written by the main process into the shm label slot.
    """
    pygame.init()

    monitors = get_monitor_bounds()
    if len(monitors) >= 2:
        mx, my, _, _ = monitors[1]
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mx},{my}"

    screen = pygame.display.set_mode((vw, vh), pygame.FULLSCREEN | pygame.NOFRAME)
    
    shm = shared_memory.SharedMemory(name=shm_name)
    clock = pygame.time.Clock()
    current = None

    font_large  = pygame.font.SysFont("monospace", 36, bold=True)
    font_medium = pygame.font.SysFont("monospace", 26)
    font_small  = pygame.font.SysFont("monospace", 20)

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()

        result = shm_read(shm)
        if result:
            w, h, data = result
            current = pygame.image.frombuffer(data, (w, h), "RGB").convert()

        screen.fill((10, 10, 10))

        if current:
            screen.blit(current, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    shm.close()
    pygame.quit()

# ============================================================================
# AUGMENTATION THREAD  - pushes new frames into shared memory continuously
# ============================================================================

class AugmentThread(threading.Thread):
    """
    Continuously generates augmented versions of the current L1-Image and
    writes them into shared memory so the secondary monitor stays animated.
    Switches source image whenever `set_class` is called.
    """
    def __init__(self, shm, vw, vh):
        super().__init__(daemon=True)
        self.shm = shm
        self.vw = vw
        self.vh = vh
        self._lock = threading.Lock()
        self._pil_img = None
        self._running = True
        self.last_descs = []

    def set_class(self, pil_img):
        with self._lock:
            self._pil_img = pil_img.copy()

    def stop(self):
        self._running = False

    def run(self):
        interval = 1.0 / AUGMENT_FPS
        while self._running:
            with self._lock:
                img = self._pil_img.copy() if self._pil_img else None

            if img is not None:
                aug, descs = random_augmentations(img)
                self.last_descs = descs

                # Fit augmented image into monitor dimensions (letterbox)
                aug_rgb = aug.convert("RGB")
                aug_rgb.thumbnail((self.vw, self.vh), Image.LANCZOS)

                # Centre on black canvas
                canvas = Image.new("RGB", (self.vw, self.vh), (10, 10, 10))
                ox = (self.vw - aug_rgb.width) // 2
                oy = (self.vh - aug_rgb.height) // 2
                canvas.paste(aug_rgb, (ox, oy))

                raw = canvas.tobytes()
                shm_write(self.shm, raw, self.vw, self.vh)

            time.sleep(interval)

# ============================================================================
# CAMERA INFERENCE HELPERS
# ============================================================================

def apply_zoom(frame, zoom_pct):
    if zoom_pct == 0:
        return frame
    h, w = frame.shape[:2]
    min_crop = 0.20
    crop_factor = 1.0 - (zoom_pct / 100.0) * (1.0 - min_crop)
    crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
    y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_frame(model, frame, class_names):
    batch = preprocess_frame(frame)
    predictions = model.predict(batch, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_conf = float(predictions[0][pred_idx])
    pred_name = class_names[pred_idx]
    top3_idx  = np.argsort(predictions[0])[-3:][::-1]
    top3      = [(class_names[i], float(predictions[0][i])) for i in top3_idx]
    return pred_name, pred_conf, top3


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


def draw_overlay(frame, expected_class, pred_name, pred_conf, top3,
                 cls_idx, total_classes, zoom_pct, class_change_time=None):
    h, w = frame.shape[:2]
    
    # Check if we're in the 3-second buffer period after class change
    in_buffer = False
    if class_change_time is not None:
        elapsed = time.time() - class_change_time
        in_buffer = elapsed < 3.0

    # Expected (ground truth)
    draw_text_with_bg(frame, f"Expected: {expected_class}",
                      (10, 38), 0.85, 2, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Predicted with status
    if in_buffer:
        # During buffer period, show "Warming up..." instead of judging
        status = "Warming up..."
        pred_text = f"Predicted: {pred_name} ({pred_conf:.1%}) {status}"
    else:
        # After buffer period, show correct/wrong judgment
        match = pred_name.strip().lower() == expected_class.strip().lower()
        status = "CORRECT" if match else "WRONG"
        pred_text = f"Predicted: {pred_name} ({pred_conf:.1%}) {status}"
    draw_text_with_bg(frame, pred_text, (10, 78), 0.75, 2, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Top-3 predictions
    draw_text_with_bg(frame, "Top 3:", (10, 120), 0.6, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))
    for i, (cls, conf) in enumerate(top3):
        draw_text_with_bg(frame, f"{i+1}. {cls}: {conf:.1%}", (10, 148 + i * 26),
                         0.55, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Class progress
    draw_text_with_bg(frame, f"Class {cls_idx+1}/{total_classes}",
                     (10, h - 55), 0.55, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Zoom level
    draw_text_with_bg(frame, f"Zoom: {zoom_pct}%",
                     (10, h - 30), 0.5, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    # Controls
    draw_text_with_bg(frame, "SPACE/ENTER=next  B=prev  S=save  Q/ESC=quit",
                     (10, h - 8), 0.42, 1, bg_color=(0, 0, 0), text_color=(255, 255, 255))

    return frame

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EfficientNetV2-S  -  Inference Test Mode (Augmented L1-Images)")
    print("="*70)

    # ── Class names ──────────────────────────────────────────────────────────
    class_names = load_class_names()
    print(f"\n✓ Loaded {len(class_names)} model classes")

    # ── L1-Images ────────────────────────────────────────────────────────────
    l1_map, l1_folder = discover_l1_images()
    if not l1_map:
        print("❌  No L1-Images found - aborting.")
        return

    # Only keep classes that exist in L1-Images
    test_classes = [c for c in sorted(l1_map.keys())]
    print(f"✓ Found {len(test_classes)} L1-Image classes in: {l1_folder}")

    # ── Model ────────────────────────────────────────────────────────────────
    result = choose_model()
    if result is None:
        return
    model_path, model_name = result
    model = load_model_keras(model_path)
    print(f"✓ Model loaded")

    # ── Camera ───────────────────────────────────────────────────────────────
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    print("✓ Camera opened")

    # ── Monitors ─────────────────────────────────────────────────────────────
    monitors = get_monitor_bounds()
    view_mon = monitors[1] if len(monitors) >= 2 else monitors[0]
    _, _, VW, VH = view_mon
    print(f"✓ Secondary monitor: {VW}×{VH}")

    # ── Shared memory ────────────────────────────────────────────────────────
    shm_size = SHM_HEADER + VW * VH * 3 + 1024
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    shm.buf[0] = 0

    stop_event = mp.Event()

    # ── Viewer process ───────────────────────────────────────────────────────
    viewer = mp.Process(
        target=viewer_process,
        args=(shm.name, VW, VH, stop_event),
        daemon=True,
    )
    viewer.start()

    # ── Augmentation thread ──────────────────────────────────────────────────
    aug_thread = AugmentThread(shm, VW, VH)

    # ── Camera window with zoom slider ───────────────────────────────────────
    WIN = f"Inference Test - {model_name}"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("Zoom %", WIN, 0, 100, lambda _: None)

    # ── State ────────────────────────────────────────────────────────────────
    cls_idx              = 0
    frame_count          = 0
    last_class_change_time = None

    def load_class(idx):
        nonlocal last_class_change_time
        cls = test_classes[idx]
        pil = Image.open(l1_map[cls]).convert("RGB")
        aug_thread.set_class(pil)
        last_class_change_time = time.time()  # Record time of class change
        print(f"\n→ Class {idx+1}/{len(test_classes)}: {cls}")
        return cls

    current_class = load_class(cls_idx)
    aug_thread.start()

    print("\nStarting inference test...")
    print("  SPACE / ENTER  → next class")
    print("  B              → previous class")
    print("  S              → save frame")
    print("  Q / ESC        → quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            zoom_pct = cv2.getTrackbarPos("Zoom %", WIN)
            frame    = apply_zoom(frame, zoom_pct)

            pred_name, pred_conf, top3 = predict_frame(model, frame, class_names)

            output = frame.copy()
            output = draw_overlay(output, current_class, pred_name, pred_conf,
                                  top3, cls_idx, len(test_classes), zoom_pct,
                                  class_change_time=last_class_change_time)

            cv2.imshow(WIN, output)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):                           # Q / ESC
                print("\nQuitting...")
                break

            elif key in (ord(' '), 13):                         # SPACE / ENTER → next
                cls_idx = (cls_idx + 1) % len(test_classes)
                current_class = load_class(cls_idx)

            elif key == ord('b'):                               # B → previous
                cls_idx = (cls_idx - 1) % len(test_classes)
                current_class = load_class(cls_idx)

            elif key == ord('s'):                               # S → save
                frame_count += 1
                fname = f"inferTest_{current_class}_{frame_count:04d}.jpg"
                cv2.imwrite(fname, frame)
                print(f"✓ Saved: {fname}")

    finally:
        stop_event.set()
        aug_thread.stop()
        aug_thread.join(timeout=2)
        viewer.join(timeout=2)
        shm.close()
        shm.unlink()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[DONE]")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()