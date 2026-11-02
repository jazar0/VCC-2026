"""
collect.py
Dual-monitor manual data collector.

Workflow:
  1. Displays a pre-made reference image for the current class on monitor 2.
  2. Operator positions the physical object, then presses SPACE.
  3. A short countdown fires, then the webcam captures one frame.
  4. The capture is saved to  Collected-Data/<class>/
  5. Advances to the next image/class automatically.

Directory layout expected:
  Reference-Images/
      apple.jpg          ← filename stem  =  class name
      banana.png
      ...
  Collected-Data/        ← created automatically
      apple/
      banana/
      ...

Usage:
  python collect.py
  python collect.py --camera 1 --countdown 3 --images-dir My-Images
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import struct
import sys
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np
import pygame
from PIL import Image

# ──────────────────────────────────────────────────────────────
# CONFIG  (override via CLI flags)
# ──────────────────────────────────────────────────────────────

IMAGES_DIR    = "Augmented_Dataset_Total"   # folder of pre-made class images
COLLECTED_DIR = "Collected-Data"     # output root
COUNTDOWN_SECS = 3                   # seconds between SPACE and shutter

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# ──────────────────────────────────────────────────────────────
# THREADED CAMERA  (identical low-latency design to augmentcollect)
# ──────────────────────────────────────────────────────────────

class ThreadedCamera:
    def __init__(self, index=0):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

# ──────────────────────────────────────────────────────────────
# SDL / MONITOR DETECTION
# ──────────────────────────────────────────────────────────────

def _find_pygame_sdl2():
    import glob, site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        hits = glob.glob(os.path.join(sp, "pygame.libs", "libSDL2-2*.so*"))
        if hits:
            return hits[0]
    from ctypes.util import find_library
    return find_library("SDL2") or "libSDL2.so"

def get_monitor_bounds():
    class SDL_Rect(ctypes.Structure):
        _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                    ("w", ctypes.c_int), ("h", ctypes.c_int)]
    try:
        sdl = ctypes.CDLL(_find_pygame_sdl2())
        sdl.SDL_Init(0x00000020)
        n   = sdl.SDL_GetNumVideoDisplays()
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

# ──────────────────────────────────────────────────────────────
# SHARED MEMORY  (same header format as augmentcollect)
# ──────────────────────────────────────────────────────────────

SHM_HEADER = 9   # 1 byte flag + 4 bytes width + 4 bytes height

def shm_write(shm, rgb_bytes, w, h):
    buf = shm.buf
    buf[1:5] = struct.pack("<I", w)
    buf[5:9] = struct.pack("<I", h)
    nbytes = w * h * 3
    buf[SHM_HEADER : SHM_HEADER + nbytes] = rgb_bytes
    buf[0] = 1   # mark as fresh

def shm_read(shm):
    buf = shm.buf
    if buf[0] != 1:
        return None
    buf[0] = 0
    w = struct.unpack("<I", bytes(buf[1:5]))[0]
    h = struct.unpack("<I", bytes(buf[5:9]))[0]
    nbytes = w * h * 3
    return w, h, bytes(buf[SHM_HEADER : SHM_HEADER + nbytes])

def pil_to_rgb_bytes(pil_img, tw, th):
    img = pil_img.convert("RGB").resize((tw, th), Image.LANCZOS)
    return img.tobytes(), tw, th

# ──────────────────────────────────────────────────────────────
# VIEWER PROCESS  (second monitor – shows reference image + label)
# ──────────────────────────────────────────────────────────────

def viewer_process(shm_name, vw, vh, stop_event, mon_x=0, mon_y=0):
    """Runs in a separate process; renders whatever is in shared memory."""
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon_x},{mon_y}"
    pygame.init()
    screen  = pygame.display.set_mode((vw, vh), pygame.NOFRAME)
    pygame.display.set_caption("Reference")
    shm     = shared_memory.SharedMemory(name=shm_name)
    clock   = pygame.time.Clock()
    current = None

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()

        result = shm_read(shm)
        if result:
            w, h, data = result
            current = pygame.image.frombuffer(data, (w, h), "RGB").convert()

        screen.fill((0, 0, 0))
        if current:
            screen.blit(current, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    shm.close()

# ──────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ──────────────────────────────────────────────────────────────

def discover_images(folder):
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

def class_from_filename(filename):
    return os.path.splitext(filename)[0].strip()

def ensure_class_dir(class_name):
    path = os.path.join(COLLECTED_DIR, class_name)
    os.makedirs(path, exist_ok=True)
    return path

def add_label_overlay(pil_img, label, countdown=None):
    """
    Burn class name (and optional countdown) into a copy of the PIL image.
    Returns a new PIL image – the original is untouched.
    """
    import PIL.ImageDraw, PIL.ImageFont
    img  = pil_img.copy().convert("RGB")
    draw = PIL.ImageDraw.Draw(img)
    w, h = img.size

    # Semi-transparent black bar at bottom
    bar_h = max(60, h // 10)
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 160))
    img.paste(overlay, (0, h - bar_h), overlay)

    # Try to load a decent font, fall back to default
    font_size = max(32, bar_h - 16)
    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = PIL.ImageFont.load_default()

    text = label if countdown is None else f"{label}  –  {countdown}"
    draw.text((16, h - bar_h + 8), text, fill=(255, 255, 255), font=font)
    return img

# ──────────────────────────────────────────────────────────────
# WEBCAM PREVIEW PHASE
# ──────────────────────────────────────────────────────────────

def webcam_preview_phase(cap):
    print("\n[PREVIEW]  Adjust camera position.")
    print("           Press SPACE or ENTER to start collection.  ESC to quit.\n")
    win = "Camera Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "SPACE = START  |  ESC = QUIT",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), 13):   # SPACE or ENTER
            break
        if key == 27:               # ESC
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ──────────────────────────────────────────────────────────────

def run_collection(cap, class_names, image_paths, folder, shm, VW, VH, countdown_secs):
    """
    Iterate over every class:
      - Show reference image on monitor 2
      - Wait for SPACE on the live-camera window
      - Count down, capture, save
    Returns total captures saved.
    """
    capture_counts = {cls: 0 for cls in class_names}
    os.makedirs(COLLECTED_DIR, exist_ok=True)

    cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)

    for cls, img_file in zip(class_names, image_paths):
        pil_ref = Image.open(os.path.join(folder, img_file)).convert("RGB")
        out_dir = ensure_class_dir(cls)

        # Push reference image (with label) to monitor 2
        labeled = add_label_overlay(pil_ref, cls)
        raw, w, h = pil_to_rgb_bytes(labeled, VW, VH)
        shm_write(shm, raw, w, h)

        print(f"\n{'─'*55}")
        print(f"  CLASS : {cls}")
        print(f"  File  : {img_file}")
        print(f"  Output: {out_dir}")
        print(f"  → Position object and press SPACE (ESC = skip class, Q = quit)")
        print(f"{'─'*55}")

        # ── Wait for operator keypress ──────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            cv2.putText(display, f"Class: {cls}",
                        (10, 36),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
            cv2.putText(display, "SPACE=capture  ESC=skip  Q=quit",
                        (10, 72),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("Live Camera", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == 27:           # ESC – skip this class
                print(f"  [SKIP] {cls}")
                break
            if key in (ord("q"), ord("Q")):
                print("\n[QUIT] User exited early.")
                cv2.destroyAllWindows()
                return capture_counts

            # Also let the viewer process close things gracefully
        else:
            continue    # ESC branch: move to next class

        # ── Countdown ─────────────────────────────────────────
        deadline = time.time() + countdown_secs
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ret, frame = cap.read()
            if not ret:
                continue

            # Update monitor-2 overlay with countdown number
            tick_label = add_label_overlay(pil_ref, cls, countdown=int(remaining) + 1)
            raw2, w2, h2 = pil_to_rgb_bytes(tick_label, VW, VH)
            shm_write(shm, raw2, w2, h2)

            display = frame.copy()
            cv2.putText(display, f"Capturing in {int(remaining)+1}...",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)
            cv2.imshow("Live Camera", display)
            cv2.waitKey(1)

        # ── Capture ───────────────────────────────────────────
        # Flush a few frames so we get the freshest one
        for _ in range(5):
            cap.read()
            time.sleep(0.01)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"  [WARN] Capture failed for {cls}, skipping.")
            continue

        capture_counts[cls] += 1
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{cls}_{capture_counts[cls]:06d}_{ts}.jpg"
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, frame)
        print(f"  [SAVED] {save_path}")

        # Brief confirmation display
        confirm = frame.copy()
        cv2.putText(confirm, f"SAVED: {filename}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        cv2.imshow("Live Camera", confirm)
        cv2.waitKey(800)

        # Restore clean reference image on monitor 2
        raw3, w3, h3 = pil_to_rgb_bytes(labeled, VW, VH)
        shm_write(shm, raw3, w3, h3)

    cv2.destroyAllWindows()
    return capture_counts

# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manual dual-monitor webcam data collector."
    )
    parser.add_argument("--camera",     type=int,   default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--countdown",  type=int,   default=COUNTDOWN_SECS,
                        help="Seconds between SPACE and shutter (default: 3)")
    parser.add_argument("--images-dir", type=str,   default=IMAGES_DIR,
                        help=f"Folder of reference images (default: {IMAGES_DIR})")
    parser.add_argument("--loop",       action="store_true",
                        help="Repeat collection cycle indefinitely until Q/Ctrl+C")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder     = os.path.join(script_dir, args.images_dir)

    if not os.path.isdir(folder):
        print(f"[ERROR] Images directory not found: {folder}")
        sys.exit(1)

    image_files = discover_images(folder)
    if not image_files:
        print(f"[ERROR] No supported images found in {folder}")
        sys.exit(1)

    class_names = [class_from_filename(f) for f in image_files]
    print(f"[INFO] Found {len(class_names)} class(es): {', '.join(class_names)}")

    # ── Camera ────────────────────────────────────────────────
    print(f"[INFO] Opening camera {args.camera} …")
    cap = ThreadedCamera(args.camera)
    webcam_preview_phase(cap)

    # ── Monitors ──────────────────────────────────────────────
    monitors = get_monitor_bounds()
    view_mon = monitors[1] if len(monitors) >= 2 else monitors[0]
    VX, VY, VW, VH = view_mon
    print(f"[INFO] Viewer monitor: {VW}×{VH} at ({VX},{VY})")

    # ── Shared memory ─────────────────────────────────────────
    shm_size = SHM_HEADER + VW * VH * 3 + 1024
    shm      = shared_memory.SharedMemory(create=True, size=shm_size)
    shm.buf[0] = 0

    stop_event = mp.Event()
    viewer     = mp.Process(
        target=viewer_process,
        args=(shm.name, VW, VH, stop_event, VX, VY),
        daemon=True,
    )
    viewer.start()

    # ── Collection ────────────────────────────────────────────
    loop_num = 0
    try:
        while True:
            loop_num += 1
            if args.loop:
                print(f"\n{'='*55}\n  COLLECTION LOOP {loop_num}\n{'='*55}")
            counts = run_collection(
                cap, class_names, image_files, folder,
                shm, VW, VH, args.countdown
            )
            print("\n[SUMMARY]")
            for cls, n in counts.items():
                print(f"  {cls:<30} {n} capture(s)")

            if not args.loop:
                break

            # Brief pause between loops
            print("\n[INFO] Loop complete – press SPACE to start next loop or Q to quit.")
            cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Live Camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n[INFO] Collection stopped.")

    finally:
        stop_event.set()
        viewer.join(timeout=2)
        shm.close()
        shm.unlink()
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE]")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()