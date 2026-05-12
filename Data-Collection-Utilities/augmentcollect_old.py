"""
augmentcollect.py
Low-latency dual-monitor data collector
FIXED:
- real-time webcam preview
- aggressive V4L2 buffer flushing
- threaded camera
- correct folder/class alignment
- no stale-frame captures
"""

import argparse
import ctypes
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
import pygame
from PIL import Image, ImageEnhance, ImageFilter

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

L1_IMAGES_DIR = "L1-Images"
COLLECTED_DIR = "Collected-Data"

CAPTURES_PER_CLASS = 10
COUNTDOWN_SECS = 3

SUPPORTED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# ──────────────────────────────────────────────────────────────
# ASYNC LOW-LATENCY CAMERA - CONTINUOUS BACKGROUND READING
# ──────────────────────────────────────────────────────────────

class ThreadedCamera:
    def __init__(self, index=0):

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(index, backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")

        # MINIMALIST LOW-LATENCY SETTINGS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # MJPG reduces USB webcam latency
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        # Standard resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # High FPS for real-time performance
        self.cap.set(cv2.CAP_PROP_FPS, 120)

        # Background thread continuously reads frames
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """Background thread: continuously read camera frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def read(self):
        """Get latest frame captured by background thread"""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        """Stop background thread and release camera"""
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

# ──────────────────────────────────────────────────────────────
# SDL DISPLAY DETECTION
# ──────────────────────────────────────────────────────────────

def _find_pygame_sdl2():
    import glob
    import site

    for sp in site.getsitepackages() + [site.getusersitepackages()]:

        hits = glob.glob(
            os.path.join(sp, "pygame.libs", "libSDL2-2*.so*")
        )

        if hits:
            return hits[0]

    from ctypes.util import find_library

    return find_library("SDL2") or "libSDL2.so"

def get_monitor_bounds():

    class SDL_Rect(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
            ("w", ctypes.c_int),
            ("h", ctypes.c_int),
        ]

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

# ──────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ──────────────────────────────────────────────────────────────

def discover_images(folder):

    return [
        f
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

def class_name_from_filename(filename):

    return os.path.splitext(filename)[0].strip()

def group_images_by_class(images):

    from collections import OrderedDict

    groups = OrderedDict()

    for img in images:

        cls = class_name_from_filename(img)

        groups.setdefault(cls, []).append(img)

    return groups

def ensure_class_dir(class_name):

    path = os.path.join(COLLECTED_DIR, class_name)

    os.makedirs(path, exist_ok=True)

    return path

# ──────────────────────────────────────────────────────────────
# AUGMENTATIONS
# ──────────────────────────────────────────────────────────────

def do_zoom(img):
    """Distance variations & image size variations"""
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
    """Rotation - keep moderate"""
    a = random.uniform(-40, 40)
    return img.rotate(a, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255)), f"rotation {a:+.1f}deg"

def do_shear(img):
    """Aspect ratio changes via shearing"""
    s = random.uniform(-0.4, 0.4)
    w, h = img.size
    img = img.transform((w, h), Image.AFFINE, (1, s, -s*h/2, 0, 1, 0),
                        resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return img, f"shear {s:+.2f}"

def do_scale(img):
    """Scaling with independent x/y - image size variations"""
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
    """Direct aspect ratio change"""
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
    """Artificial noise - increased"""
    sigma = random.uniform(10, 40)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr), f"gaussian noise s={sigma:.1f}"

def do_occlusion(img):
    """Occlusion - keep as is"""
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
    """Varied lighting conditions & day/night scenarios"""
    f = random.uniform(0.4, 1.8)
    return ImageEnhance.Brightness(img).enhance(f), f"brightness x{f:.2f}"

def do_saturation(img):
    """Minor color changes - increased intensity"""
    f = random.uniform(0.3, 1.8)
    return ImageEnhance.Color(img).enhance(f), f"saturation x{f:.2f}"

def do_contrast(img):
    """Varied lighting conditions - increased"""
    f = random.uniform(0.5, 1.8)
    return ImageEnhance.Contrast(img).enhance(f), f"contrast x{f:.2f}"

def do_color_tint(img):
    """Minor color changes & natural illumination variations"""
    r, g, b = [random.randint(-40, 40) for _ in range(3)]
    arr = np.array(img, dtype=np.int16)
    arr[:,:,0] = np.clip(arr[:,:,0]+r, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]+g, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]+b, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), f"tint R{r:+d} G{g:+d} B{b:+d}"

def do_blur(img):
    """Blur for varied focus/distance - increased"""
    radius = random.uniform(0.5, 4.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), f"blur r={radius:.1f}px"

def random_augmentations(pil_img):
    """Rich augmentation pool with all transformations"""
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

# ──────────────────────────────────────────────────────────────
# SHARED MEMORY VIEWER
# ──────────────────────────────────────────────────────────────

SHM_HEADER = 9

def shm_write(shm, rgb_bytes, w, h):

    buf = shm.buf

    buf[1:5] = struct.pack("<I", w)
    buf[5:9] = struct.pack("<I", h)

    nbytes = w * h * 3

    buf[SHM_HEADER:SHM_HEADER+nbytes] = rgb_bytes

    buf[0] = 1

def shm_read(shm):

    buf = shm.buf

    if buf[0] != 1:
        return None

    buf[0] = 0

    w = struct.unpack("<I", bytes(buf[1:5]))[0]
    h = struct.unpack("<I", bytes(buf[5:9]))[0]

    nbytes = w * h * 3

    data = bytes(buf[SHM_HEADER:SHM_HEADER+nbytes])

    return w, h, data

def pil_to_rgb_bytes(img, tw, th):

    img = img.convert("RGB")

    img = img.resize((tw, th), Image.LANCZOS)

    return img.tobytes(), tw, th

def viewer_process(shm_name, vw, vh, stop_event):

    pygame.init()

    screen = pygame.display.set_mode((vw, vh), pygame.NOFRAME)

    shm = shared_memory.SharedMemory(name=shm_name)

    clock = pygame.time.Clock()

    current = None

    while not stop_event.is_set():

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                stop_event.set()

        result = shm_read(shm)

        if result:

            w, h, data = result

            current = pygame.image.frombuffer(
                data,
                (w,h),
                "RGB"
            ).convert()

        screen.fill((0,0,0))

        if current:
            screen.blit(current, (0,0))

        pygame.display.flip()

        clock.tick(60)

    shm.close()

# ──────────────────────────────────────────────────────────────
# PREVIEW
# ──────────────────────────────────────────────────────────────

def webcam_preview_phase(cap):

    print("\n[PREVIEW] SPACE=start ESC=quit\n")

    win = "Preview"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.putText(
            frame,
            "SPACE = START",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord(' '), 13):
            break

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2.0)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(script_dir, L1_IMAGES_DIR)

    images = discover_images(folder)

    if not images:
        print("[ERROR] No images found.")
        sys.exit(1)

    class_groups = group_images_by_class(images)

    class_names = list(class_groups.keys())

    os.makedirs(COLLECTED_DIR, exist_ok=True)

    print(f"[INFO] Opening camera {args.camera}")

    cap = ThreadedCamera(args.camera)

    webcam_preview_phase(cap)

    monitors = get_monitor_bounds()

    if len(monitors) < 2:
        view_mon = monitors[0]
    else:
        view_mon = monitors[1]

    _, _, VW, VH = view_mon

    shm_size = SHM_HEADER + VW * VH * 3 + 1024

    shm = shared_memory.SharedMemory(
        create=True,
        size=shm_size
    )

    shm.buf[0] = 0

    stop_event = mp.Event()

    viewer = mp.Process(
        target=viewer_process,
        args=(shm.name, VW, VH, stop_event),
        daemon=True
    )

    viewer.start()

    cls_idx = 0
    capture_count = {}
    for cls in class_names:
        capture_count[cls] = 0

    loop_num = 0

    try:
        while True:

            loop_num += 1
            print(f"\n{'='*60}\n  COLLECTION LOOP {loop_num}\n{'='*60}\n")

            for cls_idx, cls in enumerate(class_names):

                print(f"\n=== Loop {loop_num} - {cls} ===")

                path = os.path.join(
                    folder,
                    class_groups[cls][0]
                )

                original = Image.open(path).convert("RGB")

                out_dir = ensure_class_dir(cls)

                for i in range(CAPTURES_PER_CLASS):

                    aug, descs = random_augmentations(original)

                    raw, w, h = pil_to_rgb_bytes(aug, VW, VH)

                    shm_write(shm, raw, w, h)
                    time.sleep(0.5)

                    ret, frame = cap.read()

                    if not ret:
                        print("[WARN] Capture failed.")
                        continue

                    capture_count[cls] += 1
                    filename = f"{cls}_{capture_count[cls]:06d}.jpg"

                    cv2.imwrite(
                        os.path.join(out_dir, filename),
                        frame
                    )

                    print(f"[CAPTURE] {filename}")

                    preview = frame.copy()

                    cv2.putText(
                        preview,
                        filename,
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )

                    cv2.imshow("Live Camera", preview)

                    cv2.waitKey(1)

                    time.sleep(args.interval)

            print(f"\n[INFO] Completed loop {loop_num} - restarting collection cycle...")

    except KeyboardInterrupt:
        print("\n\n[INFO] Collection stopped by user (Ctrl+C)")

    finally:
        stop_event.set()

        viewer.join(timeout=2)

        shm.close()
        shm.unlink()

        cap.release()

        cv2.destroyAllWindows()

        print("\n[DONE]")

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    main()