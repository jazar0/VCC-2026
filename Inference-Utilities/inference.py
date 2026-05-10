import json

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE  = 224
SAT_SCALE = 1.8   # bump up if yellow still looks dull, 1.0 = no change

# ============================================================
# CUSTOM LAYER STUBS
# The model was saved with custom augmentation layers baked in.
# At inference time they are no-ops (training=False), but Keras
# still needs their class definitions to deserialise the model.
# We pass them explicitly via custom_objects at load time to
# avoid any decorator / registry namespace mismatches.
# ============================================================

class RandomHueSat(layers.Layer):
    def __init__(self, hue_delta=0.05, sat_lower=0.7, sat_upper=1.4, **kwargs):
        super().__init__(**kwargs)
        self.hue_delta = hue_delta
        self.sat_lower = sat_lower
        self.sat_upper = sat_upper
    def call(self, x, training=None): return x
    def get_config(self):
        return {**super().get_config(),
                "hue_delta": self.hue_delta,
                "sat_lower": self.sat_lower,
                "sat_upper": self.sat_upper}

class RandomGaussianNoise(layers.Layer):
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    def call(self, x, training=None): return x
    def get_config(self):
        return {**super().get_config(), "stddev": self.stddev}

class RandomShear(layers.Layer):
    def __init__(self, shear_range=0.1, **kwargs):
        super().__init__(**kwargs)
        self.shear_range = shear_range
    def call(self, x, training=None): return x
    def get_config(self):
        return {**super().get_config(), "shear_range": self.shear_range}

class RandomOcclusion(layers.Layer):
    def __init__(self, patch_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self.patch_ratio = patch_ratio
    def call(self, x, training=None): return x
    def get_config(self):
        return {**super().get_config(), "patch_ratio": self.patch_ratio}

# ============================================================
# LOAD MODEL & CLASSES
# ============================================================

CUSTOM_OBJECTS = {
    "RandomHueSat":        RandomHueSat,
    "RandomGaussianNoise": RandomGaussianNoise,
    "RandomShear":         RandomShear,
    "RandomOcclusion":     RandomOcclusion,
}

model = keras.models.load_model(
    "efficientnetv2b0_classifier.keras",
    custom_objects=CUSTOM_OBJECTS
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ============================================================
# PREPROCESSING
# Returns (model_input, preview_bgr) so we can show what the model sees
# ============================================================

def preprocess(frame, box, sat_scale=SAT_SCALE):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    img  = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # Saturation boost — HSV so only S channel is touched
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    preview_bgr = img.copy()   # save BGR version for display before channel swap

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img, preview_bgr

# ============================================================
# ROI BOX
# - Drag anywhere INSIDE the box → move the whole box
# - Drag a corner handle        → free resize from that corner
# - Drag an edge handle         → resize that edge only
# Keys: R = reset, Q = quit
# ============================================================

HANDLE_RADIUS = 10
MIN_BOX_SIZE  = 50

class ROIBox:
    def __init__(self, frame_w, frame_h):
        margin_x = frame_w // 4
        margin_y = frame_h // 4
        self.x1, self.y1 = margin_x, margin_y
        self.x2, self.y2 = frame_w - margin_x, frame_h - margin_y
        self.fw, self.fh = frame_w, frame_h
        self._drag = None
        self._ox, self._oy = 0, 0

    def reset(self, frame_w, frame_h):
        self.__init__(frame_w, frame_h)

    def coords(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def _handles(self):
        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2
        return {
            "tl": (self.x1, self.y1),
            "tr": (self.x2, self.y1),
            "bl": (self.x1, self.y2),
            "br": (self.x2, self.y2),
            "tc": (cx,      self.y1),
            "bc": (cx,      self.y2),
            "lc": (self.x1, cy),
            "rc": (self.x2, cy),
        }

    def _hit_handle(self, mx, my):
        for name, (hx, hy) in self._handles().items():
            if abs(mx - hx) <= HANDLE_RADIUS and abs(my - hy) <= HANDLE_RADIUS:
                return name
        return None

    def _inside(self, mx, my):
        return self.x1 < mx < self.x2 and self.y1 < my < self.y2

    def on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle = self._hit_handle(mx, my)
            if handle:
                self._drag = handle
                hx, hy = self._handles()[handle]
                self._ox, self._oy = mx - hx, my - hy
            elif self._inside(mx, my):
                self._drag = "move"
                self._ox = mx - self.x1
                self._oy = my - self.y1

        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            d = self._drag

            if d == "move":
                w = self.x2 - self.x1
                h = self.y2 - self.y1
                self.x1 = max(0, min(mx - self._ox, self.fw - w))
                self.y1 = max(0, min(my - self._oy, self.fh - h))
                self.x2 = self.x1 + w
                self.y2 = self.y1 + h
            else:
                dx = mx - self._ox
                dy = my - self._oy
                if "l" in d: self.x1 = max(0,       min(dx, self.x2 - MIN_BOX_SIZE))
                if "r" in d: self.x2 = min(self.fw, max(dx, self.x1 + MIN_BOX_SIZE))
                if "t" in d: self.y1 = max(0,       min(dy, self.y2 - MIN_BOX_SIZE))
                if "b" in d: self.y2 = min(self.fh, max(dy, self.y1 + MIN_BOX_SIZE))

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def draw(self, frame, bar_color):
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        # Dim everything outside the box
        dimmed = (frame * 0.35).astype(np.uint8)
        dimmed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        frame[:] = dimmed

        # Box border
        cv2.rectangle(frame, (x1, y1), (x2, y2), bar_color, 2)

        # Corner + edge handles
        for name, (hx, hy) in self._handles().items():
            cv2.circle(frame, (hx, hy), HANDLE_RADIUS, bar_color, -1, cv2.LINE_AA)

        # Hint
        hint = "Drag inside to move  |  Drag corners/edges to resize  |  R = reset  |  Q = quit"
        hint_y = min(y2 + 20, frame.shape[0] - 5)
        cv2.putText(frame, hint, (x1, hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

# ============================================================
# CAMERA LOOP
# ============================================================

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera error")
    exit()

ret, first = cap.read()
fh, fw = first.shape[:2]

roi = ROIBox(fw, fh)

WIN     = "Traffic Sign Classifier"
WIN_PRE = "Model Input Preview"

cv2.namedWindow(WIN)
cv2.namedWindow(WIN_PRE)
cv2.setMouseCallback(WIN, roi.on_mouse)

cv2.createTrackbar("Saturation x10", WIN_PRE, 18, 40, lambda v: None)
cv2.setTrackbarMin("Saturation x10", WIN_PRE, 10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    SAT_SCALE = cv2.getTrackbarPos("Saturation x10", WIN_PRE) / 10.0

    box                = roi.coords()
    input_img, preview = preprocess(frame, box, SAT_SCALE)
    preds              = model.predict(input_img, verbose=0)

    class_id   = np.argmax(preds)
    confidence = float(np.max(preds))

    label     = class_names[class_id]
    conf_str  = f"{confidence * 100:.1f}%"
    bar_color = (0, 200, 80)

    cv2.imshow(WIN_PRE, preview)

    roi.draw(frame, bar_color)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    bar_width = int(fw * confidence)
    cv2.rectangle(frame, (0, 85), (bar_width, 90), bar_color, -1)

    cv2.putText(frame, label,    (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, conf_str, (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color,       2, cv2.LINE_AA)

    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        roi.reset(fw, fh)

cap.release()
cv2.destroyAllWindows()