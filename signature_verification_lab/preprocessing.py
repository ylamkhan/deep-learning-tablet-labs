"""Turn a canvas drawing or a scanned/photographed signature into a clean ink map."""
import numpy as np
from PIL import Image, ImageFilter

OUT_H, OUT_W = 128, 256  # signatures are wide, so keep a 1:2 frame


def _otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    w0 = cum / total
    w1 = 1.0 - w0
    mu0 = cum_mean / np.maximum(cum, 1)
    mu1 = (cum_mean[-1] - cum_mean) / np.maximum(total - cum, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return int(np.argmax(between))


def ink_map(image_data):
    """Return a uint8 array where ink is bright (255) and background is 0."""
    data = np.asarray(image_data)
    if data.ndim == 3 and data.shape[2] == 4:
        # Canvas output: dark strokes; untouched pixels may be transparent.
        gray = data[:, :, :3].astype(np.float32).mean(axis=2)
        alpha = data[:, :, 3].astype(np.float32) / 255.0
        ink = (255.0 - gray) * alpha
        return np.clip(ink, 0, 255).astype(np.uint8)
    if data.ndim == 3:
        gray = data[:, :, :3].astype(np.float32).mean(axis=2)
    elif data.ndim == 2:
        gray = data.astype(np.float32)
    else:
        raise ValueError('Expected a grayscale, RGB or RGBA image.')
    if gray.max() <= 1.0:
        gray *= 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    # Paper is usually lighter than ink; invert so ink becomes bright.
    if np.median(gray) > 127:
        gray = 255 - gray
    # Remove paper texture and uneven lighting with an Otsu threshold.
    t = _otsu(gray)
    return np.where(gray > t, gray, 0).astype(np.uint8)


def prepare_signature(image_data):
    """Crop, scale and center a signature into a fixed OUT_H x OUT_W frame.

    Returns (normalized float array in [0, 1], PIL preview) or (None, None) if empty.
    """
    ink = ink_map(image_data)
    ys, xs = np.where(ink > 30)
    if len(xs) < 20:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cropped = Image.fromarray(ink[y0:y1, x0:x1], mode='L')
    w, h = cropped.size
    margin = 8
    scale = min((OUT_W - 2 * margin) / w, (OUT_H - 2 * margin) / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cropped.resize((rw, rh), Image.Resampling.LANCZOS)
    frame = Image.new('L', (OUT_W, OUT_H), 0)
    frame.paste(resized, ((OUT_W - rw) // 2, (OUT_H - rh) // 2))
    # A light blur makes features tolerant to small stroke-width changes.
    frame = frame.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(frame, dtype=np.float32) / 255.0
    return arr, frame
