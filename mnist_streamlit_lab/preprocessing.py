"""Convert a white-on-black canvas into an MNIST-like (28, 28, 1) tensor."""
import numpy as np
from PIL import Image


def prepare_digit(canvas_data):
    data = np.asarray(canvas_data)
    if data.ndim == 3:
        rgb = data[:, :, :3].astype(np.float32)
        gray = rgb.mean(axis=2)
        if data.shape[2] == 4:
            alpha = data[:, :, 3].astype(np.float32) / 255.0
            gray *= alpha
    elif data.ndim == 2:
        gray = data.astype(np.float32)
    else:
        raise ValueError('Expected a grayscale, RGB or RGBA image.')
    if gray.max() <= 1.0:
        gray *= 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    cropped = Image.fromarray(gray[y0:y1, x0:x1], mode='L')
    w, h = cropped.size
    scale = 20 / max(w, h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cropped.resize((rw, rh), Image.Resampling.LANCZOS)
    result = Image.new('L', (28, 28), 0)
    result.paste(resized, ((28-rw)//2, (28-rh)//2))
    arr = np.asarray(result, dtype=np.float32)
    mass = arr.sum()
    if mass > 0:
        yy, xx = np.indices(arr.shape)
        cx = float((arr * xx).sum() / mass)
        cy = float((arr * yy).sum() / mass)
        # Shift without wrapping pixels, close to MNIST center-of-mass alignment.
        dx, dy = round(13.5-cx), round(13.5-cy)
        shifted = np.zeros_like(arr)
        sx0, sx1 = max(0, -dx), min(28, 28-dx)
        sy0, sy1 = max(0, -dy), min(28, 28-dy)
        tx0, ty0 = max(0, dx), max(0, dy)
        shifted[ty0:ty0+sy1-sy0, tx0:tx0+sx1-sx0] = arr[sy0:sy1, sx0:sx1]
        arr = shifted
    display = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='L')
    tensor = (arr / 255.0).reshape(1, 28, 28, 1)
    return tensor, display
