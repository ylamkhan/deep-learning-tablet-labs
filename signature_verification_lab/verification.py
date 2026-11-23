"""Feature extraction, per-person threshold calibration and verification."""
from dataclasses import dataclass
import numpy as np
from skimage.feature import hog

MIN_REFERENCES = 3


def extract_features(arr):
    """Concatenate equally weighted, unit-norm feature groups."""
    # Stroke orientation patterns (shape of the letters).
    h = hog(arr, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            feature_vector=True)
    # Where the ink sits in a coarse 4x8 grid (overall layout).
    grid = arr.reshape(4, arr.shape[0] // 4, 8, arr.shape[1] // 8).mean(axis=(1, 3)).ravel()
    # Horizontal and vertical projection profiles (rhythm of the strokes).
    proj_x = arr.mean(axis=0).reshape(32, -1).mean(axis=1)
    proj_y = arr.mean(axis=1).reshape(16, -1).mean(axis=1)
    parts = [h, grid, np.concatenate([proj_x, proj_y])]
    return np.concatenate([p / (np.linalg.norm(p) + 1e-8) for p in parts])


def _distance(f, refs):
    """Mean Euclidean distance from one feature vector to a set of references."""
    return float(np.linalg.norm(refs - f, axis=1).mean())


def calibrate(ref_features):
    """Leave-one-out: how far is each genuine reference from the others?"""
    refs = np.asarray(ref_features)
    loo = np.array([_distance(refs[i], np.delete(refs, i, axis=0)) for i in range(len(refs))])
    # Accept up to the worst genuine case plus a margin for natural variation.
    threshold = max(loo.max(), loo.mean() + 2 * loo.std()) * 1.15
    return threshold, loo


@dataclass
class Result:
    genuine: bool
    distance: float
    threshold: float
    score: float  # 100 = identical to references, 50 = exactly on the threshold


def verify(query_features, ref_features):
    if len(ref_features) < MIN_REFERENCES:
        raise ValueError(f'At least {MIN_REFERENCES} reference signatures are required.')
    threshold, _ = calibrate(ref_features)
    d = _distance(query_features, np.asarray(ref_features))
    score = 100.0 * 2.0 ** (-d / threshold)
    return Result(genuine=d <= threshold, distance=d, threshold=threshold, score=score)
