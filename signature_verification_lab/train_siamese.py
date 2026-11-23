"""Train a Siamese CNN on CEDAR and export model/signature_embedding.keras.

Expected layout (CEDAR, extracted from signatures.rar):
    DATA_DIR/full_org/original_<writer>_<n>.png    24 genuine signatures per writer
    DATA_DIR/full_forg/forgeries_<writer>_<n>.png  24 skilled forgeries per writer

Usage:
    python train_siamese.py --data-dir path/to/signatures
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from preprocessing import prepare_signature

SEED = 42
TEST_WRITERS = 10  # writer-independent split: these people are never seen in training

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--pairs-per-epoch', type=int, default=6000)
args = parser.parse_args()
tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)


def load_folder(folder, prefix):
    images = {}
    for path in sorted(Path(folder).glob(f'{prefix}_*_*.png')):
        writer = int(re.match(rf'{prefix}_(\d+)_', path.name).group(1))
        arr, _ = prepare_signature(np.asarray(Image.open(path).convert('L')))
        if arr is not None:
            images.setdefault(writer, []).append((arr * 255).astype(np.uint8))
    return {w: np.stack(v) for w, v in images.items()}


data_dir = Path(args.data_dir)
print('Loading and preprocessing images...')
genuine = load_folder(data_dir / 'full_org', 'original')
forged = load_folder(data_dir / 'full_forg', 'forgeries')
writers = sorted(set(genuine) & set(forged))
if len(writers) <= TEST_WRITERS:
    raise SystemExit(f'Found only {len(writers)} writers in {data_dir}; check the folder layout.')
rng.shuffle(writers)
test_writers, train_writers = writers[:TEST_WRITERS], writers[TEST_WRITERS:]
print(f'{len(writers)} writers · train {len(train_writers)} · test {len(test_writers)}')


def make_pairs(writer_ids, n):
    """1/2 genuine-genuine (label 1), 1/4 skilled forgery, 1/4 another writer (label 0)."""
    a, b, y = [], [], []
    for i in range(n):
        w = writer_ids[rng.integers(len(writer_ids))]
        g = genuine[w]
        a.append(g[rng.integers(len(g))])
        kind = i % 4
        if kind < 2:
            b.append(g[rng.integers(len(g))]); y.append(1.0)
        elif kind == 2:
            f = forged[w]; b.append(f[rng.integers(len(f))]); y.append(0.0)
        else:
            other = writer_ids[rng.integers(len(writer_ids))]
            while other == w:
                other = writer_ids[rng.integers(len(writer_ids))]
            o = genuine[other]; b.append(o[rng.integers(len(o))]); y.append(0.0)
    to_x = lambda x: (np.stack(x).astype(np.float32) / 255.0)[..., np.newaxis]
    return to_x(a), to_x(b), np.array(y, dtype=np.float32)


def build_embedding():
    inp = tf.keras.layers.Input(shape=(128, 256, 1))
    x = inp
    for filters in (32, 64, 128, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128)(x)
    out = tf.keras.layers.UnitNormalization()(x)
    return tf.keras.Model(inp, out, name='signature_embedding')


class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, a, b):
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True) + 1e-9)


def contrastive_loss(y, d, margin=1.0):
    # Pull genuine pairs together, push different/forged pairs beyond the margin.
    return tf.reduce_mean(y * tf.square(d) + (1 - y) * tf.square(tf.maximum(margin - d, 0.0)))


embedding = build_embedding()
embedding.summary()
in_a = tf.keras.layers.Input(shape=(128, 256, 1))
in_b = tf.keras.layers.Input(shape=(128, 256, 1))
distance = EuclideanDistance()(embedding(in_a), embedding(in_b))
siamese = tf.keras.Model([in_a, in_b], distance)
siamese.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=contrastive_loss)

val_a, val_b, val_y = make_pairs(test_writers, 2000)
for epoch in range(args.epochs):
    tr_a, tr_b, tr_y = make_pairs(train_writers, args.pairs_per_epoch)  # fresh pairs each epoch
    h = siamese.fit([tr_a, tr_b], tr_y, validation_data=([val_a, val_b], val_y),
                    batch_size=32, epochs=1, verbose=0)
    print(f"epoch {epoch + 1}/{args.epochs} · loss {h.history['loss'][0]:.4f} · val_loss {h.history['val_loss'][0]:.4f}")

# Evaluate on unseen writers: best global threshold, accuracy, equal error rate (EER).
d = siamese.predict([val_a, val_b], verbose=0).ravel()
thresholds = np.linspace(d.min(), d.max(), 400)
far = np.array([np.mean(d[val_y == 0] <= t) for t in thresholds])  # forgeries accepted
frr = np.array([np.mean(d[val_y == 1] > t) for t in thresholds])   # genuine rejected
best = int(np.argmin(np.abs(far - frr)))
threshold = float(thresholds[best])
accuracy = float(np.mean((d <= threshold) == (val_y == 1)))
metrics = {'test_writers': len(test_writers), 'test_pairs': int(len(val_y)),
           'threshold': threshold, 'accuracy': accuracy,
           'eer': float((far[best] + frr[best]) / 2),
           'false_accept_rate': float(far[best]), 'false_reject_rate': float(frr[best])}
Path('model').mkdir(exist_ok=True)
embedding.save('model/signature_embedding.keras')
Path('model/siamese_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
print('Model saved: model/signature_embedding.keras')
