"""Train a compact CNN on MNIST and export model/mnist_cnn.keras."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') / 255.0)[..., np.newaxis]
x_test = (x_test.astype('float32') / 255.0)[..., np.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_split=0.1, batch_size=128, epochs=5, verbose=2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
Path('model').mkdir(exist_ok=True)
model.save('model/mnist_cnn.keras')
Path('model/metrics.json').write_text(json.dumps({'test_accuracy': float(accuracy), 'test_loss': float(loss)}, indent=2), encoding='utf-8')
print(f'Test accuracy: {accuracy:.4%}')
print('Model saved: model/mnist_cnn.keras')
