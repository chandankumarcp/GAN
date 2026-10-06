"""Data loading and preprocessing."""
from tensorflow.keras.datasets import cifar10


def load_and_prepare_data():
    # Load CIFAR-10
    (train_X, _), (_, _) = cifar10.load_data()

    # Convert to float32
    X = train_X.astype('float32')

    # Scale from [0, 255] to [-1, 1]
    X = (X - 127.5) / 127.5
    return X
