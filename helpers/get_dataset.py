from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers


def preprocess(image: tf.Tensor, label: tf.Tensor, target_size: Tuple[int, int]):
    image = tf.image.resize(tf.cast(image, tf.float32) / 127.5 - 1, target_size)
    return image, label


# Create augmentation layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ]
)


def augment(image: tf.Tensor, label: tf.Tensor):
    image = data_augmentation(image)
    return image, label


def get_dataset(
    channels: int,
    batch_size: int,
    target_size: Tuple[int, int],
    data_images_dir: str = None,
):
    color_mode = "grayscale"
    if channels == 3:
        color_mode = "rgb"

    if data_images_dir is not None:
        train_ds = keras.utils.image_dataset_from_directory(
            data_images_dir,
            label_mode=None,
            image_size=target_size,
            batch_size=None,
            color_mode=color_mode,
            shuffle=True,
        ).map(lambda x: (x, 0), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Use the Fashion MNIST dataset (only use one class)
        used_class = 5
        (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
        train_images = train_images[train_labels == used_class]
        train_labels = train_labels[train_labels == used_class]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        # Set the labels to 0
        train_ds = train_ds.map(
            lambda x, y: (x, 0), num_parallel_calls=tf.data.AUTOTUNE
        )
        # Unsqueezing the channel dimension so images are (28, 28, 1) instead of (28, 28)
        train_ds = train_ds.map(
            lambda x, y: (tf.expand_dims(x, -1), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Normalize to [-1, 1], shuffle and batch
    train_ds = (
        train_ds.map(augment)
        .map(
            lambda x, y: preprocess(x, y, target_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(batch_size * 6)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Show 9 samples
    # plt.figure(figsize=(10, 10))
    # for images, _ in train_ds.take(1):
    #     for i in range(9):
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow((images[i].numpy() + 1) / 2)
    #         plt.axis("off")
    # plt.show()

    # Output statistics
    for images, _ in train_ds.take(1):
        mean = tf.reduce_mean(images)
        std = tf.math.reduce_std(images)
        max_val = tf.reduce_max(images)
        min_val = tf.reduce_min(images)

        print(f"Mean: {mean}")
        print(f"Std: {std}")
        print(f"Max: {max_val}")
        print(f"Min: {min_val}")

    return train_ds
