from typing import Tuple

import tensorflow as tf
from keras import layers


def create_lightweight_discriminator(image_shape: Tuple[int, int, int]):
    image_input = tf.keras.Input(shape=image_shape)
    start_filters = 32

    x = image_input
    for i in range(3):
        x = layers.Conv2D(start_filters, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        start_filters *= 2

    # Flatten and dense layer
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return tf.keras.Model(inputs=[image_input], outputs=x)


if __name__ == "__main__":
    image_side_length = 224
    channels = 3
    image_shape = (image_side_length, image_side_length, channels)
    discriminator = create_lightweight_discriminator(image_shape)
    discriminator.summary()
    test_image = tf.random.normal((1, *image_shape))
    test_output = discriminator(test_image)
    assert test_output.shape == (1, 1) and test_output.dtype == tf.float32
    print("Test passed!")
