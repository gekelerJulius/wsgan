import math

import tensorflow as tf
from keras import layers

from resnet_block import ResNetBlock


def create_generator(noise_dim: int, image_side_length: int, channels: int):
    noise_input = tf.keras.Input(shape=(noise_dim,))
    image_side_length_root = int(math.sqrt(image_side_length))
    x = layers.Dense(
        units=image_side_length_root * image_side_length_root * 16, use_bias=False
    )(noise_input)
    x = layers.Reshape((image_side_length_root, image_side_length_root, 16))(x)

    # Upsample the image using ResNet blocks and UpSampling2D layers
    filters_start = 4
    while x.shape[1] < image_side_length // 2:
        filters_start *= 2
        x = ResNetBlock(filters_start, 3)(x)
        x = layers.Conv2DTranspose(
            filters=filters_start // 2,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    # Final Up-Sampling to target image dimensions
    x = ResNetBlock(filters_start, 3)(x)
    x = tf.image.resize(
        x,
        [image_side_length, image_side_length],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    while x.shape[3] > channels * 8:
        x = ResNetBlock(x.shape[3] // 2, 3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        filters=channels, kernel_size=3, padding="same", activation="tanh"
    )(x)
    return tf.keras.Model(inputs=[noise_input], outputs=x)
