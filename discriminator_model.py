import tensorflow as tf
from keras import layers
from typing import Tuple

from resnet_block import ResNetBlock


def create_discriminator(image_shape: Tuple[int, int, int]):
    image_input = tf.keras.Input(shape=image_shape)

    # Discriminator architecture
    x = layers.Conv2D(64, 3, strides=2, padding="same")(image_input)

    while x.shape[1] > 4:
        x = ResNetBlock(x.shape[3], 3)(x)
        x = layers.AveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(min(512, image_shape[0] * image_shape[1]), activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1)(x)
    return tf.keras.Model(inputs=[image_input], outputs=x)
