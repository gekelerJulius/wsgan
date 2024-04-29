from typing import Tuple

import tensorflow as tf
from keras import layers
from resnet_block import ResNetBlock


def create_discriminator(image_shape: Tuple[int, int, int]):
    image_input = tf.keras.Input(shape=image_shape)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(image_input)

    while x.shape[1] > 4:
        x = ResNetBlock(x.shape[3], 3)(x)
        x = layers.AveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    # Layers that do not change dimensions
    for _ in range(3):
        x = ResNetBlock(x.shape[3], 3, stride=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(min(512, image_shape[0] * image_shape[1]), activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1)(x)
    return tf.keras.Model(inputs=[image_input], outputs=x)


if __name__ == "__main__":
    image_shape = (32, 32, 3)
    discriminator = create_discriminator(image_shape)
    discriminator.summary()
    test_image = tf.random.normal((1, *image_shape))
    test_output = discriminator(test_image)
    print(test_output)
