import math

import tensorflow as tf
from keras import layers


def create_lightweight_generator(
    noise_dim: int, image_side_length: int, channels: int
) -> tf.keras.Model:
    """Create a lightweight generator model for DCGAN.
    @param noise_dim: The dimension of the noise input
    @param image_side_length: The side length of the image
    @param channels: The number of channels in the image
    @return: The generator model
    """
    noise_input = tf.keras.Input(shape=(noise_dim,))
    start_side_length = image_side_length // 8
    start_channels = 16
    x = layers.Dense(start_side_length * start_side_length * start_channels)(
        noise_input
    )
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((start_side_length, start_side_length, start_channels))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=12, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    # Upsample the image to exact image_side_length
    x = layers.Resizing(image_side_length, image_side_length, interpolation="bilinear")(
        x
    )

    # Final convolutional layer to adjust to the correct number of output channels
    x = layers.Conv2D(
        filters=channels, kernel_size=3, strides=1, padding="same", activation="tanh"
    )(x)
    return tf.keras.Model(inputs=noise_input, outputs=x)


if __name__ == "__main__":
    noise_dim = 200
    image_side_length = 224
    channels = 3
    generator = create_lightweight_generator(noise_dim, image_side_length, channels)
    generator.summary()

    test_input = tf.random.normal((1, noise_dim))
    test_output = generator(test_input)
    assert test_output.shape == (1, image_side_length, image_side_length, channels)
    print("Generator test passed!")
