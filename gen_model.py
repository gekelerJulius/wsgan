import math

import tensorflow as tf
from keras import layers
from resnet_block import ResNetBlock


def create_generator(
    noise_dim: int, image_side_length: int, channels: int
) -> tf.keras.Model:
    """Create a generator model for DCGAN.
    @param noise_dim: The dimension of the noise input
    @param image_side_length: The side length of the image
    @param channels: The number of channels in the image
    @return: The generator model
    """
    # Noise input of shape (noise_dim,)
    noise_input = tf.keras.Input(shape=(noise_dim,))

    image_side_length_root = int(
        math.sqrt(image_side_length * image_side_length * channels / 16)
    )
    x = layers.Dense(
        units=image_side_length_root * image_side_length_root * 16, use_bias=False
    )(noise_input)
    x = layers.Reshape((image_side_length_root, image_side_length_root, 16))(x)

    # Upsample the image using ResNet blocks and UpSampling2D layers
    filters_start = 16
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

    # Layers that do not change dimensions
    for _ in range(3):
        x = ResNetBlock(filters=filters_start, kernel_size=3, stride=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    while x.shape[3] > channels * 8:
        x = ResNetBlock(x.shape[3] // 2, 3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        filters=channels, kernel_size=3, padding="same", activation="tanh"
    )(x)
    return tf.keras.Model(inputs=noise_input, outputs=x)


if __name__ == "__main__":
    generator = create_generator(200, 32, 3)
    generator.summary()

    test_input = tf.random.normal((1, 200))
    test_output = generator(test_input)
    print(test_output.shape)
