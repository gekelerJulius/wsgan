import tensorflow as tf
from tensorflow import keras

keras.utils.get_custom_objects().clear()


# This block takes filters and kernel_size as arguments
# filters: number of filters for the Conv2D layers
# kernel_size: size of the kernel for the Conv2D layers
# The block consists of two Conv2D layers with BatchNormalization and LeakyReLU
# The block is added to the input tensor
# The output is the sum of the input tensor and the block
# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to `Custom` and the name defaults to
# the class name.
@keras.utils.register_keras_serializable(package="MyLayers")
class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters, kernel_size, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters, kernel_size, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
            ]
        )

        self.matching_conv = tf.keras.layers.Conv2D(filters, 1, padding="same")

    def call(self, inputs):
        x = self.block(inputs)
        if inputs.shape[-1] != self.filters:
            inputs = self.matching_conv(
                inputs
            )  # Only apply if input and output filters differ
        return inputs + x

    def get_config(self):
        base_config = super(ResNetBlock, self).get_config()
        base_config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            }
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        filters = config["filters"]
        kernel_size = config["kernel_size"]
        return cls(filters, kernel_size)


if __name__ == "__main__":
    # Typical usage of the ResNetBlock for 32x32x3 image
    block = ResNetBlock(64, 3)
    inputs = tf.keras.Input(shape=(32, 32, 3))
    outputs = block(inputs)
    print(outputs.shape)
