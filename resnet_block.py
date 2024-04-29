import tensorflow as tf
from tensorflow import keras

keras.utils.get_custom_objects().clear()


@keras.utils.register_keras_serializable(package="MyLayers")
class ResNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        padding="same",
        kernel_initializer="he_normal",
    ):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_initializer = kernel_initializer

        self.block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=stride,
                    padding=padding,
                    kernel_initializer=kernel_initializer,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    padding=padding,
                    kernel_initializer=kernel_initializer,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.1),
            ]
        )

        self.matching_conv = tf.keras.layers.Conv2D(
            filters, 1, padding="same", kernel_initializer=kernel_initializer
        )

    def call(self, inputs):
        x = self.block(inputs)
        if inputs.shape[-1] != self.filters:
            inputs = self.matching_conv(inputs)
        return inputs + x

    def get_config(self):
        base_config = super(ResNetBlock, self).get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
