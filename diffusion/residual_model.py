import tensorflow as tf
from tensorflow import keras

from architecture.resnet_block import ResNetBlock


class ResidualModel(keras.layers.Layer):
    def __init__(self, image_side_length: int, channels: int):
        super(ResidualModel, self).__init__()

        self.resnets = keras.Sequential(
            [
                ResNetBlock(
                    filters=64,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                    kernel_initializer="he_normal",
                )
                for _ in range(1)
            ]
        )

        self.final_resnet = ResNetBlock(
            filters=channels,
            kernel_size=3,
            stride=1,
            padding="same",
            kernel_initializer="he_normal",
        )

        self.activation = keras.layers.Activation("tanh")

        self.image_side_length = image_side_length

    def call(self, inputs: tf.Tensor, training: bool = None, mask: tf.Tensor = None):
        x = self.resnets(inputs)
        x = self.final_resnet(x)
        x = self.activation(x)
        return x

    def get_config(self):
        base_config = super(ResidualModel, self).get_config()
        return {
            **base_config,
            "image_side_length": self.image_side_length,
            "channels": self.channels,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    img_side_length = 32
    channels = 1
    model = ResidualModel(img_side_length, channels)
    test_input = tf.random.normal((1, img_side_length, img_side_length, channels))
    test_output = model(test_input)
    assert test_output.shape == (1, img_side_length, img_side_length, channels)
