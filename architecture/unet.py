import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential


class TimestepEmbedding(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = layers.Dense(channels)

    def call(self, timestep):
        # Assuming timestep is a scalar or batch of scalars
        x = tf.expand_dims(timestep, -1)  # Shape: (batch_size, 1)
        x = self.dense(x)  # Shape: (batch_size, channels)
        return x


def get_conv_block(
    filters: int,
    kernel_size: tuple,
    padding: str = "same",
    strides: tuple = (1, 1),
    kernel_initializer: str = "he_normal",
    final: bool = False,
):
    layer_array = [
        layers.Conv2D(
            filters,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_initializer=kernel_initializer,
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(
            filters,
            kernel_size,
            padding=padding,
            kernel_initializer=kernel_initializer,
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(0.2),
    ]

    if final:
        layer_array.append(
            layers.Conv2D(
                filters,
                kernel_size,
                padding=padding,
                kernel_initializer=kernel_initializer,
                activation="sigmoid",
            )
        )
    return Sequential(layer_array)


class UNetBlock(layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple = (3, 3),
        padding: str = "same",
        kernel_initializer: str = "he_normal",
        **kwargs,
    ):
        super(UNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.conv_block = get_conv_block(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            final=False,
        )
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))

    def call(
        self,
        inputs: tf.Tensor,
        training=False,
        mask=None,
        timestep_embedding: tf.Tensor = None,
    ):
        x = self.conv_block(inputs)

        if timestep_embedding is not None:
            timestep_embedding = tf.reshape(
                timestep_embedding, [1, 1, 1, -1]
            )  # Adjust the shape accordingly
            timestep_embedding = tf.broadcast_to(timestep_embedding, tf.shape(x))
            x += timestep_embedding  # Adding the timestep embedding to the feature maps

        p = self.pool(x)
        return x, p

    def get_config(self):
        base_config = super(UNetBlock, self).get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
        }


@keras.utils.register_keras_serializable(package="MyLayers")
class UNet(Model):
    def __init__(
        self, output_channels: int, depth: int, initial_filters: int, **kwargs
    ):
        super(UNet, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.depth = depth
        self.initial_filters = initial_filters

        self.first_block: UNetBlock = UNetBlock(initial_filters)

        self.down_blocks = []
        self.up_blocks = []
        filters = initial_filters

        for i in range(depth):
            block = UNetBlock(filters)
            self.down_blocks.append(block)
            filters *= 2

        for i in range(depth - 1):
            block = UNetBlock(filters // 2)
            self.up_blocks.append(block)
            filters //= 2

        self.final_block = get_conv_block(
            filters=output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            final=True,
        )

    def call(self, inputs: tf.Tensor, training=False, mask=None, timestep: int = -1):
        x = inputs
        if timestep == -1:
            timestep_embedding = None
        else:
            timestep_embedding = TimestepEmbedding(self.initial_filters)(timestep)

        x, p = self.first_block(
            inputs=x, training=training, timestep_embedding=timestep_embedding
        )

        skips: list = []
        for down_block in self.down_blocks:
            x, p = down_block(x)
            skips.append(x)
            x = p

        skips = skips[::-1]
        for skip, up_block in zip(skips, self.up_blocks):
            x = layers.UpSampling2D(size=(2, 2))(x)
            x = layers.concatenate([x, skip])
            x = up_block.conv_block(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.concatenate([x, skips[-1]])
        x = self.final_block(x)
        return x

    def get_config(self):
        base_config = super(UNet, self).get_config()
        return {
            **base_config,
            "output_channels": self.output_channels,
            "depth": self.depth,
            "initial_filters": self.initial_filters,
        }

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**config)
