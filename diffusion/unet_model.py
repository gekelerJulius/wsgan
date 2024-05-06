import math
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from architecture.unet import UNet
from diffusion import config


class UNetModel(keras.Model):
    def __init__(self, image_side_length: int, channels: int, num_timesteps: int):
        super().__init__()
        self.image_side_length = image_side_length
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.unet = UNet(
            output_channels=channels,
            depth=config.UNET_DEPTH,
            initial_filters=32,
        )

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
    ):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        mask: tf.Tensor = None,
        timestep: int = 0,
    ) -> tf.Tensor:
        x = self.unet(inputs)
        return (x * 2) - 1  # Move output to [-1, 1] range

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        x, _ = data  # We ignore the labels since they are not used
        x: tf.Tensor = x

        with tf.GradientTape() as tape:
            loss = 0.0
            for t in range(self.num_timesteps, 0, -1):
                noise = tf.random.normal(shape=tf.shape(x))
                noisy_image = x + noise * (
                    t / self.num_timesteps
                )  # Incremental noise addition
                pred_image = self(noisy_image, training=True)
                loss += self.loss_fn(x, pred_image)

        # Calculate gradients and update model weights
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dict mapping metrics names to their current value
        return {"loss": loss}


if __name__ == "__main__":
    img_side_length = 16
    channels = 1
    timesteps = 200
    model = UNetModel(
        image_side_length=img_side_length,
        channels=channels,
        num_timesteps=timesteps,
    )
    model.build(input_shape=(None, img_side_length, img_side_length, channels))
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)
    model.summary()
    print("Model compiled!")
    example_input = tf.random.normal([8, img_side_length, img_side_length, channels])
    # Example input should be between -1 and 1
    example_input = tf.clip_by_value(example_input, -1.0, 1.0)
    print(f"Example input shape: {example_input.shape}")
    print(f"Example input min: {example_input.numpy().min()}")
    print(f"Example input max: {example_input.numpy().max()}")
    print(f"Example input mean: {example_input.numpy().mean()}")
    print(f"Example input std: {example_input.numpy().std()}")

    example_output = model(example_input)
    # Output is between 0 and 1, move to -1 and 1
    example_output = (example_output * 2) - 1
    print(f"Example output shape: {example_output.shape}")
    print(f"Example output min: {example_output.numpy().min()}")
    print(f"Example output max: {example_output.numpy().max()}")
    print(f"Example output mean: {example_output.numpy().mean()}")
    print(f"Example output std: {example_output.numpy().std()}")
