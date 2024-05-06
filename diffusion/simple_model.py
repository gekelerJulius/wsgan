from typing import Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SimpleModel(keras.Model):
    def __init__(self, image_side_length: int, channels: int, num_timesteps: int):
        super().__init__()
        self.image_side_length = image_side_length
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.conv1 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv2 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.conv_final = layers.Conv2D(
            channels, (3, 3), padding="same", activation="sigmoid"
        )

    def compile(
        self, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss
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
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv_final(x)
        return x

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
