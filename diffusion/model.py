import tensorflow as tf
from keras import Model


class DiffusionCustomModel(Model):
    def __init__(self, model: tf.keras.layers.Layer):
        super(DiffusionCustomModel, self).__init__()
        self.model = model

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
    ):
        super(DiffusionCustomModel, self).compile(optimizer, loss)
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data: tf.Tensor):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    def call(self, inputs: tf.Tensor, training: bool = False, mask: tf.Tensor = None):
        noisy_inputs = self.add_noise(inputs)
        predicted_noise = self.model(noisy_inputs, training=training)
        denoised_inputs = self.remove_noise(inputs, predicted_noise)
        return denoised_inputs

    @staticmethod
    def add_noise(inputs, std=0.1):
        noise = tf.random.normal(shape=tf.shape(inputs), stddev=std)
        return inputs + noise

    @staticmethod
    def remove_noise(inputs, predicted_noise):
        return inputs - predicted_noise

    def get_config(self):
        config = super(DiffusionCustomModel, self).get_config()
        config.update(
            {
                "unet": self.model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        unet = config.pop("unet")
        return cls(unet=unet)
