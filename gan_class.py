from enum import Enum

import tensorflow as tf
from keras.src.engine.data_adapter import unpack_x_y_sample_weight
from tensorflow.keras import Model

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class ImplementedLosses(Enum):
    STANDARD = "standard"
    WASSERSTEIN = "wasserstein"


class GAN(Model):
    def __init__(self, discriminator, generator, noise_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim

    def compile(self, d_optimizer, g_optimizer, loss_type=ImplementedLosses.STANDARD):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_type = loss_type

    def train_step(self, data):
        real_images, y, _ = unpack_x_y_sample_weight(data)

        # Generate noise
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Critic updates (common practice is to do this more frequently than generator updates)
        num_critic_updates = 5 if self.loss_type == ImplementedLosses.WASSERSTEIN else 1
        for _ in range(num_critic_updates):
            with tf.GradientTape() as disc_tape:
                fake_images = self.generator(random_noise, training=True)
                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(fake_images, training=True)
                d_loss = self.calculate_discriminator_loss(real_output, fake_output)

            grads_of_discriminator = disc_tape.gradient(
                d_loss, self.discriminator.trainable_variables
            )
            self.d_optimizer.apply_gradients(
                zip(grads_of_discriminator, self.discriminator.trainable_variables)
            )
            if self.loss_type == ImplementedLosses.WASSERSTEIN:
                self.clip_discriminator_weights()

        # Generator update
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(random_noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            g_loss = self.calculate_generator_loss(fake_output)

        grads_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )
        self.g_optimizer.apply_gradients(
            zip(grads_of_generator, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}

    def calculate_generator_loss(self, fake_output):
        if self.loss_type == ImplementedLosses.STANDARD:
            return self.standard_generator_loss(fake_output)
        elif self.loss_type == ImplementedLosses.WASSERSTEIN:
            return self.ws_generator_loss(fake_output)

    def calculate_discriminator_loss(self, real_output, fake_output):
        if self.loss_type == ImplementedLosses.STANDARD:
            return self.standard_discriminator_loss(real_output, fake_output)
        elif self.loss_type == ImplementedLosses.WASSERSTEIN:
            return self.ws_critic_loss(real_output, fake_output)

    def standard_discriminator_loss(self, real_output, fake_output):
        real_loss = bce_loss(tf.ones_like(real_output), real_output)
        fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output)
        total_disc_loss = (real_loss + fake_loss) / 2
        return total_disc_loss

    def standard_generator_loss(self, fake_output):
        return bce_loss(tf.ones_like(fake_output), fake_output)

    def ws_critic_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def ws_generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def clip_discriminator_weights(self, clip_value=0.01):
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -clip_value, clip_value))

    def get_config(self):
        config = super(GAN, self).get_config()
        config.update(
            {
                "discriminator": self.discriminator,
                "generator": self.generator,
                "noise_dim": self.noise_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        discriminator = config.pop("discriminator")
        generator = config.pop("generator")
        noise_dim = config.pop("noise_dim")
        return cls(discriminator, generator, noise_dim)
