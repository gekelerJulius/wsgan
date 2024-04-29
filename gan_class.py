import tensorflow as tf
from keras import Model


class ConditionalGAN(Model):
    def __init__(self, discriminator: Model, generator: Model, noise_dim: int):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim

    def compile(
        self,
        d_optimizer: tf.keras.optimizers.Optimizer,
        g_optimizer: tf.keras.optimizers.Optimizer,
    ):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        real_images, _ = data

        # Generate noise
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Critic updates
        num_critic_updates = 5
        for _ in range(num_critic_updates):
            with tf.GradientTape() as disc_tape:
                fake_images = self.generator(random_noise, training=True)
                real_output = self.discriminator(
                    real_images,
                    training=True,
                )
                fake_output = self.discriminator(fake_images, training=True)
                d_loss = self.calculate_discriminator_loss(
                    real_images,
                    real_output,
                    fake_images,
                    fake_output,
                    batch_size,
                )

            grads_of_discriminator = disc_tape.gradient(
                d_loss, self.discriminator.trainable_variables
            )
            self.d_optimizer.apply_gradients(
                zip(grads_of_discriminator, self.discriminator.trainable_variables)
            )

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

    def calculate_discriminator_loss(
        self,
        real_images: tf.Tensor,
        real_output: tf.Tensor,
        fake_images: tf.Tensor,
        fake_output: tf.Tensor,
        batch_size: int,
    ):
        real_output = tf.cast(real_output, tf.float32)
        fake_output = tf.cast(fake_output, tf.float32)
        real_images = tf.cast(real_images, tf.float32)
        fake_images = tf.cast(fake_images, tf.float32)
        lambda_gp = 10
        w_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gp = self.gradient_penalty(batch_size, real_images, fake_images)
        return w_loss + lambda_gp * gp

    import tensorflow as tf

    def gradient_penalty(
        self, batch_size: int, real_images: tf.Tensor, fake_images: tf.Tensor
    ):
        """Calculates the gradient penalty using float32 for stability.
        This method interpolates between real and fake images and computes the gradient
        penalty to encourage 1-Lipschitz continuity of the discriminator.
        """
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)

        interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            predictions = self.discriminator(interpolated_images, training=True)
            predictions = tf.cast(predictions, tf.float32)

        # Calculate the gradients relative to the interpolated images
        gradients = tape.gradient(predictions, [interpolated_images])[0]
        gradients = tf.cast(gradients, tf.float32)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gradient_penalty

    def calculate_generator_loss(self, fake_output: tf.Tensor):
        fake_output = tf.cast(fake_output, tf.float32)
        return -tf.reduce_mean(fake_output)

    def get_config(self):
        config = super(ConditionalGAN, self).get_config()
        config.update(
            {
                "discriminator": self.discriminator,
                "generator": self.generator,
                "noise_dim": self.noise_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        discriminator = config.pop("discriminator")
        generator = config.pop("generator")
        noise_dim = config.pop("noise_dim")
        return cls(discriminator, generator, noise_dim)
