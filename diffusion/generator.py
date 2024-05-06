import tensorflow as tf
import config


def generate_images(model: tf.keras.Model, num_images: int) -> tf.Tensor:
    noise = tf.random.normal(
        (num_images, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.CHANNELS)
    )
    return model.predict(noise)
