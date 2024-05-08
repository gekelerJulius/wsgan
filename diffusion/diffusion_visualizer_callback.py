import os

import tensorflow as tf
from keras import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from diffusion import config


class DiffusionVisualizerCallback(Callback):
    def __init__(
        self,
        plot_interval: int,
        model: Model,
        dataset: tf.data.Dataset,
        num_timesteps: int,
    ):
        super().__init__()
        self.plot_interval = plot_interval
        self.model = model
        self.dataset = dataset
        self.num_timesteps = num_timesteps

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_interval == 0:
            self.plot_diffusion(epoch)

    def plot_diffusion(self, epoch):
        os.makedirs(f"{config.PLOTS_DIR}/{epoch + 1}_epoch", exist_ok=True)
        plot_save_dir = f"{config.PLOTS_DIR}/{epoch + 1}_epoch"
        # Gradually apply noise to the first image in the dataset
        first_image = next(iter(self.dataset))[0][0]
        noise = tf.random.normal(shape=tf.shape(first_image), mean=0.0, stddev=1.0)
        # Create a list of images with increasing noise
        images = []
        for i in range(self.num_timesteps):
            if i % 10 == 0 or i == self.num_timesteps - 1:
                noisy_image = first_image + noise * tf.sqrt(1 - i / self.num_timesteps)
                noisy_image = tf.clip_by_value(noisy_image, -1.0, 1.0)
                images.append(noisy_image)

        # Generate noise predictions for each image
        predicted_noises = []
        for image in images:
            predicted_noises.append(self.model.predict(image[tf.newaxis, ...]))

        for i, (image, noise_prediction) in enumerate(zip(images, predicted_noises)):
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(image.numpy().reshape(config.IMAGE_SIZE))
            plt.title("Image with noise")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            noise_subtracted = image - noise_prediction
            plt.imshow(noise_subtracted.numpy().reshape(config.IMAGE_SIZE))
            plt.title("Noise subtracted")
            plt.axis("off")
            plt.savefig(f"{plot_save_dir}/{i}_timestep.png")
            plt.close()
