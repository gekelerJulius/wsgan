import os
import numpy as np
import tensorflow as tf
from keras import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from diffusion import config


class ImageGenerationCallback(Callback):
    def __init__(self, plot_interval: int, model: Model, num_timesteps: int):
        super().__init__()
        self.plot_interval = plot_interval
        self.model = model
        self.num_timesteps = num_timesteps

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_interval == 0:
            self.generate_image(epoch)

    def generate_image(self, epoch):
        # Ensure the directory for saving plots exists
        plot_save_dir = f"{config.PLOTS_DIR}/{epoch + 1}_epoch"
        os.makedirs(plot_save_dir, exist_ok=True)

        # Start with a noise image
        noise_image = tf.random.normal(
            shape=(1, *config.IMAGE_SIZE, config.CHANNELS), mean=0.0, stddev=1.0
        )

        current_image = noise_image
        images = [current_image]  # Save initial noise image
        for i in range(1, self.num_timesteps):
            # Predict the noise at this step
            predicted_noise = self.model.predict(current_image)
            # Subtract the predicted noise from the current image
            current_image -= predicted_noise

            # Save images at intervals for visualization
            if i == int(self.num_timesteps * len(images) / 9):
                images.append(current_image)

        images.append(current_image)  # Ensure the last image is saved

        # Plotting
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for ax, img in zip(axes.flatten(), images):
            ax.imshow(
                tf.clip_by_value(img, -1.0, 1.0).numpy().reshape(config.IMAGE_SIZE)
            )
            ax.axis("off")
        plt.suptitle(f"Image Generation Progression at Epoch {epoch + 1}")
        plt.savefig(f"{plot_save_dir}/generation_progress.png")
        plt.close()
