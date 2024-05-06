import tensorflow as tf
from keras import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


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
        # Gradually apply noise to the first image in the dataset
        first_image = next(iter(self.dataset))[0][0]
        noise = tf.random.normal(shape=tf.shape(first_image))
        # Create a list of images with increasing noise
        images = []
        for i in range(self.num_timesteps):
            images.append(first_image + noise * (i / self.num_timesteps))

        # Generate predictions for each image
        predictions = []
        for image in images:
            predictions.append(self.model.predict(image[tf.newaxis, ...]))

        # Plot the original image, noisy images, and predictions
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(first_image), cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(tf.squeeze(images[-1]), cmap="gray")
        plt.title("Noisy Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(predictions[-1]), cmap="gray")
        plt.title("Predicted Image")
        plt.axis("off")

        plt.suptitle(f"Epoch {epoch + 1}")
        plt.pause(0.1)
        plt.savefig(f"diffusion_epoch_{epoch + 1}.png")
        plt.close()
