from tensorflow.keras.callbacks import Callback
import os
import tensorflow as tf
import matplotlib.pyplot as plt


class ImageGenerationCallback(Callback):
    def __init__(
        self,
        generator_model,
        noise_dim,
        num_images=9,
        generate_interval=5,
        output_dir="generated_images",
    ):
        super().__init__()
        self.generator_model = generator_model
        self.noise_dim = noise_dim
        self.num_images = num_images
        self.generate_interval = generate_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.generate_interval == 0:
            self.generate_and_save_images(epoch + 1)

    def generate_and_save_images(self, epoch):
        random_noise = tf.random.normal([self.num_images, self.noise_dim])
        predictions = self.generator_model(random_noise, training=False)

        fig = plt.figure(figsize=(9, 9), num=epoch)
        plt.clf()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.suptitle(f"Epoch {epoch}", fontsize=16)
        for i in range(predictions.shape[0]):
            plt.subplot(3, 3, i + 1)
            # Adjust the pixel values to be between 0 and 255 if the output activation is tanh
            plt.imshow((predictions[i, :, :, 0] * 127.5 + 127.5).numpy(), cmap="gray")
            plt.axis("off")

        plt.savefig(os.path.join(self.output_dir, f"image_at_epoch_{epoch}.png"))
        plt.close(fig)
