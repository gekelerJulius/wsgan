from tensorflow.keras.callbacks import Callback
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from gen_model import create_generator


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

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.generate_interval == 0:
            self.generate_and_save_images(epoch + 1)

    def generate_and_save_images(self, epoch: int):
        random_noise = tf.random.normal([self.num_images, self.noise_dim])
        predictions = self.generator_model(random_noise, training=False)

        fig = plt.figure(figsize=(9, 9), num=epoch)
        plt.clf()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.suptitle(f"Epoch {epoch}", fontsize=16)
        for i in range(predictions.shape[0]):
            plt.subplot(3, 3, i + 1)
            plt.imshow((predictions[i] * 127.5 + 127.5).numpy().astype(int))
            plt.axis("off")
        plt.savefig(os.path.join(self.output_dir, f"image_at_epoch_{epoch}.png"))
        plt.close(fig)


if __name__ == "__main__":
    noise_dim = 200
    image_side_length = 32
    channels = 3
    num_classes = 10

    generator = create_generator(noise_dim, image_side_length, channels)
    image_generation_callback = ImageGenerationCallback(
        generator_model=generator,
        noise_dim=noise_dim,
        num_images=9,
        output_dir="generated_images_testing",
    )
    image_generation_callback.generate_and_save_images(0)
