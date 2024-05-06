from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt

from diffusion import config
from helpers.get_dataset import get_dataset


class PlotImagesCallback(Callback):

    def __init__(self, dataset: tf.data.Dataset, model: tf.keras.Model):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 5 != 0:
            return

        # Show 2 images, unmodified and modified by the model
        imgs = []
        for i in range(2):
            n = next(iter(self.dataset))[0][i]
            n = tf.image.resize(n, (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
            n = tf.cast(n, tf.float32)
            n = (n / 127.5) - 1
            n = tf.expand_dims(n, 0)
            imgs.append(n)

        for i in range(2):
            plt.figure(num=i)
            orig = tf.squeeze(imgs[i])
            model_img = tf.squeeze(self.model.predict(imgs[i]))
            plt.subplot(1, 2, 1)
            plt.imshow(orig, cmap="gray")
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(model_img, cmap="gray")
            plt.title("Model")
            plt.pause(0.1)
