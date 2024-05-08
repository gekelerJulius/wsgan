import os
from typing import List

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from diffusion import config


class LossPlotCallback(Callback):
    def __init__(self, plot_interval: int, labels: List[str]):
        super().__init__()
        self.plot_interval = plot_interval
        self.labels = labels
        self.losses = {label: [] for label in labels}

    def on_epoch_end(self, epoch, logs=None):
        for label in self.labels:
            self.losses[label].append(logs[label])

        if (epoch + 1) % self.plot_interval == 0:
            self.draw_losses()

    def draw_losses(self):
        plt.figure(figsize=(10, 5), num="Losses")
        plt.clf()
        for label in self.labels:
            plt.plot(self.losses[label], label=label)
        plt.title("Losses over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend([label for label in self.labels], loc="upper right")
        plt.savefig(os.path.join(config.PLOTS_DIR, "losses.png"))
        plt.close()
