from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class LossPlotCallback(Callback):
    def __init__(self, plot_interval=5):
        super().__init__()
        self.plot_interval = plot_interval
        self.gen_losses = []
        self.disc_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.gen_losses.append(logs.get("g_loss"))
        self.disc_losses.append(logs.get("d_loss"))

        if (epoch + 1) % self.plot_interval == 0:
            self.draw_losses()

    def draw_losses(self):
        plt.figure(figsize=(10, 5), num="Losses")
        plt.clf()
        plt.plot(self.gen_losses, label="Generator Loss")
        plt.plot(self.disc_losses, label="Discriminator Loss")
        plt.title("Losses over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.pause(0.01)
        plt.show()
