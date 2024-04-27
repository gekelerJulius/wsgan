from tensorflow.keras.callbacks import Callback
import os


class ModelCheckpointCallback(Callback):
    def __init__(self, generator, discriminator, save_path, save_freq=5):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.generator.save_weights(
                os.path.join(self.save_path, f"generator_epoch_{epoch + 1}.h5")
            )
            self.discriminator.save_weights(
                os.path.join(self.save_path, f"discriminator_epoch_{epoch + 1}.h5")
            )
            print(f"\nSaved weights at epoch {epoch + 1}")
