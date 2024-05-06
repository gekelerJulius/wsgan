import os

import numpy as np
from matplotlib import pyplot as plt

from diffusion.diffusion_visualizer_callback import DiffusionVisualizerCallback
from diffusion.unet_model import UNetModel
from helpers.get_dataset import get_dataset
import config
import tensorflow as tf
from tensorflow import keras

from helpers.loss_plot_callback import LossPlotCallback


def main():
    dataset = get_dataset(
        channels=config.CHANNELS,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        data_images_dir=config.DATA_DIR,
    )

    model_weights_path = (
        f"{config.EPOCHS}_epochs_"
        f"{config.TIMESTEPS}_timesteps_"
        f"{config.BATCH_SIZE}_batch_size_"
        f"{config.LEARNING_RATE}_lr_model_weights.h5"
        f"{config.UNET_DEPTH}_depth.h5"
    )
    model = UNetModel(
        image_side_length=config.IMAGE_SIDE_LENGTH,
        channels=config.CHANNELS,
        num_timesteps=config.TIMESTEPS,
    )
    model.build(input_shape=(None, *config.IMAGE_SIZE, config.CHANNELS))
    (
        model.load_weights(model_weights_path)
        if os.path.exists(model_weights_path)
        else None
    )

    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = keras.losses.MeanSquaredError()
    print("Compiling model...")
    model.compile(optimizer=optimizer, loss=loss_fn)
    print("Model summary:")
    model.summary()
    print("Training model...")

    loss_plot_callback = LossPlotCallback(
        plot_interval=1,
        labels=["loss"],
    )
    diff_vis_callback = DiffusionVisualizerCallback(
        plot_interval=1,
        model=model,
        dataset=dataset,
        num_timesteps=config.TIMESTEPS,
    )

    model.fit(
        dataset,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1,
        shuffle=True,
        callbacks=[
            loss_plot_callback,
            diff_vis_callback,
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                "model_weights.h5",
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ],
    )
    print("Model trained!")

    # Save weights
    model.save_weights("model_weights.h5")
    print("Model weights saved to model_weights.h5!")

    def generate_image(model: keras.Model, num_timesteps: int) -> np.ndarray:
        # Start with random noise
        noise = tf.random.normal(
            [1, *config.IMAGE_SIZE, config.CHANNELS]
        )  # Adjust shape according to your model's input

        image = noise

        for t in range(num_timesteps):
            image = model(image, timestep=t)

        # Convert tensor to numpy for visualization and return
        return image.numpy()

    def plot_image(image: np.ndarray) -> None:
        # Unnormalize image from [-1, 1] to [0, 1]
        image = (image + 1) / 2
        plt.figure(figsize=(10, 10))
        plt.imshow(np.squeeze(image, axis=0), cmap="gray")
        plt.axis("off")
        plt.show()

    # Assuming 'model' is your trained DiffusionModel instance
    generated_images = generate_image(model, num_timesteps=config.TIMESTEPS)
    plot_image(generated_images)


if __name__ == "__main__":
    main()
