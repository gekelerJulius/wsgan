import os

from diffusion.diffusion_visualizer_callback import DiffusionVisualizerCallback
from diffusion.image_generation_callback import ImageGenerationCallback
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
        # data_images_dir=config.DATA_DIR,
        data_images_dir=None,
    )

    model_weights_name = (
        f"{config.EPOCHS}_epochs_"
        f"{config.TIMESTEPS}_timesteps_"
        f"{config.BATCH_SIZE}_batch_size_"
        f"{config.LEARNING_RATE}_lr_model_weights.h5"
        f"{config.UNET_DEPTH}_depth.h5"
    )
    model_path = os.path.join(config.MODELS_DIR, model_weights_name)
    model = UNetModel(
        image_side_length=config.IMAGE_SIDE_LENGTH,
        channels=config.CHANNELS,
        num_timesteps=config.TIMESTEPS,
    )
    model.build(input_shape=(None, *config.IMAGE_SIZE, config.CHANNELS))
    (model.load_weights(model_path) if os.path.exists(model_path) else None)

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
    img_gen_callback = ImageGenerationCallback(
        plot_interval=1,
        model=model,
        num_timesteps=config.TIMESTEPS,
    )

    # model.fit(
    #     dataset,
    #     epochs=config.EPOCHS,
    #     batch_size=config.BATCH_SIZE,
    #     verbose=1,
    #     shuffle=True,
    #     callbacks=[
    #         loss_plot_callback,
    #         diff_vis_callback,
    #         img_gen_callback,
    #         tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5),
    #         tf.keras.callbacks.ModelCheckpoint(
    #             model_path,
    #             monitor="loss",
    #             save_best_only=True,
    #             save_weights_only=True,
    #         ),
    #     ],
    # )
    # print("Model trained!")

    for i in range(10):
        img_gen_callback.generate_image(i)
    print("Model trained!")


if __name__ == "__main__":
    main()
