import tensorflow as tf

from diffusion.plot_images_callback import PlotImagesCallback


def train_model(
    model: tf.keras.Model, dataset: tf.data.Dataset, epochs: int, batch_size: int
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    plot_images_callback = PlotImagesCallback(dataset, model)

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[
            plot_images_callback,
            # tf.keras.callbacks.EarlyStopping(monitor="loss", patience=6),
        ],
    )
