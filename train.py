import os
from enum import Enum
from typing import Union

import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras

from gan_class import ConditionalGAN
from gen_images_callback import ImageGenerationCallback
from loss_plot_callback import LossPlotCallback
from model_checkpoint_callback import ModelCheckpointCallback
from lightweight_gen import create_lightweight_generator as create_generator
from lightweight_disc import create_lightweight_discriminator as create_discriminator


def train():
    training_active = True

    # Suppressing tf.hub warnings
    tf.get_logger().setLevel("ERROR")

    # configure the GPU
    keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Restrict TensorFlow to only allocate 80% of the total memory of each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=1024 * 10
                        )
                    ],
                )
        except RuntimeError as e:
            print(e)



    class AvailableDatasets(Enum):
        FASHION_MNIST = "fashion_mnist"
        CELEBA = "celeba"
        FROMPATH = "frompath"

    active_dataset: AvailableDatasets = AvailableDatasets.CELEBA

    # data_images_dir: Union[str, None] = "real_images_segmented_cats"
    data_images_dir: Union[str, None] = None

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    generated_images_dir = "generated_images"
    os.makedirs(generated_images_dir, exist_ok=True)

    # set the parameters for dataset
    image_side_length = 64
    target_size = (image_side_length, image_side_length)
    channels = 1
    image_shape = (target_size[0], target_size[1], channels)
    increase_factor = 4
    BATCH_SIZE = int(64 * increase_factor)
    noise_dim = 200
    start_epoch = 0
    EPOCHS = 5000
    num_examples_to_generate = 9
    # Default lr is 1e-4
    generator_factor = 1
    discriminator_factor = 0.01
    generator_lr = 1e-4 * 0.25 * increase_factor * generator_factor
    discriminator_lr = 1e-4 * 0.25 * increase_factor * discriminator_factor

    # Normalization helper
    def preprocess(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.resize(tf.cast(image, tf.float32) / 127.5 - 1, target_size)
        return image, label

    # Create augmentation layers
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ]
    )

    def augment(image, label):
        image = data_augmentation(image)
        return image, label

    def get_datasets() -> tf.data.Dataset:
        color_mode = "grayscale"
        if channels == 3:
            color_mode = "rgb"

        train_ds = None
        if active_dataset == AvailableDatasets.CELEBA:
            celeb_a_path = "data/celeb_a"
            train_ds = keras.utils.image_dataset_from_directory(
                celeb_a_path,
                label_mode=None,
                image_size=target_size,
                batch_size=None,
                color_mode=color_mode,
                shuffle=True,
            ).map(lambda x: (x, 0), num_parallel_calls=tf.data.AUTOTUNE)


        elif active_dataset == AvailableDatasets.FASHION_MNIST:
            used_class = 5
            (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
            train_images = train_images[train_labels == used_class]
            train_labels = train_labels[train_labels == used_class]
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_ds = train_ds.map(
                lambda x, y: (x, 0), num_parallel_calls=tf.data.AUTOTUNE
            )
            # Unsqueezing the channel dimension so images are (28, 28, 1) instead of (28, 28)
            train_ds = train_ds.map(
                lambda x, y: (tf.expand_dims(x, -1), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif active_dataset == AvailableDatasets.FROMPATH and data_images_dir is not None:
            train_ds = keras.utils.image_dataset_from_directory(
                data_images_dir,
                label_mode=None,
                image_size=target_size,
                batch_size=None,
                color_mode=color_mode,
                shuffle=True,
            ).map(lambda x: (x, 0), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            raise ValueError("No dataset selected")
        # Normalize to [-1, 1], shuffle and batch
        train_ds = (
            train_ds.map(augment)
            .map(preprocess, tf.data.AUTOTUNE)
            .shuffle(BATCH_SIZE * 6)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Show 9 samples
        # plt.figure(figsize=(10, 10))
        # for images, _ in train_ds.take(1):
        #     for i in range(9):
        #         plt.subplot(3, 3, i + 1)
        #         plt.imshow((images[i].numpy() + 1) / 2)
        #         plt.axis("off")
        # plt.show()
        # plt.close()
        return train_ds

    dataset = get_datasets()
    total_batches = len(list(dataset))
    total_samples = total_batches * BATCH_SIZE
    steps_per_epoch = total_samples // BATCH_SIZE

    # Learning rate schedulers
    # decay_steps = 5 * steps_per_epoch
    # gen_lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=generator_lr,
    #     decay_steps=10000,
    #     decay_rate=0.95,
    #     staircase=True,
    # )
    #
    # disc_lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=discriminator_lr,
    #     decay_steps=10000,
    #     decay_rate=0.95,
    #     staircase=True,
    # )

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(generator_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr)

    # Load from models directory
    models_list = os.listdir(models_dir)
    generator_models = [model for model in models_list if "generator" in model]
    discriminator_models = [model for model in models_list if "discriminator" in model]

    generator = create_generator(noise_dim, image_side_length, channels)
    discriminator = create_discriminator(image_shape)

    if generator_models and discriminator_models:
        saved_epochs = [
            int(model.split("_")[-1].split(".")[0]) for model in generator_models
        ]
        latest_saved_epoch = max(saved_epochs)
        latest_saved_epoch_generator_file = [
            model for model in generator_models if str(latest_saved_epoch) in model
        ][0]
        latest_saved_epoch_discriminator_file = [
            model for model in discriminator_models if str(latest_saved_epoch) in model
        ][0]

        if latest_saved_epoch_generator_file and latest_saved_epoch_discriminator_file:
            generator.load_weights(
                os.path.join(models_dir, latest_saved_epoch_generator_file)
            )
            discriminator.load_weights(
                os.path.join(models_dir, latest_saved_epoch_discriminator_file)
            )
            start_epoch = latest_saved_epoch
            print(f"Loaded models from epoch {start_epoch}")

    discriminator.summary()
    generator.summary()

    gan = ConditionalGAN(
        discriminator=discriminator,
        generator=generator,
        noise_dim=noise_dim,
    )
    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_lr),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=generator_lr),
    )

    loss_plot_callback = LossPlotCallback(plot_interval=1)
    checkpoint_callback = ModelCheckpointCallback(
        generator=generator,
        discriminator=discriminator,
        save_path=models_dir,
        save_freq=10,
    )
    image_gen_callback = ImageGenerationCallback(
        generator_model=gan.generator,
        noise_dim=200,
        num_images=9,
        generate_interval=10,
        output_dir=generated_images_dir,
    )

    gan.fit(
        dataset,
        initial_epoch=start_epoch,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        callbacks=[loss_plot_callback, image_gen_callback, checkpoint_callback],
    )


if __name__ == "__main__":
    train()
