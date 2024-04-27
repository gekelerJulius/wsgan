import os
import tensorflow as tf
from keras import layers
from tensorflow import keras

from discriminator_model import create_discriminator
from gan_class import GAN, ImplementedLosses
from gen_images_callback import ImageGenerationCallback
from gen_model import create_generator
from loss_plot_callback import LossPlotCallback
from model_checkpoint_callback import ModelCheckpointCallback

# images_path = "D:\\Programming\\pixilart_api_backend\\downloaded_images\\pixel_art_cats"
images_path = None
training_active = True

# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

# configure the GPU
keras.mixed_precision.set_global_policy("mixed_float16")
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

# set the parameters for dataset
image_side_length = 28
target_size = (image_side_length, image_side_length)
channels = 1
image_shape = (target_size[0], target_size[1], channels)
increase_factor = 8
BATCH_SIZE = 64 * increase_factor
noise_dim = 200
start_epoch = 0
EPOCHS = 500
num_examples_to_generate = 9
# Default lr is 1e-4
generator_factor = 1
discriminator_factor = 0.05
generator_lr = 1e-4 * 0.25 * increase_factor * generator_factor
discriminator_lr = 1e-4 * 0.25 * increase_factor * discriminator_factor

generated_images_dir = "generated_images"
os.makedirs(generated_images_dir, exist_ok=True)

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Normalization helper
def preprocess(image: tf.Tensor, label: tf.Tensor):
    image = tf.image.resize(tf.cast(image, tf.float32) / 127.5 - 1, target_size)
    return image, label


# Create augmentation layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.3),
    ]
)


def augment(image, label):
    image = data_augmentation(image)
    return image, label


def get_datasets() -> tf.data.Dataset:
    color_mode = "grayscale"
    if channels == 3:
        color_mode = "rgb"
    if images_path is not None:
        train_ds = keras.utils.image_dataset_from_directory(
            images_path,
            label_mode=None,
            image_size=target_size,
            batch_size=None,
            color_mode=color_mode,
            shuffle=True,
        ).map(lambda x: (x, 0), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Use MNIST dataset
        (train_images, _), (_, _) = keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
        train_ds = tf.data.Dataset.from_tensor_slices(train_images).map(
            lambda x: (x, 0), num_parallel_calls=tf.data.AUTOTUNE
        )

    # Normalize to [-1, 1], shuffle and batch
    train_ds = (
        train_ds.map(augment)
        .map(preprocess, tf.data.AUTOTUNE)
        .shuffle(BATCH_SIZE * 10)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Show a few samples
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for j in range(4):
    #         plt.subplot(2, 2, j + 1)
    #         plt.imshow(images[j].numpy().reshape(target_size), cmap="gray")
    #         plt.title(f"Label: {labels[j]}")
    #         plt.axis("off")
    # plt.show()
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

if generator_models:
    latest_generator_model = sorted(generator_models)[-1]
    generator.load_weights(os.path.join(models_dir, latest_generator_model))
    print(f"Loaded generator model: {latest_generator_model}")

if discriminator_models:
    latest_discriminator_model = sorted(discriminator_models)[-1]
    discriminator.load_weights(os.path.join(models_dir, latest_discriminator_model))
    print(f"Loaded discriminator model: {latest_discriminator_model}")

discriminator.summary()
generator.summary()

gan = GAN(discriminator=discriminator, generator=generator, noise_dim=noise_dim)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_lr),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=generator_lr),
    loss_type=ImplementedLosses.WASSERSTEIN,
)

loss_plot_callback = LossPlotCallback(plot_interval=1)
checkpoint_callback = ModelCheckpointCallback(
    generator=generator, discriminator=discriminator, save_path=models_dir, save_freq=10
)
image_gen_callback = ImageGenerationCallback(
    generator_model=gan.generator,
    noise_dim=200,
    num_images=9,
    generate_interval=5,
    output_dir=generated_images_dir,
)


gan.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    batch_size=BATCH_SIZE,
    verbose=1,
    initial_epoch=start_epoch,
    shuffle=True,
    callbacks=[loss_plot_callback, image_gen_callback, checkpoint_callback],
)
