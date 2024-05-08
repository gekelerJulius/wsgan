# Configuration parameters
import os

INCREASE_FACTOR = 1
IMAGE_SIDE_LENGTH = int(28 * INCREASE_FACTOR)
IMAGE_SIZE = (IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3 * INCREASE_FACTOR
EPOCHS = 1000
CHANNELS = 1
TIMESTEPS = 200
UNET_DEPTH = 2
DATA_DIR = "D:\\Programming\\wsgan\\data\\real_images_segmented_cats"
PLOTS_DIR = "D:\\Programming\\wsgan\\diffusion\\plots"
MODELS_DIR = "D:\\Programming\\wsgan\\diffusion\\models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"IMAGE_SIDE_LENGTH: {IMAGE_SIDE_LENGTH}")
print(f"IMAGE_SIZE: {IMAGE_SIZE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print(f"CHANNELS: {CHANNELS}")
print(f"TIMESTEPS: {TIMESTEPS}")
