import random

import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import List, Dict


class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name: str, augmentation_func):
        """ Добавляет новое преобразование в pipeline. """
        self.augmentations[name] = augmentation_func

    def remove_augmentation(self, name: str):
        """ Удаляет указанное преобразование из pipeline. """
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image: Image.Image) -> Image.Image:
        """ Применяет все зарегистрированные преобразования к изображению. """
        result = image
        for _, func in self.augmentations.items():
            result = func(result)
        return result

    def get_augmentations(self) -> List[str]:
        """ Возвращает список зарегистрированных преобразований. """
        return list(self.augmentations.keys())

def random_horizontal_flip(img: Image.Image, prob=0.5):
    if random.random() < prob:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_rotation(img: Image.Image, degrees=(-10, 10)):
    angle = random.uniform(*degrees)
    return img.rotate(angle, expand=True)


def gaussian_noise(img: Image.Image, mean=0, stddev=10):
    img_array = np.array(img)
    noise = np.random.normal(loc=mean, scale=stddev, size=img_array.shape)
    noisy_array = img_array + noise.astype(np.int16)  # Используем int16 для предотвращения потери точности
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def color_jitter(img: Image.Image, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    return jitter(img)

def resize_to_square(img: Image.Image, size=224):
    return img.resize((size, size))

pipeline_light = AugmentationPipeline()
pipeline_medium = AugmentationPipeline()
pipeline_heavy = AugmentationPipeline()

# Light configuration
pipeline_light.add_augmentation("resize", lambda img: resize_to_square(img))
pipeline_light.add_augmentation("flip", random_horizontal_flip)
pipeline_light.add_augmentation("jitter", lambda img: color_jitter(img, brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05))

# Medium configuration
pipeline_medium.add_augmentation("resize", lambda img: resize_to_square(img))
pipeline_medium.add_augmentation("flip", random_horizontal_flip)
pipeline_medium.add_augmentation("rotate", random_rotation)
pipeline_medium.add_augmentation("jitter", lambda img: color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

# Heavy configuration
pipeline_heavy.add_augmentation("resize", lambda img: resize_to_square(img))
pipeline_heavy.add_augmentation("flip", random_horizontal_flip)
pipeline_heavy.add_augmentation("rotate", lambda img: random_rotation(img, degrees=(-30, 30)))  # Более сильный угол поворота
pipeline_heavy.add_augmentation("noise", gaussian_noise)
pipeline_heavy.add_augmentation("jitter", lambda img: color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))

import os
from tqdm import tqdm

images_dir = 'data/train/'
output_dirs = ["results_light", "results_medium", "results_heavy"]

for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(images_dir):
    for filename in tqdm(files):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(root, filename)
            img = Image.open(img_path)

            # Light pipeline
            light_result = pipeline_light.apply(img)
            light_result.save(os.path.join(output_dirs[0], filename))

            # Medium pipeline
            medium_result = pipeline_medium.apply(img)
            medium_result.save(os.path.join(output_dirs[1], filename))

            # Heavy pipeline
            heavy_result = pipeline_heavy.apply(img)
            heavy_result.save(os.path.join(output_dirs[2], filename))