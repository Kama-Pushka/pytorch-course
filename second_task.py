import os
import random
from PIL import ImageFilter, ImageEnhance, Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from lessons.lesson5_extra_augs import *


class CustomAugmentation:
    def apply_random_blur(self, img, max_radius=5, p=0.5):
        """ Случайное гауссово размытие.
        :param max_radius: максимальный радиус размытия (по умолчанию 5)
        :param p: вероятность применения размытия (по умолчанию 0.5)
        """
        if random.random() < p:
            blur_radius = random.uniform(0, max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img

    def apply_perspective_transform(self, img, distortion_scale=0.5, p=0.5):
        """ Перспективное преобразование.
        :param distortion_scale: масштаб искажения перспективы (от 0 до 1)
        :param - p: вероятность применения преобразования (по умолчанию 0.5)
        """
        if random.random() < p:
            width, height = img.size

            # Начало и конец координат углов
            start_points = [(0, 0), (width, 0), (width, height), (0, height)]

            # Макс. смещение угла зависит от размера изображения и scales
            max_dx = int(distortion_scale * width)
            max_dy = int(distortion_scale * height)

            # Рассчитываем случайные координаты для нового положения углов
            distorted_points = [
                (
                    random.uniform(-max_dx, max_dx) + point[0],
                    random.uniform(-max_dy, max_dy) + point[1]
                ) for point in start_points
            ]

            return TF.perspective(img, start_points, distorted_points)
        return img

    def adjust_brightness_and_contrast(self, img, min_brightness=0.5, max_brightness=1.5, min_contrast=0.5, max_contrast=1.5, p=0.5):
        """ Коррекция яркости и контрастности.
        :param min_brightness, max_brightness: минимальные/максимальные значения яркости
        :param min_contrast, max_contrast: минимальные/максимальные значения контрастности
        :param p: вероятность применения преобразования (по умолчанию 0.5) """
        if random.random() < p:
            brightness_factor = random.uniform(min_brightness, max_brightness)
            contrast_factor = random.uniform(min_contrast, max_contrast)
            enhancer = ImageEnhance.Brightness(img)
            brightened_img = enhancer.enhance(brightness_factor)
            enhancer = ImageEnhance.Contrast(brightened_img)
            return enhancer.enhance(contrast_factor)
        return img


train_dir = 'data/train'
custom_augs = CustomAugmentation()

# Сбор первых изображений из каждого класса
class_folders = sorted(os.listdir(train_dir))[:5]
image_paths = [os.path.join(train_dir, folder, os.listdir(os.path.join(train_dir, folder))[0]) for folder in
               class_folders]

num_classes = len(class_folders)

# Добавление изображение на plot
def show_image(ax, img, title=None):
    if not isinstance(img, torch.Tensor):
        to_tensor = T.ToTensor()
        augmented_img = to_tensor(img)
    else:
        augmented_img = img

    aug_np = augmented_img.permute(1, 2, 0).numpy()
    aug_np = np.clip(aug_np, 0, 1)
    ax.imshow(aug_np)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')


for idx, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))

    # Кастомные преобразования
    aug_img_1 = custom_augs.apply_random_blur(img, max_radius=3, p=1.0)
    aug_img_2 = custom_augs.apply_perspective_transform(img, distortion_scale=0.3, p=1.0)
    aug_img_3 = custom_augs.adjust_brightness_and_contrast(img, min_brightness=0.7, max_brightness=1.3,
                                                           min_contrast=0.7, max_contrast=1.3, p=1.0)
    show_image(axes[0][0], aug_img_1, f'{class_folders[idx]}\nRandom Blur')
    show_image(axes[0][1], aug_img_2, 'Perspective Transform')
    show_image(axes[0][2], aug_img_3, 'Brightness & Contrast Adjust.')

    # Преобразования из extra_augs.py
    noise_aug = T.Compose([
        T.ToTensor(),
        AddGaussianNoise(0., 0.2)
    ])
    erase_aug = T.Compose([
        T.ToTensor(),
        RandomErasingCustom(p=1.0)
    ])
    cutout_aug = T.Compose([
        T.ToTensor(),
        CutOut(p=1.0, size=(32, 32))
    ])

    augs_from_extra = noise_aug(img)
    erased_img = erase_aug(img)
    cutout_img = cutout_aug(img)

    show_image(axes[1][0], augs_from_extra, 'Gaussian Noise')
    show_image(axes[1][1], erased_img, 'Random Erasing')
    show_image(axes[1][2], cutout_img, 'Cut Out')

    plt.suptitle(f'Class: {class_folders[idx]} Visualization', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()