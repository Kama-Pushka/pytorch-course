import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

train_dir = 'data/train'

transformations = {
    'original': None,
    'horizontal_flip': T.RandomHorizontalFlip(p=1),
    'random_crop': T.RandomCrop(size=(224, 224)),
    'color_jitter': T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    'rotation': T.RandomRotation(degrees=(-45, 45)),
    'grayscale': T.RandomGrayscale(p=1),
}

class_folders = sorted(os.listdir(train_dir))[:5]
image_paths = []
for folder in class_folders:
    images_in_folder = os.listdir(os.path.join(train_dir, folder))
    first_image = os.path.join(train_dir, folder, images_in_folder[0])
    image_paths.append(first_image)


def show_image(image):
    """Отображаем изображение."""
    plt.imshow(image)
    plt.axis('off')


for img_path in image_paths:
    original_img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(nrows=2, ncols=len(transformations), figsize=(18, 7))

    for i, (name, transform) in enumerate(transformations.items()):
        if transform is not None:
            augmented_img = transform(original_img)
        else:
            augmented_img = original_img.copy()

        axes[0][i].imshow(augmented_img)
        axes[0][i].set_title(name.capitalize())
        axes[0][i].axis('off')

        transformed_img = original_img
        for t in transformations.values():
            if t is not None:
                transformed_img = t(transformed_img)

        axes[1][i].imshow(transformed_img)
        axes[1][i].set_title(f'Sequential {name}')
        axes[1][i].axis('off')

    plt.show()