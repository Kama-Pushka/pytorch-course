import timeit

import pandas as pd
import psutil
import gc
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def measure_time(func, args=(), num_runs=10):
    timer = timeit.Timer(lambda: func(*args))
    execution_times = timer.repeat(number=num_runs)
    return sum(execution_times) / len(execution_times)


def memory_usage_mb():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # MB


def load_and_augment_image(image_path, target_size):
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    tensor = transform(img)
    return tensor


import os
import glob

images_dir = 'data/train/'

all_images = glob.glob(os.path.join(images_dir, '*/*'))

resolutions = [(64, 64), (128, 128), (224, 224), (512, 512)]

results = {"resolution": [], "load_time": [], "memory_usage": []}

for resolution in resolutions:
    side_length = max(resolution)
    total_load_time = 0
    initial_memory = memory_usage_mb()

    for image_path in all_images[:100]:  # Берём первые 100 изображений
        load_time = measure_time(load_and_augment_image, args=(image_path, resolution))
        total_load_time += load_time
        gc.collect()  # Освобождаем память после каждой итерации

    results["resolution"].append(side_length) # side_length x side_length
    results["load_time"].append(total_load_time / 100)  # Среднее время загрузки
    results["memory_usage"].append(memory_usage_mb() - initial_memory)  # Потребление памяти

df_results = pd.DataFrame(results)

# График времени загрузки
plt.figure(figsize=(10, 6))
sns.lineplot(x="resolution", y="load_time", marker="o", data=df_results)
plt.title("Время загрузки и обработки изображений разных размеров")
plt.xlabel("Размер стороны изображения")
plt.ylabel("Среднее время (секунды)")
plt.grid(True)
plt.show()

# График потребления памяти
plt.figure(figsize=(10, 6))
sns.lineplot(x="resolution", y="memory_usage", marker="o", data=df_results)
plt.title("Потребление памяти при обработке изображений разных размеров")
plt.xlabel("Размер стороны изображения")
plt.ylabel("Память (Мб)")
plt.grid(True)
plt.show()