import os
import glob
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = 'data/'


def collect_data(base_dir):
    datasets = ['train', 'test']
    data = {'dataset': [], 'class': [], 'height': [], 'width': []}

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        classes = sorted(os.listdir(dataset_dir))

        for cls in classes:
            all_files = []
            for ext in extensions:
                all_files.extend(glob.glob(os.path.join(dataset_dir, cls, ext)))

            print(f"{dataset}/{cls}: {len(all_files)} изображений.")

            for file in all_files:
                try:
                    with Image.open(file) as img:
                        w, h = img.size
                        data['dataset'].append(dataset)
                        data['class'].append(cls)
                        data['height'].append(h)
                        data['width'].append(w)
                except Exception as e:
                    print(f"Пропускаю файл {file}. Причина: {e}")

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = collect_data(base_dir)

    avg_height = df['height'].mean()
    avg_width = df['width'].mean()
    min_size = df[['height', 'width']].min().values
    max_size = df[['height', 'width']].max().values

    print(f"Средний размер изображений: ({avg_height:.2f}, {avg_width:.2f}).")
    print(f"Минимальный размер изображений: ({min_size[0]}, {min_size[1]}).")
    print(f"Максимальный размер изображений: ({max_size[0]}, {max_size[1]}).")

    plt.figure(figsize=(12, 6))
    sns.histplot(df.query("dataset=='train'")['height'], bins=30, kde=True, color='blue', alpha=0.5,
                 label="Train Height")
    sns.histplot(df.query("dataset=='train'")['width'], bins=30, kde=True, color='green', alpha=0.5,
                 label="Train Width")
    sns.histplot(df.query("dataset=='test'")['height'], bins=30, kde=True, color='red', alpha=0.5, label="Test Height")
    sns.histplot(df.query("dataset=='test'")['width'], bins=30, kde=True, color='orange', alpha=0.5, label="Test Width")
    plt.title("Распределение размеров изображений (Train vs Test)")
    plt.xlabel("Размер (пиксели)")
    plt.ylabel("Количество изображений")
    plt.legend()
    plt.show()

    grouped_df = df.groupby(['dataset', 'class']).size().reset_index(name='count')

    g = sns.FacetGrid(grouped_df, col="dataset", height=5, aspect=1.2)
    g.map(sns.barplot, 'class', 'count', order=grouped_df.sort_values(by=['class'])['class'].unique()) # palette="viridis"
    g.set_axis_labels("Класс", "Количество изображений")
    g.fig.suptitle("Количество изображений по классам (Train vs Test)")
    plt.show()