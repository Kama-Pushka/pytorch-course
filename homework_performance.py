import torch
import time
from torch.cuda import Event
from tabulate import tabulate

## 3.1 Подготовка данных (5 баллов)
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами

sizes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]

matrices_cpu = []
for size in sizes:
    matrices_cpu.append(torch.rand(*size))

# переместим матрицы на GPU, если доступно
device = 'cuda' if torch.cuda.is_available() else 'cpu'
matrices_gpu = [m.to(device) for m in matrices_cpu]

### 3.2 Функция измерения времени (5 баллов)
# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU

def measure_time(func, tensor_a, tensor_b=None, gpu=False):
    """Функция подсчета времени выполнения операций с тензорами."""
    if gpu and torch.cuda.is_available():
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        start_event.record()
        func(tensor_a, tensor_b) if tensor_b is not None else func(tensor_a)
        end_event.record()

        torch.cuda.synchronize()  # ждем завершения операций на GPU
        return start_event.elapsed_time(end_event)
    else:
        start_time = time.time()
        func(tensor_a, tensor_b) if tensor_b is not None else func(tensor_a)
        end_time = time.time()
        return (end_time - start_time) * 1000  # преобразование в милисекунды


### 3.3 Сравнение операций (10 баллов)
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде

operations = {
    'Матричное умножение': lambda a, b: torch.matmul(a, b.permute(0, 2, 1)),
    'Поэлементное сложение': lambda a, b: a + b,
    'Поэлементное умножение': lambda a, b: a * b,
    'Транспонирование': lambda a: a.permute(0, 2, 1),
    'Сумма всех элементов': lambda a: torch.sum(a)
}

results = {}
for op_name, operation in operations.items():
    cpu_times = []
    gpu_times = []

    for i in range(len(sizes)):
        matrix_a = matrices_cpu[i].detach()
        # если аргументов два - инициализируем второй
        matrix_b = matrices_cpu[i].detach() if operation.__code__.co_argcount == 2 else None

        cpu_time = measure_time(operation, matrix_a, matrix_b)
        cpu_times.append(cpu_time)

        if torch.cuda.is_available():
            matrix_a_gpu = matrices_gpu[i].detach()
            matrix_b_gpu = matrices_gpu[i].detach() if operation.__code__.co_argcount == 2 else None

            gpu_time = measure_time(operation, matrix_a_gpu, matrix_b_gpu, gpu=True)
            gpu_times.append(gpu_time)
        else:
            gpu_times.append(float('inf'))  # бесконечность, если GPU отсутствует

    results[op_name] = {'CPU': cpu_times, 'GPU': gpu_times}

# создаем сравнительную таблицу
table_data = []
headers = ['Операция', 'Размер', 'CPU (мс)', 'GPU (мс)', 'Ускорение']
for op_name, times in results.items():
    for idx, size in enumerate(sizes):
        speedup = float('inf') if times['GPU'][idx] == float('inf') else round(times['CPU'][idx] / times['GPU'][idx], 2)
        table_row = [
            op_name,
            str(size),
            f"{round(times['CPU'][idx], 3)}",
            f"{round(times['GPU'][idx], 3)}" if times['GPU'][idx] != float('inf') else '-',
            f'{speedup}x' if speedup != float('inf') else '-'
        ]
        table_data.append(table_row)

print(tabulate(table_data, headers=headers, tablefmt="grid"))

### 3.4 Анализ результатов (5 баллов)
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
# - Почему некоторые операции могут быть медленнее на GPU?
# - Как размер матриц влияет на ускорение?
# - Что происходит при передаче данных между CPU и GPU?

# В README.md