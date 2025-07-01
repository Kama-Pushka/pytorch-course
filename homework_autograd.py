import torch

### 2.1 Простые вычисления с градиентами (8 баллов)
# Создайте тензоры x, y, z с requires_grad=True
x = torch.rand(2, requires_grad=True)
y = torch.rand(2, requires_grad=True)
z = torch.rand(2, requires_grad=True)
print(x, y, z)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = (x**2 + y**2 + z**2 + 2*x*y*z)
print(f)

# Найдите градиенты по всем переменным
f = f.sum()
f.backward()
print(x.grad, y.grad, z.grad)

# Проверьте результат аналитически
print(2*x + 2*y*z, 2*y + 2*x*z, 2*z + 2*y*x)
print("------------")

### 2.2 Градиент функции потерь (9 баллов)
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
x = torch.arange(1, 5, 1)
y_true = torch.arange(1, 5, 1) * 2
print(x)
print(y_true)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y_pred = w * x + b
print(y_pred)

MSE = (1 / len(y_pred)) * sum((y_pred - y_true)**2)
print(MSE)
MSE.backward()

# Найдите градиенты по w и b
print(w.grad)
print(b.grad)
print("------------")

### 2.3 Цепное правило (8 баллов)
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.arange(1, 5, 1, dtype=torch.float32, requires_grad=True)
f = torch.sin(x**2 + 1)

# Найдите градиент df/dx
grad = 2 * x * torch.cos(x**2 + 1)
print(grad)

# Проверьте результат с помощью torch.autograd.grad
autograd = torch.autograd.grad(f.sum(), x)
print(autograd)