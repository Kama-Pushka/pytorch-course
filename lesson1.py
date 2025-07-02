import torch

print(torch.cuda.is_available())

tensor = torch.zeros(2, 3, device='cuda:0')
tensor_one = torch.ones(2, 3, device='cuda:0')
tensor_arange = torch.arange(0, 10, 1, dtype=torch.float32, device=torch.device('cuda'))
print(tensor, tensor_one, tensor_arange)

tensor_random = torch.rand(5, 5)
tensor_randint = torch.randint(1, 5, (5, 5))
print(tensor_random)
print(tensor_randint)

#

tensor1 = torch.tensor([1,2,3,4])
tensor2 = torch.tensor([[5, 6], [7, 8]])
tensor3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tensor4 = torch.rand(1, 2, 3, 4)

print(tensor1, tensor2, tensor3, tensor4)
print(tensor1.size(), tensor1.shape, tensor1.ndim) # same
print(tensor2.size(), tensor2.shape, tensor2.ndim)
print(tensor3.size(), tensor3.shape, tensor3.ndim)
print(tensor4.size(), tensor4.shape, tensor4.ndim)

# Slice

print(tensor4[0, :, 2, :2])
print(tensor4[:, :2])
print(tensor4[0, 1, 2, 3])

print(tensor4[tensor4>0.1])
print(tensor4>0.1)

# изменения размерности

tensor5 = torch.arange(0, 20, 1)

print(tensor5.shape)
tensor6 = tensor5.view(10, 2) # тот же самый кусок памяти
tensor7 = tensor5.reshape(5, 2, 2) # может возвращать копию (если тензор разбросан по памяти)
print(tensor6)
print(tensor7)

tensor8 = tensor5.view(-1, 2, 2, 1) # вместо -1 вычисл крайнее значение = 5
print(tensor8)

# добавление/удаление размерности

tensor9 = tensor5.unsqueeze(0) # увеличить размерность по измерению
print(tensor5, tensor5.shape)
print(tensor9, tensor9.shape)

print(tensor9.squeeze(0)) # уменьшение размерности (не оч понял как работает)

# перестановки местами размерностей

print(tensor9.permute(1,0), tensor9.shape) # или transpose() (реже)

tensor10 = tensor5.unsqueeze(0).reshape(1, -1, 5) # [1, 4, 5]
tensor10 = tensor10.permute(2, 1, 0) # делает его разделенным (нелинейным!) в памяти

print(tensor10.is_contiguous())
tensor10 = tensor10.contiguous() # создает объект линейный в памяти
print(tensor10.is_contiguous())

# "каст" (перемещение между устройствами, касты к типам)

tensor11 = torch.rand(3, 4)
tensor11 = tensor11.to('cuda:0')
print(tensor11)
tensor11 = tensor11.to('cpu')
print(tensor11)

print(tensor11.to(torch.int))
print(tensor11.int())

print(tensor11.numpy(), tensor11.tolist())

### Действия с тензорами

a = torch.rand(3, 4) # поэлементно
b = torch.rand(3, 4)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(torch.pow(a, 10))
print(torch.sqrt(a))
print(torch.exp(a))
print(torch.log(a))

print(a @ b.T) # умножение матриц (по всем правилам, в т.ч. размерности!)

a = torch.rand(10, 3, 4) # поэлементно
b = torch.rand(10, 3, 4)

print(a @ b.permute(0, 2, 1)) # batch matrix mult

print(a == b) # вернет маску

print(a)
print(a.sum())
print(a.mean())
print(a.min(), a.argmax(0), a.argmax(1)) # индекс по измерению
print(a.max(), a.argmin(0), a.argmin(1))
print(a.prod()) # произведение элементов
print(a.std()) # стандартное отклонение
print(a.var()) # вариация

# Broadcast (последние измерения должны совпадать!)

a = torch.randint(1, 4, (2, 3))
b = torch.randint(1, 4, (3, ))
print(a, b)
print(a + b)
print(a @ b)

# Autograd (вычисления градиента) (вычисление производной?)

a = torch.rand(2, requires_grad=True)
b = torch.rand(2, requires_grad=True)

print(a, b)
print(a.grad, b.grad)

y = (a**2 + 17 * b).sum()
print(y)
y.backward() # обратное распространение
print(a.grad, b.grad)

#

a = torch.rand(2)
b = torch.rand(2)

print(a, b)
a.add_(b)
print(a)

a.zero_()
a = b.clone()

# копия без градиентов

a = torch.rand(2, requires_grad=True)

y = (a**2 + 3*a).sum()

b = y.detach() # полная копия + убираем градиент
y.backward()
# b.backward() # градиента нет

# ИЛИ (убираем градиенты через no_grad()) - лучше так

with torch.no_grad():
    y = (a**2 + 3*a).sum()
print(y, y.grad_fn)