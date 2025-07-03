import torch
import json

class FCN(torch.nn.Module):
    def __init__(self, config_path: str = None, input_size: int = None, num_classes: int = 10, **kwargs):
        super().__init__()

        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = kwargs
        self.input_size = input_size or self.config.get('input_size', 28 * 28)
        self.num_classes = num_classes or self.config.get('num_classes', 10)

        self.layers = self._build_layers()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    # Чтобы можно было динамически генерить модели из конфига
    def _build_layers(self):
        """
        {
            'layers': {
                {'type': 'linear', size: 1000},
                {'type': 'relu'},
                {'type': 'sigmoid'},
                {'type': 'dropout', 'rate': 0.1},
                {'type': 'batch_norm'},
            }
        }
        """
        layers = []
        prev_size = self.input_size

        layer_config = self.config['layers']

        for layer in layer_config:
            layer_type = layer['type']
            match layer_type:
                case 'linear':
                    layers.append(torch.nn.Linear(prev_size, layer['size']))
                    prev_size = layer['size']
                case 'relu':
                    layers.append(torch.nn.ReLU())
                case 'sigmoid':
                    layers.append(torch.nn.Sigmoid())
                case 'dropout':
                    layers.append(torch.nn.Dropout(layer['rate']))
                case 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(prev_size)) # т.к. модель линейная - 1d, для свертки уже 2d...
                case _:
                    raise ValueError(f'Unknown layer type: {layer_type}')
        layers.append(torch.nn.Linear(prev_size, self.num_classes)) # выходной слой, для классификации на выходе нужно иметь число нейронов = кол-ву классов в датасете

        """ эквивалент
        for layer in layers:
            x = layer.forward(x)
        """
        layers = torch.nn.Sequential(*layers)
        #layers.forward(torch.randn(1, self.input_size))

        return layers

    def forward(self, x):
        x = x.view(x.size(0), -1) # выпрямляем изображение
        return self.layers(x)

if __name__ == "__main__":
    config = {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
        ]
    }

    model = FCN(**config)
    print(model)