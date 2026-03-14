from dataclasses import dataclass


@dataclass
class ModelSettings:
    in_channels: int = 3
    num_classes: int = 10
    num_filters: int = 32
    num_layers: int = 3
    num_hidden_units: int = 128
    kernel_size: int = 3
    padding: int = 1
    max_pool_kernel: int = 2
    img_size: int = 32


@dataclass
class TrainingSettings:
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.001
    mean: tuple[float, float, float] = (0.4377, 0.4438, 0.4728)
    std: tuple[float, float, float] = (0.1980, 0.2010, 0.1970)