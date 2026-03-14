import torch.nn as nn
from src.settings import ModelSettings

class SimpleConvModel(nn.Module):
    def __init__(self, settings: ModelSettings):
        super().__init__()

        layers = []
        in_channels = settings.in_channels
        out_channels = settings.num_filters

        for _ in range(settings.num_layers):
            layers += [
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=settings.kernel_size,
                          padding=settings.padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=settings.max_pool_kernel)
            ]
            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*layers)

        feature_map_size = settings.img_size // (2**settings.num_layers)
        flat_size = feature_map_size ** 2 * in_channels

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=flat_size,
                out_features=settings.num_hidden_units
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=settings.num_hidden_units,
                out_features=settings.num_classes
            ))
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
