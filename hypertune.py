from src.settings import TrainingSettings, ModelSettings
from src.data import get_dataloaders
from src.model import SimpleConvModel

import torch
import torch.nn as nn

from loguru import logger

train_settings = TrainingSettings()
model_settings = ModelSettings()

train_data, test_data = get_dataloaders(train_settings)

model = SimpleConvModel(model_settings)
loss_function = nn.CrossEntropyLoss()
optimzier = torch.optim.Adam(model.parameters(), lr=train_settings.learning_rate)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")
model = model.to(device)

for _ in range(train_settings.epochs):
    print()
