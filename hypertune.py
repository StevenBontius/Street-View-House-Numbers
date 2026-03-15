from src.settings import TrainingSettings, ModelSettings
from src.data import get_dataloaders
from src.model import SimpleConvModel

import torch
import torch.nn as nn

from loguru import logger

train_settings = TrainingSettings()
model_settings = ModelSettings()

train_loader, test_loader = get_dataloaders(train_settings)
model = SimpleConvModel(model_settings)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_settings.learning_rate)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")
model = model.to(device)

best_train_loss = float("inf")
best_test_loss = float("inf")
patience_counter = 0

for epoch in range(train_settings.epochs):
    total_train_loss = 0
    total_test_loss = 0
    
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    model.eval()
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            total_test_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_test_loss = total_test_loss / len(test_loader)
    logger.info(f"Epoch: {epoch} / {train_settings.epochs}: train loss: {avg_train_loss:.4f}: test loss: {avg_test_loss:.4f}")

    # early stopping
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
    else:
        logger.info("Increasing patience counter")
        patience_counter += 1

    if patience_counter >= train_settings.patience:
        logger.info(f"Stopping early at epoch: {epoch}")
        logger.info(f"Best test loss : {best_test_loss:.4f}")
        break