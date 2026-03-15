from src.settings import TrainingSettings, ModelSettings
from src.data import get_dataloaders
from src.model import SimpleConvModel
from src.utils import get_device

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Any
from loguru import logger
from ray import tune


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch

    Returns:
        avg_loss: average loss over all batches
        avg_accuracy: average accuracy over all batches
    """
    model.train()

    total_loss = 0
    total_accuracy = 0
    num_batches = len(dataloader)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Tests the model for one epoch

    Returns:
        avg_loss: average loss over all batches
        avg_accuracy: average accuracy over all batches
    """
    model.eval()

    total_loss = 0
    total_accuracy = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def train(config: dict[str, Any]) -> None:
    model_settings = ModelSettings(
        num_layers=config["num_layers"],
        num_filters=config["num_filters"],
        num_hidden_units=config["num_hidden_units"],
    )
    train_settings = TrainingSettings(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    train_loader, test_loader = get_dataloaders(train_settings)

    device = get_device()
    model = SimpleConvModel(model_settings).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings.learning_rate)

    best_test_loss = float("inf")
    patience_counter = 0

    for epoch in range(train_settings.epochs):
        train_loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_accuracy = test_epoch(
            model=model, dataloader=test_loader, loss_fn=loss_fn, device=device
        )

        logger.info(
            f"Epoch: {epoch} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f} Accuracy: {test_accuracy:.4f}"
        )

        tune.report(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            logger.info("Increasing patience counter")
            patience_counter += 1

        if patience_counter >= train_settings.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(f"Best test loss: {best_test_loss:.4f}")
            break


if __name__ == "__main__":
    config = {
        "num_layers": 3,
        "num_filters": 32,
        "num_hidden_units": 128,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    train(config)
