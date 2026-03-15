from src.settings import TrainingSettings, ModelSettings
from src.data import get_dataloaders
from src.model import SimpleConvModel
from src.utils import get_device

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from typing import Any
from loguru import logger
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter

NUM_SAMPLES = 20


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    """Train the model for one epoch

    Returns:
        avg_loss: average loss over all batches
        avg_accuracy: average accuracy over all batches
    """
    model.train()

    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
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
        accuracy(y_hat, y)

    avg_loss = total_loss / num_batches
    avg_accuracy = accuracy.compute().item()

    return avg_loss, avg_accuracy


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    """Tests the model for one epoch

    Returns:
        avg_loss: average loss over all batches
        avg_accuracy: average accuracy over all batches
    """
    model.eval()

    total_loss = 0
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    num_batches = len(dataloader)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()
            accuracy(y_hat, y)

    avg_loss = total_loss / num_batches
    avg_accuracy = accuracy.compute().item()

    return avg_loss, avg_accuracy


def train(config: dict[str, Any]) -> None:
    if ray.is_initialized():
        logger.disable("")
        
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
    best_test_accuracy = 0
    patience_counter = 0

    for epoch in range(train_settings.epochs):
        train_loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_classes=model_settings.num_classes,
        )
        test_loss, test_accuracy = test_epoch(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=model_settings.num_classes,
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
            best_test_accuracy = test_accuracy
            patience_counter = 0
        else:
            logger.info("Increasing patience counter")
            patience_counter += 1

        if patience_counter >= train_settings.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(
                f"Best test loss: {best_test_loss:.4f} and accuracy: {best_test_accuracy:.4f}"
            )
            break


if __name__ == "__main__":
    ray.init()

    tune_dir = Path(__file__).parent / "logs" / "ray"
    tune_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ray tune directory: {tune_dir}")

    config = {
        "num_layers": tune.choice([2, 3, 4]),
        "num_filters": tune.choice([16, 32, 64, 128]),
        "num_hidden_units": tune.choice([64, 128, 256, 512]),
        "batch_size": 32,
        "learning_rate": 0.001,
    }

    tuner = tune.Tuner(
        train,
        param_space=config,
        run_config=ray.air.RunConfig(
            storage_path=str(tune_dir),
            name="svhn_experiment",
            progress_reporter=CLIReporter(
                metric_columns=["train_loss", "test_loss", "test_accuracy"]
            ),
        ),
        tune_config=tune.TuneConfig(
            num_samples=NUM_SAMPLES,
            metric="test_loss",
            mode="min",
        ),
    )

    results = tuner.fit()

    df = results.get_dataframe()
    logger.info(f"Results shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    df.to_csv("logs/results.csv", index=False)

    ray.shutdown()
