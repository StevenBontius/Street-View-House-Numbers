from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from loguru import logger

from src.settings import TrainingSettings


def get_dataloaders(settings: TrainingSettings) -> tuple[DataLoader, DataLoader]:
    data_dir = Path.home() / ".cache" / "SVHN"
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {data_dir}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.mean, std=settings.std),
        ]
    )

    train_data = datasets.SVHN(
        root=data_dir, split="train", transform=transform, download=True
    )
    test_data = datasets.SVHN(
        root=data_dir, split="test", transform=transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_data, batch_size=settings.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=settings.batch_size, shuffle=False
    )

    return (train_loader, test_loader)
