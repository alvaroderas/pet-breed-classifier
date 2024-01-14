"""
Contains functionality for setting up image classifcation data through
the creation of PyTorch DataLoaders.

Author: Alvaro Deras
Date: January 13, 2024
"""
import os, torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

data_root = "../dataset"


def setup_data(data_root: str="../dataset",
               batch_size: int=BATCH_SIZE,
               num_workers: int=NUM_WORKERS):
    """
    Returns a tuple of dataloaders and class names.

    Takes in data root, creates datasets, turns them into PyTorch DataLoaders.

    Parameter data_root: the root directory where the dataset is located
    Precondition: data_root is a valid path

    Parameter batch_size: the number of samples per batch in each DataLoader
    Precondition: batch_size is an int

    Parameter num_workers: the number of workers per DataLoader
    Precondition: num_workers is an int
    """
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(
        root=os.path.join(data_root, "DATASET/TRAIN"),
        transform=data_transform,
    )

    test_data = datasets.ImageFolder(
        root=os.path.join(data_root, "DATASET/TEST"),
        transform=data_transform,
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names