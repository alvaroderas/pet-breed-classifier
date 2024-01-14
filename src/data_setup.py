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

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

data_root = "../Data"

train_data = datasets.OxfordIIITPet(
    root=data_root,
    split="trainval",
    download=True,
    transform=data_transform,
    target_transform=None
)

test_data = datasets.OxfordIIITPet(
    root=data_root,
    split="test",
    download=True,
    transform=data_transform,
    target_transform=None
)

def setup_data(train_data: str=train_data,
                      test_data: str=test_data,
                      batch_size: int=BATCH_SIZE,
                      num_workers: int=NUM_WORKERS):
    """
    Returns a tuple of dataloaders and class names.

    Takes in training and test datasets and converts them into PyTorch DataLoaders.

    Parameter train_data: the training dataset
    Precondition: train_data is an instance of torch.utils.data.Dataset

    Parameter test_data: the testing dataset
    Precondition: test_data is an instance of torch.utils.data.Dataset

    Parameter batch_size: the number of samples per batch in each DataLoader
    Precondition: batch_size is an int

    Parameter num_workers: the number of workers per DataLoader
    Precondition: num_workers is an int
    """
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names