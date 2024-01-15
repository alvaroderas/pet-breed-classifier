import os
import requests
import zipfile
from pathlib import Path

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

data_root = "../data/waste"

data_path = Path("../data/")
image_path = data_path / "waste"

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

zip_file_path = data_path / "archive.zip"

if not zip_file_path.is_file():
    with open(zip_file_path, "wb") as f:
        request = requests.get("http://dl.dropboxusercontent.com/scl/fi/3q0nbtsk43qixqtsf9j19/archive.zip?rlkey=8b14v0cyj1kyctaebv05yhr7m&dl=0")
        print("Downloading waste dataset...")
        f.write(request.content)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        print("Unzipping waste classification data...")
        zip_ref.extractall(image_path)

    os.remove(zip_file_path)
else:
    print(f"Zip file {zip_file_path} already exists. Skipping download and extraction.")

def setup_data(data_root: str=data_root,
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