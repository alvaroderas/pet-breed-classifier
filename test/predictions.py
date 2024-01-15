"""
Test script for Waste Classifier model.

Author: Alvaro Deras
Date: January 14, 2024
"""
import torch, os
import numpy as np
import seaborn as sns
import model_builder
from torch import transforms, datasets
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm


def plot_predictions(model: torch.nn.Module, device: torch.device):
    """
    Displays a confusion matrix for the Waste Classifier model

    Parameter model: the model to have its predictions plotted
    Precondition: model is an instance of torch.nn.Module
    """
    data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    data_root = "../dataset"
    test_data = datasets.ImageFolder(
        root=os.path.join(data_root, "DATASET/TEST"),
        transform=data_transform,
    )

    test_indices = list(range(len(test_data)))

    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    shuffle_dataset = True
    if shuffle_dataset:
        np.random.shuffle(test_indices)

    test_split = 0.8
    test_size = int(np.floor(test_split * len(test_data)))
    test_indices, _ = test_indices[:test_size], test_indices[test_size:]

    test_sampler = SubsetRandomSampler(test_indices)

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=32,
        sampler=test_sampler,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    y_preds = []
    true_labels = []
    model.eval()

    threshold = 0.5

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions..."):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = (y_logit> threshold).long()
            y_preds.append(y_pred.cpu())
            true_labels.append(y.cpu())

    y_pred_tensor = torch.cat(y_preds)
    true_labels_tensor = torch.cat(true_labels)

    class_names = ["Organic", "Recyclable"]

    true_labels_np = true_labels_tensor.numpy()
    y_pred_np = y_pred_tensor.numpy()

    cm = confusion_matrix(true_labels_np, y_pred_np)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Waste Classifier Predictions")
    plt.show()

HIDDEN_UNITS = 32
MODEL_SAVE_PATH = "models/trained_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model = model_builder.WasteClassifier(
    input_shape=3,
    hidden_units=HIDDEN_UNITS
)

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

plot_predictions(model=loaded_model, device=device)