"""
Contains functions for training and testing the Waste Classifier model.

Author: Alvaro Deras
Date: January 13, 2024
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Returns a tuple of training loss and training accuracy metrics.

    Trains a given PyTorch model for an epoch by converting model to training
    mode and proceeds to undergo training steps.

    Parameter model: the model to be trained
    Precondition: model is an instance of torch.nn.Module

    Parameter dataloader: the dataloader iterable for the dataset
    Precondition: dataloader is an instance of torch.utils.data.DataLoader

    Parameter loss_function: the loss function used for optimization
    Precondition: loss_function is an instance of torch.nn.Module

    Parameter optimizer: the optimizer used for updating model parameters
    Precondition: optimizer is an instance of torch.optim.Optimizer

    Parameter device: the target device to compute on (e.g. "cuda" or "cpu")
    Precondition: device is a valid torch.device
    """
    model.train()

    train_loss = 0
    train_acc = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y.view(-1).long()
        y = y.to(device)

        y_logits = model(X).squeeze().to(device)

        loss = loss_function(y_logits, y.type(torch.float32))
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        y_pred_class = torch.round(torch.sigmoid(y_logits))
        train_acc += (y_pred_class == y).sum().item()/len(y_pred_class)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Returns a tuple of testing loss and testing accuracy metrics.

    Tests a given PyTorch model for an epoch by converting model to evaluation
    mode and undergoes testing steps.

    Parameter model: the model to be trained
    Precondition: model is an instance of torch.nn.Module

    Parameter dataloader: the dataloader iterable for the dataset
    Precondition: dataloader is an instance of torch.utils.data.DataLoader

    Parameter loss_function: the loss function used for optimization
    Precondition: loss_function is an instance of torch.nn.Module

    Parameter device: the target device to compute on (e.g. "cuda" or "cpu")
    Precondition: device is a valid torch.device
    """
    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            test_pred_logits = model(X).squeeze()

            loss = loss_function(test_pred_logits, y.float())
            test_loss += loss.item()

            test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """
    Returns a dictionary of training and testing metrics.

    Trains and tests a PyTorch model by combining train_step() and test_step()
    functions for a given number of epochs.

    Parameter model: the model to be trained
    Precondition: model is an instance of torch.nn.Module

    Parameter train_dataloader: the dataloader iterable for the dataset
    Precondition: train_dataloader is an instance of torch.utils.data.DataLoader

    Parameter test_dataloader: the dataloader iterable for the dataset
    Precondition: test_dataloader is an instance of torch.utils.data.DataLoader

    Parameter loss_function: the loss function used for optimization
    Precondition: loss_function is an instance of torch.nn.Module

    Parameter optimizer: the optimizer used for updating model parameters
    Precondition: optimizer is an instance of torch.optim.Optimizer

    Parameter epochs: the amount of epochs to undergo training
    Precondition: epochs is an int greater or equal to 1

    Parameter device: the target device to compute on (e.g. "cuda" or "cpu")
    Precondition: device is a valid torch.device
    """
    metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc":[]}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_function=loss_function,
                                        device=device)
        print(
            f"\nEpoch: {epoch+1} |"
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} |"
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f}"
        )

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)

    return metrics