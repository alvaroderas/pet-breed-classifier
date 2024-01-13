"""
Contains any necessary utility functions for a PyTorch model.

Author: Alvaro Deras
Date: January 13, 2024
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_directory: str,
               model_name: str):
    """
    Saves a PyTorch model to target directory.
    
    Parameter model: the model to save
    Precondition: model is an instance of torch.nn.Module

    Parameter target_directory: the directory for the model to be saved in
    Precondition: target_directory is a string

    Parameter model_name: the filename for the model
    Precondition:  model_name is a string that ends in '.pth' or '.pt'
    """
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'invalid model name'

    directory_path = Path(target_directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    model_save_path = directory_path / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    