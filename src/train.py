"""
Trains a PyTorch convolutional neural network mode to classify images.

Author: Alvaro Deras
Date: January 13, 2024
"""
import os
import torch
import model_builder, engine, data_setup, utils

from torchvision import transforms

NUM_EPOCHS = 15
HIDDEN_UNITS = 32
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, class_names = data_setup.setup_data()

def main():
    model = model_builder.WasteClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
    ).to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=LEARNING_RATE)

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

# Optional save
utils.save_model(model=model,
                 target_dir="../models",
                 model_name="trained_model")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
