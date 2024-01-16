"""
Trains a PyTorch convolutional neural network mode to classify images.

Author: Alvaro Deras
Date: January 13, 2024
"""
import os, torch, requests, zipfile
import model_builder, engine, data_setup

from pathlib import Path

from torchvision import transforms

def main():
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 32

    data_root = "../data/waste"

    data_path = Path("../data/")
    image_path = data_path / "waste"

    if image_path.is_dir():
        print(f"{image_path} directory exists. Skipping directory creation.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        zip_file_path = data_path/"archive.zip"
        with open(zip_file_path, "wb") as f:
            request = requests.get("http://dl.dropboxusercontent.com/scl/fi/3q0nbtsk43qixqtsf9j19/archive.zip?rlkey=8b14v0cyj1kyctaebv05yhr7m&dl=0")
            print("Downloading waste dataset...")
            f.write(request.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            print("Unzipping waste classification data...")
            zip_ref.extractall(image_path)

        os.remove(zip_file_path)

    NUM_EPOCHS = 25
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.0003
    MODEL_NAME = "trained_model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_dataloader, test_dataloader, class_names = data_setup.setup_data()

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
                    model_name=MODEL_NAME)

if __name__ == '__main__':
    main()
