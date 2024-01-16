# Waste Classifier

A convolutional neural network for waste classification based on whether it is organic or recyclable.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Waste Classifier is a deep learning model built for the purpose of waste classification. It utilizes a custom convolutional neural network (CNN) titled WasteClassifier to classify waste into two categories: organic and recyclable. This is a simple deep learning personal project using PyTorch that aims to contribute to waste management initiatives by automating the sorting process, impacting the environment for the better.

## Features

- Convolutional Neural Network for waste classification.
- Binary classification: Organic and Recyclable.
- Model training script for customization.
- Model testing script for predictions with a trained model

## Screenshots

An example of prediction results from a trained model, a 90% accuracy:
![Predictions](imgs/wastepredictions90.png)

## Installation

To get started, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/waste-classifier.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```


## Model Training
1. Edit any constants to experiment with training.
2. Run the model training script:

    ```bash
    python train_model.py
    ```
## Usage

To use the Waste Classifier to visualize waste classification predictions, run the script:

```bash
python predictions.py
```

## License
This project is licensed under the [MIT License](LICENSE).
