# Image Filtering and Classification with ResNet-50 and Transfer Learning

## Project Overview

This project implements an image classification pipeline using the ResNet-50 architecture with transfer learning. It also includes image filtering techniques to preprocess and enhance images before classification. The goal is to build a robust model that can classify images into predefined categories with high accuracy.

## Features

- **Image Filtering:** Preprocessing with techniques such as Gaussian filtering and image denoising.
- **Transfer Learning:** Fine-tuning a pre-trained ResNet-50 model on a custom dataset.
- **Model Training:** Training the model with custom data and evaluating its performance.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1 score.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using:

```bash
pip install torch torchvision numpy matplotlib scikit-learn

```

# Setup and Installation

## Clone the Repository:

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/NanGyeThote/PictoNet.git
cd PictoNet
```

## Prepare Your Dataset

Ensure your dataset is organized in the following structure:

```bash
data/
    train/
        cat/img1.jpg
        dog/img1.jpg
        horse/img1.jpg
    val/
        cat/img2.jpg
        dog/img2.jpg
        horse/img2.jpg
```
## Run the Filtering and Classification Script

### Execute the following command to run the filtering and classification script:

```bash
python main.py
```
Make sure to update the dataset paths and any configuration settings as needed in the config.py file before running the script.

## Usage
1. **Filtering Images**:

The filtering script applies techniques like Gaussian filtering and denoising.
The filtered images are saved to a specified directory.
Training the Model:

The training script uses transfer learning with a pre-trained ResNet-50 model.
Modify the hyperparameters and paths as needed in the config.py file.
Evaluating the Model:

After training, the model is evaluated on the validation set.
Evaluation metrics (accuracy, precision, recall, F1 score) are reported.
