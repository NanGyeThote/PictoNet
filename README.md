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

#Setup and Installation

##Clone the Repository:

```bash
git clone https://github.com/NanGyeThote/PictoNet.git
cd PictoNet
```

#Prepare Your Dataset:

##Ensure your dataset is organized in the following structure:

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
#Run the Filtering and Classification Script:

```bash
python main.py
```
