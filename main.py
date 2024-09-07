import streamlit as st
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import requests
import matplotlib.pyplot as plt

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define denoising functions
def denoise_gaussian(image):
    img_array = np.array(image)
    img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    return Image.fromarray(img_array)

def denoise_median(image):
    img_array = np.array(image)
    img_array = cv2.medianBlur(img_array, 5)
    return Image.fromarray(img_array)

def denoise_bilateral(image):
    img_array = np.array(image)
    img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    return Image.fromarray(img_array)

# Function to classify image
def classify_image(image, class_labels):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    class_id = predicted.item()
    class_label = class_labels[class_id] if class_id < len(class_labels) else "Unknown"
    return class_id, class_label

# Function to download class labels
def download_class_labels():
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Load class labels
class_labels = download_class_labels()

# Streamlit app
st.title('Image Denoising and Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)
    
    # Denoising method selection
    denoise_method = st.selectbox(
        'Choose a denoising method',
        ['Gaussian Blur', 'Median Filtering', 'Bilateral Filtering']
    )
    
    # Apply the selected denoising method
    if denoise_method == 'Gaussian Blur':
        denoised_image = denoise_gaussian(image)
    elif denoise_method == 'Median Filtering':
        denoised_image = denoise_median(image)
    elif denoise_method == 'Bilateral Filtering':
        denoised_image = denoise_bilateral(image)
    
    # Display the denoised image
    st.image(denoised_image, caption='Denoised Image', use_column_width=True)
    
    # Classification
    class_id, class_name = classify_image(denoised_image, class_labels)
    
    # Show classification results
    st.write(f'Predicted class ID: {class_id}')
    st.write(f'Predicted class name: {class_name}')
    
    # Allow users to download the denoised image
    buffer = io.BytesIO()
    denoised_image.save(buffer, format="JPEG")
    buffer.seek(0)
    st.download_button(
        label="Download Denoised Image",
        data=buffer,
        file_name="denoised_image.jpg",
        mime="image/jpeg"
    )
