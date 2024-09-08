import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import models, transforms
import requests
import io

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define file size limit (in bytes)
FILE_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB

def no_filter(image):
    """Return the image as is (no filtering)."""
    return image

def filter_image(image, method):
    """Apply the selected filter to the image."""
    if method == 'Gaussian Blur':
        filtered_image = image.filter(ImageFilter.GaussianBlur(radius=2))
    elif method == 'Median Filter':
        filtered_image = image.filter(ImageFilter.MedianFilter(size=5))
    elif method == 'Bilateral Filter':
        # PIL does not support bilateral filtering; use Gaussian Blur as an alternative
        filtered_image = image.filter(ImageFilter.GaussianBlur(radius=2))
    else:
        filtered_image = image
    
    return filtered_image

def classify_image(image):
    """Classify the image using a pre-trained ResNet model."""
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def get_class_names():
    """Retrieve class names from the ImageNet labels URL."""
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return response.json()

def main():
    st.title('PictoNet: Image Filtering and Classification :rocket:')

    # Create a menu
    menu = ["Home", "Filter & Classify"]
    choice = st.sidebar.selectbox("Select an option", menu)
    
    if choice == "Home":
        st.subheader("Welcome to PictoNet")
        st.write("Upload an image, select a filtering method, and view the classification results.")
        st.write("""
        # About This App

        This Streamlit app allows you to upload an image, apply various denoising methods, 
        and classify the denoised image using a pre-trained ResNet model. You can choose from 
        Gaussian Blur, Median Filtering, or Bilateral Filtering to denoise the image before 
        performing classification.

        **Denoising Methods:**
        - **Gaussian Blur:** Applies a Gaussian filter to reduce noise.
        - **Median Filtering:** Replaces each pixel with the median of neighboring pixels.
        - **Bilateral Filtering:** (Placeholder) Uses Gaussian blur as a proxy.
        - **No Filter:** No filtering applied.

        **Classification:**
        The denoised image is classified using a ResNet-50 model trained on the ImageNet dataset.
        But **Please upload images within the classes of animals and clothes. Otherwise the dataset may be out of ranges.**

        **How to Use:**
        1. Upload an image. ( Not more than 5 MB )
        2. Select a denoising method.
        3. View the denoised image and classification results.
        """)

        # Add rotating rocket emoji
        st.subheader("Rotating Rocket Emoji")
        rocket_animation_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @keyframes rotate {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

                .rotating-rocket {
                    display: inline-block;
                    font-size: 50px;
                    animation: rotate 2s linear infinite;
                }
            </style>
        </head>
        <body>
            <div class="rotating-rocket">🚀</div>
        </body>
        </html>
        """
        st.components.v1.html(rocket_animation_html, height=100)

    elif choice == "Filter & Classify":
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Check file size
            file_size = uploaded_file.size
            if file_size > FILE_SIZE_LIMIT:
                st.warning(f"The uploaded file is too large. Please upload a file smaller than {FILE_SIZE_LIMIT / (1024 * 1024)} MB.")
            else:
                # Open and display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Select filtering method
                filter_method = st.selectbox('Choose a filtering method:', 
                                            ['None', 'No Filter', 'Gaussian Blur', 'Median Filter', 'Bilateral Filter'])
                
                if filter_method == 'None':
                    st.write("Please select a valid filtering method to proceed with classification.")
                else:
                    # Apply the selected filter or no filter
                    if filter_method == 'No Filter':
                        filtered_image = no_filter(image)
                    else:
                        filtered_image = filter_image(image, filter_method)
                    
                    # Display the filtered image
                    st.image(filtered_image, caption=f'Filtered Image ({filter_method})', use_column_width=True)
                    
                    # Classify the filtered image
                    class_id = classify_image(filtered_image)
                    
                    # Retrieve class names from the URL
                    class_names = get_class_names()
                    
                    # Ensure class_id is within the valid range
                    if class_id < len(class_names):
                        st.write(f'Class Name: {class_names[class_id]}')
                    else:
                        st.write('Class Name: Unknown (ID out of range)')

if __name__ == "__main__":
    main()
