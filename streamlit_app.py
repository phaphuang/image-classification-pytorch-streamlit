import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
    
    # Load labels
    with open("imagenet_classes.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Get the top 5 predictions
    _, indices = torch.topk(output, 5)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    # Convert indices to class labels
    predictions = [(labels[idx], probabilities[idx].item()) for idx in indices[0]]
    
    return predictions

# Streamlit app
def main():
    st.title("Image Classification with PyTorch and Streamlit")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        
        # Make predictions
        predictions = predict(uploaded_file)
        
        st.write("### Predictions:")
        for i, (label, probability) in enumerate(predictions):
            st.write(f"{i + 1}. {label}: {probability:.2f}%")

if __name__ == "__main__":
    main()