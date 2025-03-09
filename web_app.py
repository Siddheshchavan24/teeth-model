import streamlit as st
import numpy as np
import gdown
from PIL import Image
from tensorflow.keras.models import load_model

# Google Drive file ID
file_id = "1NWkTQVANbycVLP9Gso11H1vtLiieXQ6-"
model_path = "teeth_model.h5"

# Function to download and load model
@st.cache_resource
def load_teeth_model():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    model = load_model(model_path)
    return model

# Load the model
model = load_teeth_model()

# Define class names
class_names = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Tooth Discoloration', 'Ulcers']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model's expected input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🦷 Teeth Disease Detection")
uploaded_file = st.file_uploader("📷 Upload an image of teeth", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)
    
    # Preprocess image and make prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Display prediction result
    st.write(f"### 🏥 Predicted Class: **{class_names[predicted_class]}**")
