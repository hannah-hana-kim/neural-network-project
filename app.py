import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('test5.h5')

# Define the labels for predictions
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Streamlit app title
st.title("ASL Alphabet Predictor")
st.write("Take a picture or upload an image to predict ASL signs.")

# Enable camera checkbox
enable_camera = st.checkbox("Enable camera")

# Camera input
picture = st.camera_input("Take a picture", disabled=not enable_camera)

# File upload input
uploaded_file = st.file_uploader("Upload a JPEG image", type=["jpg", "jpeg"])

# Function to preprocess image
from PIL import ImageOps

# Function to preprocess image with flipping
def preprocess_image(image, flip=False):
    # Convert image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Optionally flip the image horizontally
    if flip:
        image = ImageOps.mirror(image)
    # Resize the image to the input shape of the model
    image = image.resize((32, 32))
    # Convert the image to a NumPy array
    img_array = np.array(image)
    # Normalize the pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Process camera picture
if picture:
    # Convert the uploaded picture to a PIL Image
    image = Image.open(picture)

    # Display the picture
    st.image(image, caption="Captured Image", width=250)

    # Preprocess the image with flipping enabled
    img_array = preprocess_image(image, flip=True)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_class_index]

    # Display the prediction
    st.write(f"### Predicted Class: {predicted_label}")


# Process uploaded JPEG file
if uploaded_file:
    # Convert the uploaded file to a PIL Image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", width=250)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_class_index]

    # Display the prediction
    st.write(f"### Predicted Class: {predicted_label}")
