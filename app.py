import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from PIL import ImageOps

# Load the trained model
model = load_model('labeled.h5')

labels = [
    'R', 'U', 'I', 'N', 'G', 'Z', 'T', 'S', 'A', 'F', 'O', 'H', 
    'Blank', 'M', 'J', 'C', 'D', 'V', 'Q', 'X', 'E', 'B', 'K', 'L', 
    'Y', 'P', 'W'
]

# Description of the hand signs for educational purposes
sign_descriptions = {
    'R': "**R**: Cross your index and middle fingers while keeping the other fingers curled into a fist.",
    'U': "**U**: Extend your index and middle fingers straight up while curling the other fingers into a fist.",
    'I': "**I**: Hold up your pinky finger while keeping the other fingers curled into a fist.",
    'N': "**N**: Extend your index and middle fingers upright while curling the other fingers into a fist.",
    'G': "**G**: Extend your index finger straight and point it downward while curling the other fingers into a fist.",
    'Z': "**Z**: Trace the letter 'Z' in the air using your index finger.",
    'T': "**T**: Place your thumb between your index and middle fingers, forming a 'T' shape.",
    'S': "**S**: Curl your fingers into a fist with your thumb placed over the top of your fingers.",
    'A': "**A**: Form a fist with your thumb placed next to the side of your index finger.",
    'F': "**F**: Touch the tips of your thumb and index finger while curling the other fingers.",
    'O': "**O**: Form a circle with your fingers while keeping the other fingers curled.",
    'H': "**H**: Extend your index and middle fingers upright while keeping the other fingers curled into a fist.",
    'Blank': "**Blank**: Indicates no letter or gesture.",
    'M': "**M**: Curl your three middle fingers while keeping your pinky and index fingers bent in the fist.",
    'J': "**J**: Move your pinky finger in the shape of the letter 'J' while holding the other fingers curled.",
    'C': "**C**: Curl your fingers to form a 'C' shape with a gap between the tips of your fingers and palm.",
    'D': "**D**: Extend your index finger while curling the other fingers into a fist to form a 'D' shape.",
    'V': "**V**: Extend your index and middle fingers to form a V while curling the other fingers.",
    'Q': "**Q**: Extend your index and thumb to form a circle while curling the other fingers inward.",
    'X': "**X**: Curl your index finger into a hook shape while keeping the other fingers curled.",
    'E': "**E**: Curl your fingers toward your palm while keeping the thumb straight.",
    'B': "**B**: Open your hand, extending all fingers together and straight.",
    'K': "**K**: Extend your index and middle fingers while placing your thumb between them and curling the other fingers into a fist.",
    'L': "**L**: Extend your index finger and thumb to form an L shape while curling the other fingers into a fist.",
    'Y': "**Y**: Extend your thumb and pinky fingers while curling the other fingers into a fist.",
    'P': "**P**: Extend your index and middle fingers while keeping the other fingers curled, with the palm facing downward.",
    'W': "**W**: Extend your index, middle, and ring fingers to form a W shape while curling your pinky."
}


# Streamlit app title
st.title("ASL Alphabet Predictor")
st.write("Take a picture or upload an image to predict ASL signs and learn the corresponding hand sign.")

with st.expander("About This App"):
    st.markdown("""
        **ASL Alphabet Predictor** uses a pre-trained deep learning model to recognize American Sign Language (ASL) letters from images. 
        Here's how to use the app:
        
        - **Capture Image**: Use your camera to take a picture of your hand signing an ASL letter.
        - **Upload File**: Upload a JPEG image of an ASL sign.
        - The model will predict the corresponding letter from the image.
        
        **Notes**:
        - The image should be clear and show the ASL sign against a plain background for best results.
        - This app recognizes only the ASL alphabet letters, not words or gestures.

        **Model Details**:
        - **Model Architecture**: Custom CNN trained on labeled ASL images.
        - **Input Size**: 32x32 pixels, normalized to values between 0 and 1.
    """)

with st.expander("How to Sign the Alphabet"):
    st.markdown("""
        ## Learn How to Sign Each Letter in the ASL Alphabet
        
        The American Sign Language (ASL) alphabet consists of 26 unique hand shapes. Each hand shape corresponds to a letter of the English alphabet.

        **Sign Language Alphabet:**
        
        - **A**: The letter 'A' is signed with a fist, with the thumb placed next to the side of the index finger.
        - **B**: Open your hand and extend all fingers, keeping them together and straight.
        - **C**: Make a "C" shape by curling your fingers, leaving a gap between the tips of your fingers and the palm.
        - **D**: Form a "D" by extending your index finger and keeping the rest of your fingers curled into a fist.
        - **E**: Make a fist with your thumb and index finger touching, and the other fingers bent down.
        - **F**: Form an "O" with your thumb and index finger, and keep the other fingers curled.
        - **G**: Extend your index finger and thumb to form a "G" shape, while curling your other fingers into a fist.
        - **H**: Extend your index and middle fingers while keeping the other fingers curled into a fist.
        - **I**: Hold up your pinky finger while keeping the other fingers curled into a fist.
        - **J**: Make the sign for 'I' and move your pinky finger in the shape of the letter 'J'.
        - **K**: Extend your index and middle fingers while keeping your thumb between them, and curl the rest of your fingers into a fist.
        - **L**: Extend your index finger and thumb to form a right angle, with the other fingers curled into a fist.
        - **M**: Form an "M" by curling your three middle fingers while keeping your pinky and index fingers bent in the fist.
        - **N**: Similar to "M", but with your index and middle fingers extended instead of all three middle fingers.
        - **O**: Form an "O" shape by curling your fingers and thumb together, with the other fingers curled in.
        - **P**: Extend your index and middle fingers while keeping the other fingers curled, with your palm facing downward.
        - **Q**: Similar to "G", but with your thumb and index finger pointing downwards.
        - **R**: Cross your index and middle fingers while keeping the other fingers curled into a fist.
        - **S**: Curl your fingers into a fist with your thumb placed over the top of your fingers.
        - **T**: Form a "T" by placing your thumb between the index and middle fingers.
        - **U**: Extend your index and middle fingers straight up while curling the other fingers into a fist.
        - **V**: Extend your index and middle fingers to form a "V" while keeping the other fingers curled.
        - **W**: Extend your index, middle, and ring fingers to form a "W" shape while curling your pinky.
        - **X**: Bend your index finger in a "hook" shape and keep the rest of the fingers curled into a fist.
        - **Y**: Extend your thumb and pinky finger while curling the other fingers into a fist.
        - **Z**: Make a "Z" shape in the air by using your index finger to trace the letter in the air.
        
        **Practice Tip**: 
        To improve your ASL skills, try practicing each letter until you can sign it comfortably. Once you've mastered the individual letters, you can try spelling out words or even forming basic sentences.

    """)


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

# Enable camera checkbox
enable_camera = st.checkbox("Enable camera")
picture = None

# Show camera input only when the checkbox is enabled
if enable_camera:
    # Camera input
    picture = st.camera_input("Take a picture")

# File upload input
uploaded_file = st.file_uploader("Upload a JPEG image", type=["jpg", "jpeg"])

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
    st.write(f"### You have signed the letter: {predicted_label} !")
    st.write(f"#### How to sign '{predicted_label}':")
    st.write(sign_descriptions.get(predicted_label, "Description not available"))

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
    st.write(f"### You have signed the letter: {predicted_label} !")
    st.write(f"#### How to sign '{predicted_label}':")
    st.write(sign_descriptions.get(predicted_label, "Description not available"))
