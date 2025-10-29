import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# --- Configuration ---
MODEL_PATH = 'mnist_model.keras'
IMG_WIDTH = 28
IMG_HEIGHT = 28

st.set_page_config(layout="wide")

# --- Initialize session state for canvas key ---
# This is used to force a re-render of the canvas when clearing
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_0"

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)

# --- Image Pre-processing Functions ---
def preprocess_canvas_image(img_data):
    """
    Pre-processes the image data from the drawable canvas.
    Converts 280x280 RGBA to 28x28 grayscale.
    """
    # Convert canvas data to PIL Image
    img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28 (model's expected input size)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Reshape for the model: (1, 28, 28)
    # The model's first layer handles adding the "channel" dimension
    return np.expand_dims(img_array, axis=0)

def preprocess_uploaded_image(img):
    """
    Pre-processes an image file uploaded by the user.
    Converts to grayscale, inverts colors, resizes, and normalizes.
    """
    # Convert to grayscale
    img = img.convert('L')
    
    # Invert colors (MNIST is white digit on black background)
    img = ImageOps.invert(img)
    
    # Resize to 28x28
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Reshape for the model: (1, 28, 28)
    return np.expand_dims(img_array, axis=0)

# --- Prediction Function ---
def predict(image_array):
    """Runs prediction and returns the digit and confidence."""
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_digit, confidence

# --- Streamlit UI ---
st.title(" MNIST Handwritten Digit Recognizer")
st.markdown("Draw a digit or upload an image to get a prediction from a CNN model.")

tab1, tab2 = st.tabs(["Draw a Digit", "Upload an Image (Drag & Drop)"])

# --- Tab 1: Draw a Digit (Recommended) ---
with tab1:
    st.subheader("Draw your digit here (best results):")
    st.markdown("Draw a single white digit on the black canvas. The model is trained on this style.")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Create a 280x280 canvas (10x the model size for easier drawing)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Fixed fill color
            stroke_width=20,  # Thicker stroke for better visibility
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            # Use the key from session state
            key=st.session_state.canvas_key,
        )

    with col2:
        st.subheader("Prediction:")
        predict_button = st.button("Predict Drawing", key="predict_canvas")
        
        # Add the clear button
        if st.button("Clear Drawing", key="clear_canvas"):
            # Increment the key counter in session state to force a rerender
            current_key_index = int(st.session_state.canvas_key.split('_')[-1])
            st.session_state.canvas_key = f"canvas_{current_key_index + 1}"
            # Force an immediate rerun to show the cleared canvas
            st.rerun()

        if predict_button:
            if canvas_result.image_data is not None:
                # Get the image data from the canvas
                img_data = canvas_result.image_data
                
                # Pre-process the image
                processed_img = preprocess_canvas_image(img_data)
                
                # Make prediction
                digit, confidence = predict(processed_img)
                
                st.markdown(f"## Predicted Digit: `{digit}`")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            else:
                st.warning("Please draw a digit first.")

# --- Tab 2: Upload an Image ---
with tab2:
    st.subheader("Upload your image (supports drag & drop):")
    st.warning(
        "**Note:** This model was trained on simple white-on-black, centered digits. "
        "Real-world images with backgrounds or different styling may not predict accurately."
    )
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        try:
            # Open and display the uploaded image
            img = Image.open(uploaded_file)
            with col1:
                st.markdown("**Your Upload:**")
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Pre-process and predict
            processed_img = preprocess_uploaded_image(img)
            digit, confidence = predict(processed_img)

            with col2:
                st.markdown("**Prediction:**")
                st.markdown(f"## Predicted Digit: `{digit}`")
                st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        except Exception as e:
            st.error(f"Error processing image: {e}")