import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd # Import pandas here to ensure it's loaded early if needed
import os # Import OS for checking directory/file existence

# --- 1. STREAMLIT CONFIGURATION (MUST BE THE FIRST COMMAND) ---
st.set_page_config(
    page_title="CNN Digit Recognizer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL CONFIGURATION ---
# NOTE: Using the modern Keras format (.keras) is the most robust and recommended format.
MODEL_PATH_KERAS = 'cnn_digit_model.keras'
MODEL_PATH_H5 = 'cnn_digit_model.h5' 

CANVAS_SIZE = 200
IMAGE_SIZE = 28 # MNIST standard size
MODEL_INPUT_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE, 1)

# --- 3. MODEL LOADING (Cached for Efficiency) ---
@st.cache_resource
def load_cnn_model():
    """Attempts to load the model, checking for .keras first, then .h5."""
    
    model_path_to_load = None

    # Priority 1: Check for the modern .keras format (RECOMMENDED)
    if os.path.exists(MODEL_PATH_KERAS):
        model_path_to_load = MODEL_PATH_KERAS
    # Priority 2: Fallback to the legacy .h5 format
    elif os.path.exists(MODEL_PATH_H5):
        model_path_to_load = MODEL_PATH_H5
    
    if model_path_to_load is None:
        st.error("Error: Model file not found.")
        st.info(f"Please save your trained model in the app folder using: `model.save('{MODEL_PATH_KERAS}')` or `model.save('{MODEL_PATH_H5}')`.")
        return None

    # Attempt to load the model
    try:
        model = load_model(model_path_to_load)
        return model
    except Exception as e:
        # This catches file signature errors (the persistent problem)
        st.error(f"Error loading model from '{model_path_to_load}': {e}")
        st.warning(
            f"**CRITICAL ACTION REQUIRED:** The model file is structurally invalid. "
            f"You MUST go back to your training notebook and execute the following line EXACTLY: "
            f"`model.save('{MODEL_PATH_KERAS}')`. "
            f"Then, check that the resulting file size is large (10+ MB)."
        )
        return None

# Load the model once
model = load_cnn_model()

# --- 4. STREAMLIT UI SETUP (Run only if model loaded successfully) ---
st.title("✍️ Interactive CNN Digit Recognizer")
st.markdown("Draw a single digit (0-9) on the canvas below, then click 'Predict'!")


if model is not None:
    # --- Drawing Canvas ---
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="#000000",  # Background color (black)
        stroke_width=15,       # Width of the drawing stroke
        stroke_color="#FFFFFF",# Color of the drawing stroke (white, like MNIST)
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.markdown("---")
    
    # --- Prediction Logic ---
    if st.button('🚀 Predict Digit', use_container_width=True):
        if canvas_result.image_data is not None:
            # 1. Convert drawn data to PIL Image
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            
            # 2. Convert to grayscale and resize to 28x28
            img_gs = img.convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))

            # 3. Convert to numpy array and normalize
            img_array = np.array(img_gs)

            # Invert colors: 255 - pixel_value ensures black becomes 255 (white) and white becomes 0 (black).
            # Then we divide by 255.0 for final normalization to 0-1.
            img_array = 255 - img_array
            img_array = img_array / 255.0

            # 4. Reshape for the model: (1, 28, 28, 1)
            input_tensor = img_array.reshape(MODEL_INPUT_SHAPE)

            # 5. Make prediction
            prediction = model.predict(input_tensor)
            
            # Get the predicted digit (index of the highest probability)
            predicted_digit = np.argmax(prediction)
            
            # Get the confidence score
            confidence = prediction[0][predicted_digit] * 100

            st.success(f"## Predicted Digit: **{predicted_digit}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
            
            st.markdown("---")

            # --- Display all probabilities ---
            st.subheader("Model Confidence Scores")
            
            # Prepare data for a bar chart
            prob_df = pd.DataFrame({
                'Digit': list(range(10)),
                'Probability': prediction[0]
            })
            
            st.bar_chart(prob_df.set_index('Digit'))

            # --- Display Preprocessed Image ---
            with st.expander("Show Model Input Image"):
                # Display the preprocessed 28x28 image for verification
                display_img_array = (input_tensor[0, :, :, 0] * 255).astype(np.uint8)
                display_image = Image.fromarray(display_img_array, mode='L')
                st.image(
                    display_image, 
                    caption="Preprocessed 28x28 Image (Model Input)", 
                    use_container_width=True # Changed from use_column_width
                )
                st.write("The model saw the image above. It is white on black, normalized, and sized 28x28.")
        
        else:
            st.warning("Please draw a digit on the canvas first!")
else:
    # If model is None, display the final placeholder message
    st.warning("Application requires a correctly saved model file (.keras or .h5) to proceed.")
