import numpy as np
import cv2
import streamlit as st
from detector import detect_text, net
from utils import draw_results

st.title('Text Detection and Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Detecting text...")
    
    # Perform text detection
    image, results = detect_text(image, net)
    
    # Draw results
    output_image = draw_results(image, results)
    
    st.image(output_image, caption='Processed Image', use_column_width=True)
    
    st.write("Extracted Text:")
    for ((startX, startY, endX, endY), text) in results:
        st.write(f"Detected text: {text.strip()}")