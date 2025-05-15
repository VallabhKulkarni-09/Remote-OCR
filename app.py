import streamlit as st
import ollama
from PIL import Image
import io
import base64
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Remote OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.markdown("""
    # <img src="data:image/png;base64,{}" width="50" style="vertical-align: -12px;"> Gemma-3 OCR
""".format(base64.b64encode(open("./assets/gemma3.png", "rb").read()).decode()), unsafe_allow_html=True)

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using Gemma-3 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# Camera controls in sidebar
with st.sidebar:
    st.header("Camera Capture")
    
    # Initialize camera capture on first run
    if 'camera' not in st.session_state:
        st.session_state.camera = cv2.VideoCapture(0)
        st.session_state.captured_image = None

    # Display camera feed
    FRAME_WINDOW = st.image([])
    
    def capture_frame():
        ret, frame = st.session_state.camera.read()
        if ret:
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.captured_image = frame
            return frame
        return None

    # Camera preview
    if st.session_state.camera.isOpened():
        if st.button("Capture 📸"):
            frame = capture_frame()
            if frame is not None:
                st.session_state.captured_image = frame
                st.success("Image captured!")
            else:
                st.error("Failed to capture image")
        
        # Show live preview until image is captured
        if 'captured_image' not in st.session_state or st.session_state.captured_image is None:
            while True:
                frame = capture_frame()
                if frame is not None:
                    FRAME_WINDOW.image(frame)
                else:
                    break
        else:
            # Show captured image
            FRAME_WINDOW.image(st.session_state.captured_image)
            
            # Extract text button
            if st.button("Extract Text 🔍", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Convert numpy array to bytes
                        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_RGB2BGR))
                        io_buf = io.BytesIO(buffer)
                        
                        response = ollama.chat(
                            model='gemma3:12b',
                            messages=[{
                                'role': 'user',
                                'content': """Analyze the text in the provided image. Extract all readable content
                                            and present it in a structured Markdown format that is clear, concise, 
                                            and well-organized. Ensure proper formatting (e.g., headings, lists, or
                                            code blocks) as necessary to represent the content effectively.""",
                                'images': [io_buf.getvalue()]
                            }]
                        )
                        st.session_state['ocr_result'] = response.message.content
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            
            # Add retake button
            if st.button("Retake 🔄"):
                st.session_state.captured_image = None
                st.rerun()
    else:
        st.error("Could not access camera. Please make sure your camera is connected and permissions are granted.")

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown(st.session_state['ocr_result'])
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Gemma-3 Vision Model | [Report an Issue](https://github.com/patchy631/ai-engineering-hub/issues)")