import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Real-time Remote Edge Detection")
    
    try:
        # Create a video capture object
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the camera. Please make sure your camera is connected and not in use by another application.")
            return
        # Create a placeholder for video frames
        frame_placeholder = st.empty()
        
        # Set fixed parameters for edge detection
        threshold1 = 100
        threshold2 = 200
        blur_kernel = 5
        stop_button = st.button("Stop")
        
        while not stop_button:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
                
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to better detect patterns
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Detect edges using Canny
            edges = cv2.Canny(gray, threshold1, threshold2)
            
            # Make edges thicker
            kernel = np.ones((5,5), np.uint8)
            thick_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Detect patterns (circular and rectangular shapes)
            pattern_kernel = np.ones((3,3), np.uint8)
            patterns = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, pattern_kernel)
            pattern_contours, _ = cv2.findContours(patterns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create pattern mask
            pattern_mask = np.zeros_like(frame_rgb)
            
            # Draw patterns and edges
            for contour in pattern_contours:
                area = cv2.contourArea(contour)
                if area > 50 and area < 500:  # Filter by size to detect button-sized patterns
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h
                    if 0.5 <= aspect_ratio <= 2.0:  # Filter patterns by shape
                        cv2.rectangle(pattern_mask, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Create edge mask
            edges_mask = cv2.cvtColor(thick_edges, cv2.COLOR_GRAY2RGB)
            edges_mask[np.where((edges_mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0]  # Red for edges
            
            # Combine original frame with patterns and edges
            overlay = cv2.addWeighted(frame_rgb, 1.0, edges_mask, 0.7, 0)
            overlay = cv2.addWeighted(overlay, 1.0, pattern_mask, 0.5, 0)
            
            # Display the frame
            frame_placeholder.image(overlay, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources when stopped
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    main()