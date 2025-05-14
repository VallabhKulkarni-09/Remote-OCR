import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Remote Image Edge Detection")
    
    try:
        # Set fixed parameters for edge detection
        threshold1 = 100
        threshold2 = 200
        blur_kernel = 5
        
        # Create placeholders for preview and processed image
        preview_placeholder = st.empty()
        processed_placeholder = st.empty()
        
        # Initialize session state to store the camera state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = True
            st.session_state.frame = None
        
        # Add capture button at the top
        capture_button = st.button("Capture Image")
        
        # Create a video capture object
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the camera. Please make sure your camera is connected and not in use by another application.")
            return
            
        # Show preview and handle capture
        while st.session_state.camera_active and not capture_button:
            ret, frame = cap.read()
            if ret:
                # Convert to RGB for display
                preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(preview_frame, use_container_width=True)
                st.session_state.frame = frame  # Store the frame for processing
                # Add small delay to prevent excessive CPU usage
                cv2.waitKey(1)
            else:
                st.error("Failed to get camera preview")
                break
        
        # Process the captured frame when button is clicked
        if capture_button:
            # Stop the preview
            st.session_state.camera_active = False
            frame = st.session_state.frame
            
            if frame is None:
                st.error("No frame captured")
                return
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
            
            # Draw patterns and edges, and store button regions
            button_regions = []
            for contour in pattern_contours:
                area = cv2.contourArea(contour)
                if area > 50 and area < 500:  # Filter by size to detect button-sized patterns
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h
                    if 0.5 <= aspect_ratio <= 2.0:  # Filter patterns by shape
                        # Draw rectangle on pattern mask
                        cv2.rectangle(pattern_mask, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Extract and enhance button region
                        button_region = frame[y-5:y+h+5, x-5:x+w+5]  # Add padding
                        if button_region.size > 0:  # Check if region is valid
                            # Enhance the button region
                            button_gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
                            button_enhanced = cv2.equalizeHist(button_gray)  # Enhance contrast
                            button_denoised = cv2.fastNlMeansDenoising(button_enhanced)  # Reduce noise
                            button_sharpened = cv2.filter2D(button_denoised, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))  # Sharpen
                            button_regions.append({
                                'original': cv2.cvtColor(button_region, cv2.COLOR_BGR2RGB),
                                'enhanced': cv2.cvtColor(cv2.cvtColor(button_sharpened, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                            })
            
            # Create edge mask
            edges_mask = cv2.cvtColor(thick_edges, cv2.COLOR_GRAY2RGB)
            edges_mask[np.where((edges_mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0]  # Red for edges
            
            # Combine original frame with patterns and edges
            overlay = cv2.addWeighted(frame_rgb, 1.0, edges_mask, 0.7, 0)
            overlay = cv2.addWeighted(overlay, 1.0, pattern_mask, 0.5, 0)
            
            # Display the processed image
            st.subheader("Full Image with Detected Buttons")
            processed_placeholder.image(overlay, use_container_width=True)
            
            # Display enhanced button regions
            if button_regions:
                st.subheader(f"Detected Buttons ({len(button_regions)})")
                cols = st.columns(2 * len(button_regions))  # 2 columns per button (original and enhanced)
                for i, button in enumerate(button_regions):
                    with cols[i*2]:
                        st.write(f"Button {i+1} - Original")
                        st.image(button['original'], use_column_width=True)
                    with cols[i*2 + 1]:
                        st.write(f"Button {i+1} - Enhanced")
                        st.image(button['enhanced'], use_column_width=True)
            else:
                st.info("No buttons detected in the image")
            
            # Clear the preview
            preview_placeholder.empty()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources when stopped
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    main()