import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image
import torch
from io import BytesIO
import requests
from typing import List, Dict, Tuple, Optional
import easyocr
import math
import os
import wget
from pathlib import Path

# Import required models with better error handling
MODELS_AVAILABLE = {
    'grounding_dino': False,
    'sam': False,
    'easyocr': False
}

try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.vl_utils import create_positive_map_from_span
    MODELS_AVAILABLE['grounding_dino'] = True
except ImportError:
    st.warning("Grounding DINO not available. Will use alternative detection methods.")

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    MODELS_AVAILABLE['sam'] = True
except ImportError:
    st.warning("SAM not available. Will use OpenCV-based segmentation.")

try:
    import easyocr
    MODELS_AVAILABLE['easyocr'] = True
except ImportError:
    st.warning("EasyOCR not available. Will use OpenCV-based OCR.")

try:
    import supervision as sv
except ImportError:
    st.info("Supervision library not available (optional).")

class ModelDownloader:
    """Handle model downloads and setup"""
    
    @staticmethod
    def download_file(url: str, filename: str, force_download: bool = False) -> bool:
        """Download a file if it doesn't exist"""
        if os.path.exists(filename) and not force_download:
            return True
            
        try:
            st.info(f"Downloading {filename}...")
            wget.download(url, filename)
            st.success(f"Downloaded {filename}")
            return True
        except Exception as e:
            st.error(f"Failed to download {filename}: {e}")
            return False
    
    @staticmethod
    def setup_grounding_dino():
        """Setup Grounding DINO model files"""
        # URLs for model files
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        
        # Download files
        config_path = "configs/GroundingDINO_SwinT_OGC.py"
        model_path = "models/groundingdino_swint_ogc.pth"
        
        config_success = ModelDownloader.download_file(config_url, config_path)
        model_success = ModelDownloader.download_file(model_url, model_path)
        
        return config_success and model_success, config_path, model_path
    
    @staticmethod
    def setup_sam():
        """Setup SAM model files"""
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        os.makedirs("models", exist_ok=True)
        model_path = "models/sam_vit_h_4b8939.pth"
        
        success = ModelDownloader.download_file(model_url, model_path)
        return success, model_path

class RemoteControlAnalyzer:
    """Main class for analyzing remote control images"""
    
    def __init__(self):
        self.grounding_dino_model = None
        self.sam_predictor = None
        self.ocr_reader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {self.device}")
        
    def load_grounding_dino(self):
        """Load Grounding DINO model with proper setup"""
        if not MODELS_AVAILABLE['grounding_dino']:
            return None
            
        try:
            # Setup model files
            setup_success, config_path, checkpoint_path = ModelDownloader.setup_grounding_dino()
            
            if not setup_success:
                st.error("Failed to setup Grounding DINO model files")
                return None
            
            # Load configuration
            args = SLConfig.fromfile(config_path)
            args.device = self.device
            model = build_model(args)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            model.eval()
            return model.to(self.device)
            
        except Exception as e:
            st.error(f"Error loading Grounding DINO: {e}")
            return None
    
    def load_sam(self):
        """Load Segment Anything Model with proper setup"""
        if not MODELS_AVAILABLE['sam']:
            return None
            
        try:
            # Setup model files
            setup_success, model_path = ModelDownloader.setup_sam()
            
            if not setup_success:
                st.error("Failed to setup SAM model files")
                return None
            
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            return predictor
            
        except Exception as e:
            st.error(f"Error loading SAM: {e}")
            return None
    
    def load_ocr(self):
        """Load EasyOCR reader"""
        if not MODELS_AVAILABLE['easyocr']:
            return None
            
        try:
            reader = easyocr.Reader(['en'])
            return reader
        except Exception as e:
            st.error(f"Error loading EasyOCR: {e}")
            return None
    
    def detect_remote_opencv_fallback(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback remote detection using OpenCV
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest rectangular contour (likely the remote)
            best_contour = None
            max_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area and area > 1000:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4-8 vertices)
                    if 4 <= len(approx) <= 8:
                        best_contour = contour
                        max_area = area
            
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                return (x, y, x + w, y + h)
                
        except Exception as e:
            st.error(f"OpenCV fallback detection failed: {e}")
            
        return None
    
    def detect_remote_with_dino(self, image: np.ndarray, text_prompt: str = "remote control") -> Optional[Tuple[int, int, int, int]]:
        """
        Detect remote control using Grounding DINO or OpenCV fallback
        """
        # Try Grounding DINO first
        if MODELS_AVAILABLE['grounding_dino']:
            if self.grounding_dino_model is None:
                self.grounding_dino_model = self.load_grounding_dino()
                
            if self.grounding_dino_model is not None:
                try:
                    # Convert image to RGB if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image
                        
                    # Prepare image for DINO
                    image_pil = Image.fromarray(image_rgb)
                    
                    # Run inference
                    with torch.no_grad():
                        boxes, logits, phrases = self.grounding_dino_model(image_pil, text_prompt)
                    
                    if len(boxes) > 0:
                        # Get the box with highest confidence
                        best_box_idx = torch.argmax(logits)
                        box = boxes[best_box_idx]
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w = image.shape[:2]
                        x1 = int(box[0] * w)
                        y1 = int(box[1] * h)
                        x2 = int(box[2] * w)
                        y2 = int(box[3] * h)
                        
                        return (x1, y1, x2, y2)
                        
                except Exception as e:
                    st.warning(f"DINO detection failed: {e}. Using OpenCV fallback.")
        
        # Fallback to OpenCV detection
        st.info("Using OpenCV-based remote detection...")
        return self.detect_remote_opencv_fallback(image)
    
    def segment_remote_opencv_fallback(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Fallback segmentation using OpenCV
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Create a simple rectangular mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            
            # Try to refine the mask using GrabCut
            try:
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Create initial mask for GrabCut
                gc_mask = np.zeros(image.shape[:2], np.uint8)
                gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
                gc_mask[y1+10:y2-10, x1+10:x2-10] = cv2.GC_FGD
                
                # Apply GrabCut
                cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                
                # Create final mask
                refined_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
                return refined_mask
                
            except:
                # If GrabCut fails, return simple rectangular mask
                return (mask > 0).astype(np.uint8)
                
        except Exception as e:
            st.error(f"OpenCV segmentation failed: {e}")
            return None
    
    def segment_remote_with_sam(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Segment the remote control using SAM with OpenCV fallback
        """
        # Try SAM first
        if MODELS_AVAILABLE['sam']:
            if self.sam_predictor is None:
                self.sam_predictor = self.load_sam()
                
            if self.sam_predictor is not None:
                try:
                    self.sam_predictor.set_image(image)
                    
                    # Use bounding box as prompt
                    input_box = np.array([bbox])
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=False,
                    )
                    
                    if len(masks) > 0:
                        return masks[0]
                        
                except Exception as e:
                    st.warning(f"SAM segmentation failed: {e}. Using OpenCV fallback.")
        
        # Fallback to OpenCV
        st.info("Using OpenCV-based segmentation...")
        return self.segment_remote_opencv_fallback(image, bbox)
    
    def detect_buttons_opencv(self, image: np.ndarray, remote_mask: np.ndarray) -> List[np.ndarray]:
        """
        Detect buttons using OpenCV when SAM is not available
        """
        try:
            # Apply mask to image
            masked_image = image.copy()
            masked_image[remote_mask == 0] = 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            button_masks = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (buttons should be reasonably sized)
                if 100 < area < 5000:
                    # Create mask for this button
                    button_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(button_mask, [contour], 255)
                    
                    # Only include if it's within the remote region
                    if np.any(button_mask & remote_mask):
                        button_masks.append((button_mask > 0).astype(np.uint8))
            
            return button_masks
            
        except Exception as e:
            st.error(f"OpenCV button detection failed: {e}")
            return []
    
    def segment_buttons_with_sam(self, image: np.ndarray, remote_mask: np.ndarray) -> List[np.ndarray]:
        """
        Segment buttons using SAM or OpenCV fallback
        """
        # Try SAM first
        if MODELS_AVAILABLE['sam'] and self.sam_predictor is not None:
            try:
                # Create a masked image (only the remote region)
                masked_image = image.copy()
                masked_image[~remote_mask.astype(bool)] = 0
                
                self.sam_predictor.set_image(masked_image)
                
                # Use automatic mask generation for button detection
                mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam_predictor.model,
                    points_per_side=32,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.8,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                
                masks = mask_generator.generate(masked_image)
                
                # Filter masks that are within the remote region and likely buttons
                button_masks = []
                for mask_data in masks:
                    mask = mask_data['segmentation']
                    
                    # Check if mask is within remote region
                    if np.any(mask & remote_mask):
                        # Filter by size (buttons should be reasonably sized)
                        area = np.sum(mask)
                        if 100 < area < 5000:
                            button_masks.append(mask.astype(np.uint8))
                
                if len(button_masks) > 0:
                    return button_masks
                    
            except Exception as e:
                st.warning(f"SAM button segmentation failed: {e}. Using OpenCV fallback.")
        
        # Fallback to OpenCV
        st.info("Using OpenCV-based button detection...")
        return self.detect_buttons_opencv(image, remote_mask)
    
    def extract_contours_from_masks(self, masks: List[np.ndarray]) -> List[Dict]:
        """
        Extract contours and bounding boxes from segmentation masks
        """
        buttons = []
        
        for i, mask in enumerate(masks):
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Extract polygon points
                polygon = [(int(point[0][0]), int(point[0][1])) for point in approx_contour]
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (x, y, x + w, y + h)
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                buttons.append({
                    'id': f'button_{i+1}',
                    'polygon': polygon,
                    'bbox': bbox,
                    'centroid': (cx, cy),
                    'label': '?'
                })
        
        return buttons
    
    def opencv_text_detection(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        OpenCV-based text detection and recognition fallback
        """
        texts = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply MSER for text detection
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=60,
                _max_area=14400,
                _max_variation=0.25,
                _min_diversity=0.2,
                _max_evolution=200,
                _area_threshold=1.01,
                _min_margin=0.003,
                _edge_blur_size=5
            )
            
            regions, bboxes = mser.detectRegions(gray)
            
            # For each detected region, try to extract text
            for bbox in bboxes:
                x, y, w, h = bbox
                
                # Filter by aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0 and w > 15 and h > 8:
                    # Extract ROI
                    roi = gray[y:y+h, x:x+w]
                    
                    # Simple preprocessing
                    roi = cv2.resize(roi, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                    roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    
                    # For now, we'll just mark detected regions
                    # In a real implementation, you'd use a proper OCR engine here
                    cx, cy = x + w//2, y + h//2
                    texts.append(("?", (cx, cy)))  # Placeholder text
                    
        except Exception as e:
            st.warning(f"OpenCV text detection failed: {e}")
            
        return texts
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using EasyOCR with OpenCV fallback
        """
        text_boxes = []
        
        # Try EasyOCR first
        if MODELS_AVAILABLE['easyocr']:
            try:
                if self.ocr_reader is None:
                    self.ocr_reader = self.load_ocr()
                
                if self.ocr_reader is not None:
                    results = self.ocr_reader.readtext(image, detail=1)
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:
                            bbox_array = np.array(bbox)
                            x1 = int(np.min(bbox_array[:, 0]))
                            y1 = int(np.min(bbox_array[:, 1]))
                            x2 = int(np.max(bbox_array[:, 0]))
                            y2 = int(np.max(bbox_array[:, 1]))
                            text_boxes.append((x1, y1, x2, y2))
                    
                    return text_boxes
                    
            except Exception as e:
                st.warning(f"EasyOCR failed: {e}. Using OpenCV fallback.")
        
        # OpenCV fallback
        st.info("Using OpenCV-based text detection...")
        return self._opencv_text_detection_boxes(image)
    
    def _opencv_text_detection_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """OpenCV text detection returning bounding boxes"""
        text_boxes = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # MSER detection
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=60,
                _max_area=14400,
                _max_variation=0.25,
                _min_diversity=0.2,
                _max_evolution=200,
                _area_threshold=1.01,
                _min_margin=0.003,
                _edge_blur_size=5
            )
            
            regions, bboxes = mser.detectRegions(gray)
            
            for bbox in bboxes:
                x, y, w, h = bbox
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0 and w > 10 and h > 5:
                    text_boxes.append((x, y, x + w, y + h))
                    
        except Exception as e:
            st.warning(f"OpenCV text detection failed: {e}")
            
        return text_boxes
    
    def extract_text_with_ocr(self, image: np.ndarray, text_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Extract text using EasyOCR or OpenCV fallback
        """
        # Try EasyOCR first
        if MODELS_AVAILABLE['easyocr']:
            if self.ocr_reader is None:
                self.ocr_reader = self.load_ocr()
                
            if self.ocr_reader is not None:
                try:
                    results = self.ocr_reader.readtext(image, detail=1)
                    texts = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3 and text.strip():
                            bbox_array = np.array(bbox)
                            cx = int(np.mean(bbox_array[:, 0]))
                            cy = int(np.mean(bbox_array[:, 1]))
                            texts.append((text.strip(), (cx, cy)))
                    
                    return texts
                    
                except Exception as e:
                    st.warning(f"EasyOCR text extraction failed: {e}. Using OpenCV fallback.")
        
        # OpenCV fallback
        st.info("Using OpenCV-based text extraction...")
        return self.opencv_text_detection(image)
    
    def associate_text_with_buttons(self, buttons: List[Dict], texts: List[Tuple[str, Tuple[int, int]]]) -> List[Dict]:
        """
        Associate detected text with nearest buttons
        """
        result_buttons = [button.copy() for button in buttons]
        
        for text, text_centroid in texts:
            min_distance = float('inf')
            closest_button_idx = -1
            
            for i, button in enumerate(result_buttons):
                button_centroid = button['centroid']
                distance = math.sqrt(
                    (text_centroid[0] - button_centroid[0])**2 + 
                    (text_centroid[1] - button_centroid[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_button_idx = i
            
            # Associate text with closest button if within reasonable distance
            if closest_button_idx >= 0 and min_distance < 100:
                result_buttons[closest_button_idx]['label'] = text
        
        return result_buttons
    
    def create_output_json(self, remote_bbox: Tuple[int, int, int, int], buttons: List[Dict]) -> Dict:
        """Create the final JSON output structure"""
        x1, y1, x2, y2 = remote_bbox
        remote_polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        clean_buttons = []
        for button in buttons:
            clean_button = {
                'id': button['id'],
                'polygon': button['polygon'],
                'bbox': list(button['bbox']),
                'label': button['label']
            }
            clean_buttons.append(clean_button)
        
        result = {
            'remote': {
                'polygon': remote_polygon,
                'bbox': list(remote_bbox)
            },
            'buttons': clean_buttons
        }
        
        return result
    
    def visualize_results(self, image: np.ndarray, remote_bbox: Tuple[int, int, int, int], buttons: List[Dict]) -> np.ndarray:
        """Create visualization with button polygons and labels"""
        vis_image = image.copy()
        
        # Draw remote bounding box in blue
        cv2.rectangle(vis_image, (remote_bbox[0], remote_bbox[1]), 
                     (remote_bbox[2], remote_bbox[3]), (255, 0, 0), 2)
        
        # Draw button polygons in red and add labels
        for button in buttons:
            # Draw polygon
            polygon_np = np.array(button['polygon'], np.int32)
            cv2.polylines(vis_image, [polygon_np], True, (0, 0, 255), 2)
            
            # Add label
            centroid = button['centroid']
            if button['label'] != '?':
                cv2.putText(vis_image, button['label'], 
                           (centroid[0] - 10, centroid[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add button ID
            cv2.putText(vis_image, button['id'], 
                       (centroid[0] - 10, centroid[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return vis_image

def main():
    st.set_page_config(page_title="Remote Control Analyzer", layout="wide")
    
    st.title("🎮 Remote Control Button Analyzer")
    st.markdown("Upload an image of a remote control to automatically detect and label all buttons!")
    
    # Show available models
    with st.expander("📋 Model Availability Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Available" if MODELS_AVAILABLE['grounding_dino'] else "❌ Not Available (OpenCV fallback)"
            st.write(f"**Grounding DINO:** {status}")
        with col2:
            status = "✅ Available" if MODELS_AVAILABLE['sam'] else "❌ Not Available (OpenCV fallback)"
            st.write(f"**SAM:** {status}")
        with col3:
            status = "✅ Available" if MODELS_AVAILABLE['easyocr'] else "❌ Not Available (OpenCV fallback)"
            st.write(f"**EasyOCR:** {status}")
    
    # Model setup section
    if not all(MODELS_AVAILABLE.values()):
        with st.expander("🔧 Model Setup Instructions"):
            st.markdown("""
            ### Missing Dependencies
            
            To get the best results, install the following packages:
            
            ```bash
            # For Grounding DINO
            pip install groundingdino-py
            
            # For SAM
            pip install segment-anything
            
            # For EasyOCR
            pip install easyocr
            
            # For model downloads
            pip install wget
            ```
            
            **Note:** The application will work with OpenCV fallbacks even without these packages, but accuracy may be reduced.
            """)
    
    # Initialize analyzer
    analyzer = RemoteControlAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a remote control image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a remote control"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Remote Control", use_container_width=True)
        
        # Process button
        if st.button("🔍 Analyze Remote Control", type="primary"):
            with st.spinner("Processing... This may take a few minutes"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Detect remote
                status_text.text("Step 1/7: Detecting remote control...")
                progress_bar.progress(1/7)
                
                remote_bbox = analyzer.detect_remote_with_dino(image_np)
                if remote_bbox is None:
                    st.error("Could not detect remote control in the image. Please try another image.")
                    st.stop()
                
                st.success(f"Remote detected at: {remote_bbox}")
                
                # Step 2: Segment remote
                status_text.text("Step 2/7: Segmenting remote control...")
                progress_bar.progress(2/7)
                
                remote_mask = analyzer.segment_remote_with_sam(image_np, remote_bbox)
                if remote_mask is None:
                    st.error("Could not segment remote control.")
                    st.stop()
                
                # Step 3: Segment buttons
                status_text.text("Step 3/7: Detecting buttons...")
                progress_bar.progress(3/7)
                
                button_masks = analyzer.segment_buttons_with_sam(image_np, remote_mask)
                st.info(f"Found {len(button_masks)} potential buttons")
                
                # Step 4: Extract contours
                status_text.text("Step 4/7: Extracting button contours...")
                progress_bar.progress(4/7)
                
                buttons = analyzer.extract_contours_from_masks(button_masks)
                
                # Step 5: Detect text regions
                status_text.text("Step 5/7: Detecting text regions...")
                progress_bar.progress(5/7)
                
                text_boxes = analyzer.detect_text_regions(image_np)
                
                # Step 6: Extract text
                status_text.text("Step 6/7: Recognizing text...")
                progress_bar.progress(6/7)
                
                texts = analyzer.extract_text_with_ocr(image_np, text_boxes)
                st.info(f"Found {len(texts)} text labels")
                
                # Step 7: Associate text with buttons
                status_text.text("Step 7/7: Associating text with buttons...")
                progress_bar.progress(7/7)
                
                final_buttons = analyzer.associate_text_with_buttons(buttons, texts)
                
                # Create output JSON
                result_json = analyzer.create_output_json(remote_bbox, final_buttons)
                
                # Create visualization
                vis_image = analyzer.visualize_results(image_np, remote_bbox, final_buttons)
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Annotated Image")
                    st.image(vis_image, caption="Detected Buttons and Labels", use_container_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    st.json(result_json, expanded=True)
                
                # Summary statistics
                st.subheader("📊 Analysis Summary")
                labeled_buttons = sum(1 for btn in final_buttons if btn['label'] != '?')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Buttons", len(final_buttons))
                with col2:
                    st.metric("Labeled Buttons", labeled_buttons)
                with col3:
                    st.metric("Unlabeled Buttons", len(final_buttons) - labeled_buttons)
                
                # Button details table
                if final_buttons:
                    st.subheader("🔘 Button Details")
                    button_data = []
                    for btn in final_buttons:
                        button_data.append({
                            'ID': btn['id'],
                            'Label': btn['label'],
                            'Bounding Box': f"({btn['bbox'][0]}, {btn['bbox'][1]}) to ({btn['bbox'][2]}, {btn['bbox'][3]})",
                            'Polygon Points': len(btn['polygon'])
                        })
                    
                    st.dataframe(button_data, use_container_width=True)
                
                # Download results
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download JSON
                    json_str = json.dumps(result_json, indent=2)
                    st.download_button(
                        label="Download JSON Results",
                        data=json_str,
                        file_name="remote_analysis_results.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Download annotated image
                    is_success, buffer = cv2.imencode(".png", vis_image)
                    if is_success:
                        st.download_button(
                            label="Download Annotated Image",
                            data=buffer.tobytes(),
                            file_name="annotated_remote.png",
                            mime="image/png"
                        )

if __name__ == "__main__":
    main()