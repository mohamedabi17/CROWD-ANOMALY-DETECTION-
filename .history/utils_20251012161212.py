"""
Utility functions for crowd anomaly detection
"""

import os
import cv2
import numpy as np
from pathlib import Path


def create_directory(path):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_frames_from_directory(directory_path, extension='.tif'):
    """
    Read all frames from a directory
    
    Args:
        directory_path: Path to directory containing frames
        extension: File extension to filter
        
    Returns:
        List of frame file paths sorted by name
    """
    files = [f for f in os.listdir(directory_path) 
             if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(extension)]
    
    # Remove system files
    files = [f for f in files if not f.startswith('.')]
    
    # Sort files
    files.sort()
    
    return files


def load_image(image_path, grayscale=True):
    """
    Load an image from path
    
    Args:
        image_path: Path to image file
        grayscale: Whether to load as grayscale
        
    Returns:
        Loaded image
    """
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path)


def save_image(image, output_path):
    """
    Save an image to path
    
    Args:
        image: Image array to save
        output_path: Output file path
    """
    cv2.imwrite(output_path, image)


def draw_bounding_box(image, top_left, bottom_right, color=(255, 255, 0), thickness=2):
    """
    Draw a bounding box on image
    
    Args:
        image: Image to draw on
        top_left: (x, y) coordinates of top-left corner
        bottom_right: (x, y) coordinates of bottom-right corner
        color: Box color in BGR
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image


def calculate_metrics(true_positive, false_positive, false_negative):
    """
    Calculate precision, recall, and F1 score
    
    Args:
        true_positive: Number of true positives
        false_positive: Number of false positives
        false_negative: Number of false negatives
        
    Returns:
        Dictionary with precision, recall, and f1_score
    """
    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2.0 * precision * recall / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def normalize_frame(frame):
    """
    Normalize frame to [0, 1] range
    
    Args:
        frame: Input frame
        
    Returns:
        Normalized frame
    """
    return frame.astype(np.float32) / 255.0


def apply_threshold(image, threshold):
    """
    Apply threshold to image
    
    Args:
        image: Input image
        threshold: Threshold value
        
    Returns:
        Binary image
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def resize_frame(frame, width=None, height=None):
    """
    Resize frame to specified dimensions
    
    Args:
        frame: Input frame
        width: Target width (None to maintain aspect ratio)
        height: Target height (None to maintain aspect ratio)
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is None:
        aspect_ratio = height / h
        width = int(w * aspect_ratio)
    elif height is None:
        aspect_ratio = width / w
        height = int(h * aspect_ratio)
    
    return cv2.resize(frame, (width, height))


def get_video_info(video_path):
    """
    Get information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return info


def visualize_optical_flow(flow, scale=1.0):
    """
    Visualize optical flow as HSV image
    
    Args:
        flow: Optical flow array (H x W x 2)
        scale: Scaling factor for visualization
        
    Returns:
        RGB visualization of optical flow
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
