"""
Model utilities for anomaly detection
Provides helper functions for model operations
"""

import numpy as np
import cv2


def extract_features(bins, magnitude, foreground_mask, region_size):
    """
    Extract features from a region for anomaly detection
    
    Args:
        bins: Direction bins from optical flow
        magnitude: Magnitude of optical flow
        foreground_mask: Binary mask of foreground pixels
        region_size: Size of the region (n x n)
        
    Returns:
        Feature vector for classification
    """
    # Count histogram of direction bins
    hist, _ = np.histogram(bins.flatten(), bins=np.arange(10))
    
    # Calculate average magnitude
    avg_magnitude = magnitude.flatten().mean()
    
    # Count foreground pixels
    fg_count = np.count_nonzero(foreground_mask)
    
    # Combine features
    features = hist.tolist()
    features.extend([fg_count, avg_magnitude])
    
    return features


def calculate_optical_flow_features(prev_frame, curr_frame, mag_threshold=1e-3):
    """
    Calculate optical flow features between two frames
    
    Args:
        prev_frame: Previous grayscale frame
        curr_frame: Current grayscale frame
        mag_threshold: Minimum magnitude threshold
        
    Returns:
        bins: Direction bins (0-8, where 0 means no motion)
        magnitude: Flow magnitude
    """
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    
    # Extract flow components
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # Calculate angle and magnitude
    angle = ((np.arctan2(fy, fx + 1e-10) + 2 * np.pi) * 180) % 360
    magnitude = np.sqrt(fx * fx + fy * fy)
    
    # Bin angles into 8 directions (45 degrees each)
    binno = np.ceil(angle / 45).astype(np.uint8)
    
    # Set bin to 0 for low magnitude (no significant motion)
    binno[magnitude < mag_threshold] = 0
    
    return binno, magnitude


def prepare_training_data(features_list, labels_list):
    """
    Prepare training data from feature lists
    
    Args:
        features_list: List of feature vectors
        labels_list: List of corresponding labels
        
    Returns:
        X: Feature matrix
        y: Label vector
    """
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y


class AnomalyDetector:
    """
    Base class for anomaly detection models
    """
    
    def __init__(self, model_type='decision_tree'):
        self.model_type = model_type
        self.model = None
        
    def train(self, X, y):
        """Train the model on features and labels"""
        if self.model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(max_depth=5)
            self.model.fit(X, y)
        elif self.model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=5)
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def predict(self, X):
        """Predict labels for features"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict probabilities for features"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
