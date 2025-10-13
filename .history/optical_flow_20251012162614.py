"""
Optical Flow computation and analysis for anomaly detection
"""

import cv2
import numpy as np


class OpticalFlowAnalyzer:
    """
    Class for computing and analyzing optical flow
    """
    
    def __init__(self, method='farneback', mag_threshold=1e-3):
        """
        Initialize optical flow analyzer
        
        Args:
            method: Method for optical flow computation ('farneback', 'lucas_kanade')
            mag_threshold: Minimum magnitude threshold for motion detection
        """
        self.method = method
        self.mag_threshold = mag_threshold
        
    def compute_flow(self, prev_frame, curr_frame):
        """
        Compute optical flow between two frames
        
        Args:
            prev_frame: Previous grayscale frame
            curr_frame: Current grayscale frame
            
        Returns:
            flow: Optical flow field (H x W x 2)
        """
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif self.method == 'lucas_kanade':
            # Lucas-Kanade is sparse, here we use Farneback as fallback
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
            
        return flow
    
    def compute_magnitude_angle(self, flow):
        """
        Compute magnitude and angle from optical flow
        
        Args:
            flow: Optical flow field (H x W x 2)
            
        Returns:
            magnitude: Flow magnitude
            angle: Flow angle in degrees (0-360)
        """
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        
        # Compute magnitude
        magnitude = np.sqrt(fx * fx + fy * fy)
        
        # Compute angle in degrees
        angle = np.arctan2(fy, fx)
        angle = (angle * 180 / np.pi + 360) % 360
        
        return magnitude, angle
    
    def discretize_angles(self, angle, magnitude, num_bins=8):
        """
        Discretize angles into bins
        
        Args:
            angle: Flow angles in degrees (0-360)
            magnitude: Flow magnitudes
            num_bins: Number of directional bins
            
        Returns:
            bins: Discretized direction bins (0 = no motion, 1-num_bins = directions)
        """
        bin_size = 360.0 / num_bins
        bins = np.ceil(angle / bin_size).astype(np.uint8)
        
        # Wrap bin num_bins+1 to 1
        bins[bins == num_bins + 1] = 1
        
        # Set to 0 for low magnitude (no significant motion)
        bins[magnitude < self.mag_threshold] = 0
        
        return bins
    
    def create_flow_histogram(self, bins, num_bins=8):
        """
        Create histogram of flow directions
        
        Args:
            bins: Direction bins
            num_bins: Number of bins
            
        Returns:
            hist: Histogram of directions (length = num_bins + 1, index 0 is no motion)
        """
        hist, _ = np.histogram(bins.flatten(), bins=np.arange(num_bins + 2))
        return hist
    
    def visualize_flow(self, flow, frame_shape):
        """
        Create HSV visualization of optical flow
        
        Args:
            flow: Optical flow field
            frame_shape: Shape of original frame
            
        Returns:
            flow_rgb: RGB visualization of flow
        """
        h, w = frame_shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        magnitude, angle = self.compute_magnitude_angle(flow)
        
        hsv[..., 0] = (angle / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_rgb
    
    def compute_flow_features(self, prev_frame, curr_frame, foreground_mask=None):
        """
        Compute complete flow features for anomaly detection
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            foreground_mask: Optional foreground mask
            
        Returns:
            Dictionary containing flow features
        """
        # Compute flow
        flow = self.compute_flow(prev_frame, curr_frame)
        
        # Compute magnitude and angle
        magnitude, angle = self.compute_magnitude_angle(flow)
        
        # Discretize angles
        bins = self.discretize_angles(angle, magnitude)
        
        # Create histogram
        hist = self.create_flow_histogram(bins)
        
        # Apply foreground mask if provided
        if foreground_mask is not None:
            magnitude_fg = magnitude * (foreground_mask > 0)
            bins_fg = bins * (foreground_mask > 0)
            hist_fg = self.create_flow_histogram(bins_fg)
        else:
            magnitude_fg = magnitude
            hist_fg = hist
        
        return {
            'flow': flow,
            'magnitude': magnitude,
            'angle': angle,
            'bins': bins,
            'histogram': hist,
            'histogram_fg': hist_fg,
            'avg_magnitude': magnitude_fg.mean(),
            'max_magnitude': magnitude_fg.max()
        }


def compute_dense_optical_flow(prev_frame, curr_frame, method='farneback'):
    """
    Convenience function to compute dense optical flow
    
    Args:
        prev_frame: Previous grayscale frame
        curr_frame: Current grayscale frame
        method: Optical flow method
        
    Returns:
        flow: Optical flow field
    """
    analyzer = OpticalFlowAnalyzer(method=method)
    return analyzer.compute_flow(prev_frame, curr_frame)


def draw_flow_vectors(frame, flow, step=16, scale=1.0, color=(0, 255, 0)):
    """
    Draw optical flow vectors on frame
    
    Args:
        frame: Frame to draw on
        flow: Optical flow field
        step: Sampling step for vectors
        scale: Scaling factor for vector length
        color: Vector color
        
    Returns:
        frame_with_vectors: Frame with flow vectors drawn
    """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T
    
    lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    vis = frame.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, tipLength=0.3)
    
    return vis
