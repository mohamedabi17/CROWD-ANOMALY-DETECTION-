"""
Optimized Anomaly Detection System - Production Version
Incorporates best practices for improved precision and performance
"""

import sys
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join, isdir
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from UCSDped1 import TestVideoFile
import time
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import model
import utils
import optical_flow
import shutil
from collections import deque

counter = 1

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def create_optimized_training_data():
    """Create high-quality diverse training data"""
    x_train = []
    y_train = []
    
    # Normal pedestrian patterns (600 samples)
    for i in range(600):
        # Realistic walking patterns
        main_direction = (i % 8) + 1
        pattern = [0] * 9
        
        # Primary direction with natural variation
        pattern[main_direction] = 15 + np.random.randint(0, 20)
        
        # Adjacent directions (natural spread)
        if main_direction > 1:
            pattern[main_direction - 1] = 3 + np.random.randint(0, 8)
        if main_direction < 8:
            pattern[main_direction + 1] = 3 + np.random.randint(0, 8)
        
        # Some stationary behavior
        pattern[0] = 1 + np.random.randint(0, 5)
        
        # Small amounts in other directions (noise)
        for j in range(1, 9):
            if j not in [main_direction, main_direction-1, main_direction+1]:
                pattern[j] = np.random.randint(0, 3)
        
        # Realistic magnitude and foreground count for walking
        fg_count = 15 + np.random.randint(0, 25)
        magnitude = 0.01 + np.random.random() * 0.04
        
        pattern.extend([fg_count, magnitude])
        x_train.append(pattern)
        y_train.append(0)
    
    # Vehicle anomalies (150 samples)
    for i in range(150):
        pattern = [0] * 9
        # Vehicles: fast, straight movement
        main_dir = (i % 4) * 2 + 1  # Prefer cardinal directions
        pattern[main_dir] = 30 + np.random.randint(0, 25)
        pattern[0] = 0  # No stationary
        
        # Very high magnitude and foreground count
        fg_count = 40 + np.random.randint(0, 30)
        magnitude = 0.08 + np.random.random() * 0.12
        
        pattern.extend([fg_count, magnitude])
        x_train.append(pattern)
        y_train.append(1)
    
    # Running/fast movement anomalies (100 samples)
    for i in range(100):
        pattern = [0] * 9
        main_dir = (i % 8) + 1
        pattern[main_dir] = 20 + np.random.randint(0, 15)
        pattern[(main_dir % 8) + 1] = 5 + np.random.randint(0, 10)
        pattern[0] = 0  # Minimal stationary
        
        fg_count = 25 + np.random.randint(0, 20)
        magnitude = 0.05 + np.random.random() * 0.08
        
        pattern.extend([fg_count, magnitude])
        x_train.append(pattern)
        y_train.append(1)
    
    # Erratic movement anomalies (100 samples)
    for i in range(100):
        pattern = [np.random.randint(2, 12) for _ in range(9)]
        pattern[0] = np.random.randint(0, 3)  # Little stationary
        
        fg_count = 20 + np.random.randint(0, 25)
        magnitude = 0.03 + np.random.random() * 0.09
        
        pattern.extend([fg_count, magnitude])
        x_train.append(pattern)
        y_train.append(1)
    
    # Large object anomalies (50 samples)
    for i in range(50):
        pattern = [np.random.randint(5, 20) for _ in range(9)]
        
        fg_count = 50 + np.random.randint(0, 40)
        magnitude = 0.06 + np.random.random() * 0.15
        
        pattern.extend([fg_count, magnitude])
        x_train.append(pattern)
        y_train.append(1)
    
    return np.array(x_train), np.array(y_train)

def load_optimized_features(type):
    """Load features with enhanced error handling and data quality"""
    x_train = []
    y_train = []
    
    try:
        if not os.path.exists('features/'):
            print("Creating optimized training data...")
            return create_optimized_training_data()
            
        features = [f for f in listdir('features/') if f.startswith("features_test_"+type)]
        
        if not features:
            print("No feature files found. Creating optimized training data...")
            return create_optimized_training_data()
            
        for feature in features:
            file_path = os.path.join('features', feature)
            with open(file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            parts = line.split()
                            if len(parts) >= 12:
                                features_vals = [float(x) for x in parts[:-1]]
                                label = int(parts[-1])
                                
                                if len(features_vals) == 11:
                                    x_train.append(features_vals)
                                    y_train.append(label)
                        except (ValueError, IndexError):
                            continue
        
        # If loaded data is insufficient, supplement with generated data
        if len(x_train) < 100:
            print(f"Loaded {len(x_train)} samples. Supplementing with generated data...")
            gen_x, gen_y = create_optimized_training_data()
            x_train.extend(gen_x.tolist())
            y_train.extend(gen_y.tolist())
                                
    except Exception as e:
        print(f"Error loading features: {e}. Using generated data.")
        return create_optimized_training_data()
        
    return np.array(x_train), np.array(y_train)

class OptimizedUCSDTest:
    def __init__(self, path, n, detect_interval, type):
        self.path = path
        
        # Enhanced background subtraction
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=True
        )
        
        self.n = n
        self.detect_interval = detect_interval
        
        # Optimized classifier
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42
        )
        
        # Feature scaling for better performance
        self.scaler = RobustScaler()
        
        # Temporal smoothing
        self.temporal_window = 3
        self.prediction_history = deque(maxlen=self.temporal_window)
        
        # Confidence threshold
        self.confidence_threshold = 0.6
        
        # Load and prepare training data
        if not os.path.exists(path):
            print(f"Warning: Dataset path '{path}' not found!")
            x, y = create_optimized_training_data()
        else:
            x, y = load_optimized_features(type)
            
        print(f"Training with {len(x)} samples ({np.sum(y==0)} normal, {np.sum(y==1)} anomaly)")
        
        # Scale features and train
        if len(x) > 0:
            x_scaled = self.scaler.fit_transform(x)
            self.clf.fit(x_scaled, y)
            
            # Evaluate training
            scores = cross_val_score(self.clf, x_scaled, y, cv=5, scoring='f1')
            print(f"Training F1-score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Performance tracking
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.total = 0.0
        self.y = []
        self.y_pred = []
        self.confidence_scores = []

    def process_frame_optimized(self, bins, magnitude, fmask, tag_img, frame):
        """Optimized frame processing with enhanced accuracy"""
        # Minimum activity threshold
        if np.count_nonzero(fmask) < 5:
            return False
        
        h, w = bins.shape
        features_j = []
        tag_j = []
        index_i_j = []
        
        for i in range(0, h, self.n):
            i_end = min(h, i + self.n)
            for j in range(0, w, self.n):
                j_end = min(w, j + self.n)
                
                # Check if region has sufficient activity
                region_mask = fmask[i:i_end, j:j_end]
                if np.count_nonzero(region_mask) < 3:
                    continue
                
                # Extract features
                atom_bins = bins[i:i_end, j:j_end].flatten()
                atom_mag = magnitude[i:i_end, j:j_end].flatten()
                
                # Direction histogram
                hs, _ = np.histogram(atom_bins, np.arange(10))
                features = hs.tolist()
                
                # Enhanced magnitude features
                valid_mag = atom_mag[atom_mag > 1e-4]
                if len(valid_mag) > 0:
                    avg_mag = np.mean(valid_mag)
                else:
                    avg_mag = 0.0
                
                features.extend([np.count_nonzero(region_mask), avg_mag])
                
                # Ensure exactly 11 features
                while len(features) < 11:
                    features.append(0.0)
                features = features[:11]
                
                features_j.append(features)
                index_i_j.append((i, j))
                
                # Ground truth evaluation
                if tag_img is not None:
                    tag_region = tag_img[i:i_end, j:j_end]
                    ones = np.count_nonzero(tag_region)
                    tag = 1 if ones > (self.n * self.n * 0.2) else 0
                else:
                    tag = 0
                tag_j.append(tag)
        
        if not features_j:
            return False
        
        # Make predictions with confidence
        features_scaled = self.scaler.transform(features_j)
        probabilities = self.clf.predict_proba(features_scaled)
        predictions = self.clf.predict(features_scaled)
        confidences = np.max(probabilities, axis=1)
        
        # Apply confidence threshold
        high_conf_mask = confidences >= self.confidence_threshold
        filtered_predictions = predictions * high_conf_mask
        
        # Temporal smoothing
        self.prediction_history.append(filtered_predictions)
        if len(self.prediction_history) >= self.temporal_window:
            # Require consistency across frames
            temporal_votes = np.sum([pred for pred in self.prediction_history], axis=0)
            final_predictions = (temporal_votes >= 2).astype(int)
        else:
            final_predictions = filtered_predictions
        
        # Update tracking
        self.y_pred.extend(final_predictions)
        self.y.extend(tag_j)
        self.confidence_scores.extend(confidences)
        self.total += len(final_predictions)
        
        # Draw detections and update statistics
        found_anomaly = False
        for idx, (pred, conf) in enumerate(zip(final_predictions, confidences)):
            if pred == 1:
                i, j = index_i_j[idx]
                i_end = min(h, i + self.n)
                j_end = min(w, j + self.n)
                
                # Color based on confidence
                if conf > 0.8:
                    color = (0, 0, 255)  # Red for high confidence
                elif conf > 0.7:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow
                
                cv2.rectangle(frame, (j, i), (j_end, i_end), color, 2)
                found_anomaly = True
                
                # Update statistics
                if idx < len(tag_j):
                    if tag_j[idx] == 0:
                        self.false_positive += 1
                    else:
                        self.true_positive += 1
            elif idx < len(tag_j) and tag_j[idx] == 1:
                self.false_negative += 1
        
        return found_anomaly

    def process_video(self, video_name, tag_video):
        """Optimized video processing"""
        global counter 
        mag_threshold = 5e-4  # Slightly more sensitive
        
        video_path = self.path + video_name
        tag_path = self.path + tag_video
        
        if not os.path.exists(video_path):
            print(f"Warning: Video directory '{video_path}' not found. Skipping...")
            return []
            
        # Handle ground truth
        if not os.path.exists(tag_path):
            os.makedirs(tag_path, exist_ok=True)
        
        files = [f for f in listdir(video_path) if isfile(join(video_path, f)) and f.endswith('.tif')]
        files.sort()
        
        if not files:
            return []
        
        # Setup
        anomaly_detected = []
        path_folder_out = f'output_folders_optimized/out{counter}'
        os.makedirs(path_folder_out, exist_ok=True)
        
        # Load first frame
        first_frame_path = os.path.join(video_path, files[0])
        old_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
        
        if old_frame is None:
            return []
        
        h, w = old_frame.shape
        number_frame = 0
        
        for tif in files:
            frame_path = os.path.join(video_path, tif)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            
            if frame is None:
                continue
                
            if number_frame % self.detect_interval == 0:
                # Enhanced foreground detection
                fmask = self.fgbg.apply(frame)
                
                # Morphological operations to clean mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fmask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, kernel)
                
                # Optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Ground truth
                tag_img = None
                if os.path.exists(tag_path):
                    gt_files = [f for f in listdir(tag_path) if f.endswith('.bmp')]
                    if gt_files and number_frame < len(gt_files):
                        gt_files.sort()
                        tag_img_path = os.path.join(tag_path, gt_files[number_frame])
                        tag_img = cv2.imread(tag_img_path, cv2.IMREAD_GRAYSCALE)
                
                # Compute flow features
                fx, fy = flow[:,:,0], flow[:,:,1]
                angle = ((np.arctan2(fy, fx + 1e-10) + 2*np.pi) * 180) % 360
                binno = np.ceil(angle / 45).astype(np.uint8)
                magnitude = np.sqrt(fx*fx + fy*fy)
                binno[magnitude < mag_threshold] = 0
                
                # Process frame
                found_anomaly = self.process_frame_optimized(binno, magnitude, fmask, tag_img, frame)
                if found_anomaly:
                    anomaly_detected.append(number_frame)
            
            # Save frame
            output_frame_path = os.path.join(path_folder_out, f'frame{number_frame:03d}.jpg')
            cv2.imwrite(output_frame_path, frame)
            
            number_frame += 1
            old_frame = frame
        
        counter += 1
        return anomaly_detected

if __name__ == '__main__':
    print("🚀 Optimized Crowd Anomaly Detection System")
    print("=" * 55)
    
    # Clean up previous output
    shutil.rmtree("output_folders_optimized", ignore_errors=True)
    
    ucsdped = 'UCSDped1'
    dataset_path = 'UCSD_Anomaly_Dataset.v1p2/' + ucsdped + '/Test/'
    
    try:
        ucsd_test = OptimizedUCSDTest(dataset_path, 10, 5, ucsdped)
        
        # Get test directories
        dir_test = [f for f in listdir(dataset_path) if isdir(join(dataset_path, f)) and not f.endswith('_gt')]
        dir_test.sort()
        
        total_correct = 0.0
        total_should_found = 0.0
        total_found = 0.0
        
        os.makedirs('output_folders_optimized', exist_ok=True)
        
        print(f"Processing {len(dir_test)} test videos...")
        
        for directory in dir_test:
            if not directory.endswith("gt"):
                start_time = time.time()
                
                try:
                    anomaly_detected = ucsd_test.process_video(directory+'/', directory + '_gt/')
                    time_video = passed_time(start_time)
                    
                    if time_video > 0:
                        fps = 200.0 / time_video
                        print(f"{directory}: {fps:.1f} FPS, {len(anomaly_detected)} anomalies detected")
                    
                    total_found += len(anomaly_detected)
                    index_video = int(directory[-3:]) if directory[-3:].isdigit() else 1
                    
                    if index_video in TestVideoFile:
                        intersection = set(anomaly_detected).intersection(TestVideoFile[index_video])
                        total_correct += len(intersection)
                        total_should_found += len(TestVideoFile[index_video])
                        
                except Exception as e:
                    print(f"Error processing {directory}: {e}")
                    continue

        # Results
        print("\n" + "="*55)
        print("📊 OPTIMIZED RESULTS:")
        print("="*55)
        
        if total_found > 0 and total_should_found > 0:
            precision = total_correct / total_found
            recall = total_correct / total_should_found
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Frame-level metrics:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-score: {f1:.3f}")
        
        # Pixel-level results
        if len(ucsd_test.y) > 0:
            pixel_tp = ucsd_test.true_positive
            pixel_fp = ucsd_test.false_positive
            pixel_fn = ucsd_test.false_negative
            
            if pixel_tp + pixel_fp > 0:
                pixel_precision = pixel_tp / (pixel_tp + pixel_fp)
                pixel_recall = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn > 0 else 0
                pixel_f1 = 2.0 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall) if pixel_precision + pixel_recall > 0 else 0
                
                print(f"\nPixel-level metrics:")
                print(f"  Precision: {pixel_precision:.3f}")
                print(f"  Recall: {pixel_recall:.3f}")
                print(f"  F1-score: {pixel_f1:.3f}")
                
                try:
                    fpr, tpr, _ = roc_curve(ucsd_test.y, ucsd_test.y_pred, pos_label=1)
                    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                    print(f"  EER: {eer:.3f}")
                    print(f"  Detection Rate: {1-eer:.3f}")
                except:
                    pass
        
        print(f"\n🎯 Average confidence: {np.mean(ucsd_test.confidence_scores):.3f}")
        print(f"📁 Results saved to: output_folders_optimized/")
        print("="*55)
            
    except Exception as e:
        print(f"Fatal error: {e}")