"""
Simple test script to verify the anomaly detection system works
"""
import sys
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join, isdir
import random
from sklearn.tree import DecisionTreeClassifier
from UCSDped1 import TestVideoFile
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import model
import utils
import optical_flow
import shutil

def load_sample_features():
    """Load sample training features for testing"""
    x_train = []
    y_train = []
    
    # Normal patterns (label 0)
    for i in range(20):
        normal_pattern = [10+i%5, 5+i%3, 3+i%2, 2, 1, 2, 3, 5+i%3, 8+i%4, 25+i%10, 0.02+i*0.001]
        x_train.append(normal_pattern)
        y_train.append(0)
    
    # Anomalous patterns (label 1)
    for i in range(10):
        anomaly_pattern = [2+i%3, 1, 0, 0, 0, 0, 1, 2+i%2, 25+i%5, 45+i%15, 0.08+i*0.005]
        x_train.append(anomaly_pattern)
        y_train.append(1)
    
    return x_train, np.array(y_train)

def main():
    print("🎯 Testing Crowd Anomaly Detection System")
    print("=" * 50)
    
    # Create sample training data
    print("📊 Loading training features...")
    x_train, y_train = load_sample_features()
    print(f"   ✅ Loaded {len(x_train)} training samples")
    
    # Train classifier
    print("🧠 Training classifier...")
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train)
    print("   ✅ Classifier trained successfully")
    
    # Test prediction
    print("🔍 Testing prediction...")
    test_normal = [12, 8, 4, 3, 2, 1, 2, 4, 6, 30, 0.025]
    test_anomaly = [1, 0, 0, 0, 0, 0, 0, 1, 30, 50, 0.095]
    
    pred_normal = clf.predict([test_normal])
    pred_anomaly = clf.predict([test_anomaly])
    
    print(f"   Normal pattern prediction: {pred_normal[0]} (should be 0)")
    print(f"   Anomaly pattern prediction: {pred_anomaly[0]} (should be 1)")
    
    # Check dataset path
    dataset_path = 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/'
    print(f"📁 Checking dataset path: {dataset_path}")
    
    if os.path.exists(dataset_path):
        print("   ✅ Dataset found!")
        
        # List available test videos
        dir_test = [f for f in listdir(dataset_path) if isdir(join(dataset_path, f)) and not f.endswith('_gt')]
        print(f"   📺 Found {len(dir_test)} test videos: {dir_test[:5]}...")
        
        # Test with first video
        if dir_test:
            test_video = dir_test[0]
            video_path = os.path.join(dataset_path, test_video)
            print(f"   🎬 Testing with video: {test_video}")
            
            # Check frames
            frames = [f for f in listdir(video_path) if f.endswith('.tif')]
            print(f"   🖼️  Found {len(frames)} frames")
            
            if frames:
                # Load first frame
                first_frame_path = os.path.join(video_path, frames[0])
                frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
                
                if frame is not None:
                    print(f"   ✅ Successfully loaded frame: {frame.shape}")
                    
                    # Test optical flow computation
                    if len(frames) > 1:
                        second_frame_path = os.path.join(video_path, frames[1])
                        frame2 = cv2.imread(second_frame_path, cv2.IMREAD_GRAYSCALE)
                        
                        if frame2 is not None:
                            print("   🌊 Computing optical flow...")
                            flow = cv2.calcOpticalFlowFarneback(frame, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            print(f"   ✅ Optical flow computed: {flow.shape}")
                            
                            # Test feature extraction
                            fx, fy = flow[:,:,0], flow[:,:,1]
                            magnitude = np.sqrt(fx*fx + fy*fy)
                            print(f"   📈 Flow magnitude range: {magnitude.min():.4f} - {magnitude.max():.4f}")
                            
                else:
                    print("   ❌ Could not load frame")
            else:
                print("   ❌ No frames found")
        else:
            print("   ❌ No test videos found")
    else:
        print("   ❌ Dataset not found!")
        print("   💡 Expected structure:")
        print("      UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001/*.tif")
    
    # Test features directory
    print(f"📂 Checking features directory...")
    if os.path.exists('features/'):
        feature_files = [f for f in listdir('features/') if f.startswith('features_test_UCSDped1')]
        print(f"   ✅ Features directory found with {len(feature_files)} files")
    else:
        print("   ❌ Features directory not found")
    
    print("\n🎉 System test completed!")
    print("=" * 50)

if __name__ == '__main__':
    main()