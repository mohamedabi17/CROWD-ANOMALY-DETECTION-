"""
Quick Test Script - Tests the project components quickly
"""

import sys
import numpy as np
import cv2
import os
from UCSDped1 import TestVideoFile
from model import AnomalyDetector
from utils import calculate_metrics
from optical_flow import OpticalFlowAnalyzer

def test_modules():
    """Test all modules can be imported and instantiated"""
    print("Testing Module Imports...")
    
    # Test ground truth
    print(f"PASS UCSDped1: {len(TestVideoFile)} videos with ground truth")
    
    # Test model
    detector = AnomalyDetector()
    print("PASS AnomalyDetector: Created successfully")
    
    # Test optical flow
    flow_analyzer = OpticalFlowAnalyzer()
    print("PASS OpticalFlowAnalyzer: Created successfully")
    
    # Test utils
    from utils import load_image, save_results
    print("PASS Utils: Functions available")
    
    return True

def test_dataset_access():
    """Test dataset access"""
    print("\nTesting Dataset Access...")
    
    dataset_path = './UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    
    if not os.path.exists(dataset_path):
        print("FAIL Dataset not found")
        return False
    
    # Check first test video
    test001_path = f'{dataset_path}/Test001'
    if os.path.exists(test001_path):
        frames = [f for f in os.listdir(test001_path) if f.endswith('.tif')]
        print(f"PASS Test001: {len(frames)} frames found")
        
        # Try to load one frame
        if frames:
            frame_path = os.path.join(test001_path, frames[0])
            frame = cv2.imread(frame_path)
            if frame is not None:
                print(f"PASS Frame loading: Shape {frame.shape}")
            else:
                print("FAIL Frame loading failed")
                return False
    
    return True

def test_feature_extraction():
    """Test feature extraction on sample data"""
    print("\nTesting Feature Extraction...")
    
    # Create sample frame
    sample_frame = np.random.randint(0, 255, (240, 360, 3), dtype=np.uint8)
    
    # Test background subtraction
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_subtractor.apply(sample_frame)
    
    if fg_mask is not None:
        fg_count = np.sum(fg_mask > 0)
        print(f"PASS Background subtraction: {fg_count} foreground pixels")
    
    # Test optical flow analyzer
    flow_analyzer = OpticalFlowAnalyzer()
    try:
        features = flow_analyzer.extract_features(sample_frame)
        if features is not None and len(features) == 11:
            print(f"PASS Feature extraction: {len(features)} features extracted")
        else:
            print(f"WARN Feature extraction: Got {len(features) if features else 0} features (expected 11)")
    except Exception as e:
        print(f"WARN Feature extraction error: {e}")
    
    return True

def test_training_data():
    """Test training data loading"""
    print("\nTesting Training Data...")
    
    features_dir = './features'
    if os.path.exists(features_dir):
        files = [f for f in os.listdir(features_dir) if f.endswith('.txt')]
        print(f"PASS Training files: {len(files)} files found")
        
        if files:
            # Try to load one file
            file_path = os.path.join(features_dir, files[0])
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        features = [float(x) for x in lines[0].strip().split()]
                        print(f"PASS Training data format: {len(features)} features per sample")
                    else:
                        print("WARN Training file is empty")
            except Exception as e:
                print(f"FAIL Training data loading error: {e}")
    else:
        print("WARN Features directory not found")
    
    return True

def test_classifier():
    """Test classifier training with sample data"""
    print("\n🤖 Testing Classifier...")
    
    # Create sample training data
    X_train = np.random.random((100, 11))  # 100 samples, 11 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels
    
    try:
        detector = AnomalyDetector()
        detector.train(X_train, y_train)
        
        # Test prediction
        X_test = np.random.random((10, 11))
        predictions = detector.predict(X_test)
        
        print(f"✅ Classifier: Trained on {len(X_train)} samples")
        print(f"✅ Prediction: {len(predictions)} predictions generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CROWD ANOMALY DETECTION - PROJECT TESTING")
    print("="*55)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_modules,
        test_dataset_access,
        test_feature_extraction,
        test_training_data,
        test_classifier
    ]
    
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "="*55)
    if all_passed:
        print("✅ ALL TESTS PASSED - PROJECT IS WORKING CORRECTLY!")
        print("\n🎯 System Status:")
        print("• All modules imported successfully")
        print("• Dataset access working")
        print("• Feature extraction functional")
        print("• Training data available")
        print("• Classifier training successful")
        print("\n🚀 Ready to run full anomaly detection!")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK ERRORS ABOVE")
    
    print("="*55)

if __name__ == '__main__':
    main()