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



counter = 1

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def load_train_features(type):
    x_train = []
    y_train = []
    
    # Check if features directory exists
    if not os.path.exists('features/'):
        print("Warning: features/ directory not found. Creating sample features...")
        os.makedirs('features/', exist_ok=True)
        # Generate some sample training data
        sample_features = []
        # Normal patterns (label 0)
        for i in range(50):
            # Features: 9 direction bins + fg_count + avg_mag + i + j + label
            normal_pattern = [10+i%5, 5+i%3, 3+i%2, 2, 1, 2, 3, 5+i%3, 8+i%4, 25+i%10, 0.02+i*0.001, i*2, i*3, 0]
            sample_features.append(' '.join(map(str, normal_pattern)))
        
        # Anomalous patterns (label 1)
        for i in range(25):
            # Anomalous: different direction distribution, higher magnitude
            anomaly_pattern = [2+i%3, 1, 0, 0, 0, 0, 1, 2+i%2, 25+i%5, 45+i%15, 0.08+i*0.005, i*2, i*3, 1]
            sample_features.append(' '.join(map(str, anomaly_pattern)))
        
        # Save sample features
        with open('features/features_test_' + type + '_sample.txt', 'w') as f:
            f.write('\n'.join(sample_features))
    
    # Load features
    try:
        features = [f for f in listdir('features/') if f.startswith("features_test_"+type)]
        if not features:
            print(f"Warning: No feature files found for type '{type}'. Using default training data...")
            # Create minimal training data
            x_train = [[10, 5, 3, 2, 1, 2, 3, 5, 8, 25, 0.02] for _ in range(10)]
            x_train.extend([[2, 1, 0, 0, 0, 0, 1, 2, 25, 45, 0.08] for _ in range(5)])
            y_train = [0] * 10 + [1] * 5
            return x_train, np.array(y_train)
            
        for feature in features:
            file = open('features/' + feature, "r")
            feature_text = file.read().split("\n")
            for f in feature_text:
                if f != "" and not f.startswith("#"):  # Skip comments and empty lines
                    try:
                        feat_parts = f.split(" ")
                        if len(feat_parts) >= 14:  # Ensure we have enough features
                            feat_all = [float(feat) for feat in feat_parts if feat]
                            if len(feat_all) >= 14:
                                x_train.append(feat_all[:-1])  # All features except label
                                y_train.append(int(feat_all[-1]))  # Label
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed line: {f}")
                        continue
            file.close()
    except Exception as e:
        print(f"Error loading features: {e}")
        # Return minimal default data
        x_train = [[10, 5, 3, 2, 1, 2, 3, 5, 8, 25, 0.02, 0, 0] for _ in range(10)]
        x_train.extend([[2, 1, 0, 0, 0, 0, 1, 2, 25, 45, 0.08, 0, 0] for _ in range(5)])
        y_train = [0] * 10 + [1] * 5

    return x_train, np.array(y_train)


class UCSDTest:
    def __init__(self, path, n, detect_interval, type):
        self.path = path
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.n = n
        self.detect_interval = detect_interval
        # self.classifier = VideoClassifier()
        self.clf = DecisionTreeClassifier(max_depth=5)
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"Warning: Dataset path '{path}' not found!")
            print("Please ensure the UCSD dataset is in the correct location.")
            print("Expected structure: UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/")
            # Create a dummy classifier with minimal data
            x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.1, 0, 0]]
            y = [0]
        else:
            x, y = load_train_features(type)
            
        if len(x) == 0:
            print("Warning: No training features loaded. Using default data.")
            x = [[10, 5, 3, 2, 1, 2, 3, 5, 8, 25, 0.02, 0, 0] for _ in range(10)]
            x.extend([[2, 1, 0, 0, 0, 0, 1, 2, 25, 45, 0.08, 0, 0] for _ in range(5)])
            y = [0] * 10 + [1] * 5
            
        self.clf.fit(x, y)
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.should_find = 0.0
        self.total = 0.0
        self.y = []
        self.y_pred = []

    def process_frame(self, bins, magnitude, fmask, tag_img, frame):
        if np.count_nonzero(fmask) == 0:
            return False
        bin_count = np.zeros(9, np.uint8)
        h,w = bins.shape
        found_anomaly = False
        features_j = []
        tag_j = []
        index_i_j = []
        for i in range(0, h, self.n):
            i_end = min(h, i+self.n)
            if np.count_nonzero(fmask[i]) > 0:
                for j in range(0, w, self.n):
                    j_end = min(w, j+self.n)
                    if np.count_nonzero(fmask[i:i_end, j:j_end]) > 0:
                        # Get the atom for bins
                        atom_bins = bins[i:i_end, j:j_end].flatten()

                        # Average magnitude
                        atom_mag = magnitude[i:i_end, j:j_end].flatten().mean()
                        atom_fmask = fmask[i:i_end, j:j_end].flatten()

                        # Count of foreground values
                        f_cnt = np.count_nonzero(atom_fmask)
                        f_cnt_2 = np.count_nonzero(fmask[i:i_end, j:j_end].flatten())

                        # Get the direction bins values
                        hs, _ = np.histogram(atom_bins, np.arange(10))
                        features = hs.tolist()
                        features.extend([f_cnt, f_cnt_2, atom_mag, i, i+self.n, j, j+self.n])
                        features_j.append(features)
                        # vector = np.array(features)
                        tag_atom = tag_img[i:i_end, j:j_end].flatten()
                        ones = np.count_nonzero(tag_atom)
                        # if ones > 20:
                        #     tag = 1
                        # else:
                        #     tag = 0
                        tag = 1
                        if ones < 50:
                            tag = 0
                        tag_j.append(tag)
                        index_i_j.append((i,j))
        predicted = self.clf.predict(features_j)
        self.y_pred.extend(predicted)
        self.y.extend(tag_j)
        self.total += len(predicted)
        for index, pred in enumerate(predicted):
            pred = pred.item()
            i, j = index_i_j[index]
            if pred == 1:
                if tag_j[index] == 0:
                    self.false_positive += 1
                else:
                    self.true_positive += 1
                j_end = min(w, j+self.n)
                i_end = min(h, i+self.n)
                cv2.rectangle(frame, (j, i), (j_end, i_end), (255, 255, 0), 2)
                found_anomaly = True
            elif tag_j[index] == 1:
                self.false_negative += 1
        return found_anomaly

    def process_video(self, video_name, tag_video):
        global counter 
        mag_threshold=1e-3
        elements = 0
        files = [f for f in listdir(self.path+video_name) if isfile(join(self.path+video_name, f))]
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        if '._.DS_Store' in files:
            files.remove('._.DS_Store')
        files_tag = [f for f in listdir(self.path+tag_video) if isfile(join(self.path+tag_video, f))]
        if '.DS_Store' in files_tag:
            files_tag.remove('.DS_Store')
        if '._.DS_Store' in files_tag:
            files_tag.remove('._.DS_Store')
        files_tag.sort()
        files.sort()
        number_frame = 0
        old_frame = None
        mots = []
        old_frame = cv2.imread(self.path + video_name + '001.tif', cv2.IMREAD_GRAYSCALE)
        width = old_frame.shape[0]
        height = old_frame.shape[1]
        h, w = old_frame.shape[:2]
        bins = np.zeros((h, w, self.detect_interval), np.uint8)
        mag = np.zeros((h, w, self.detect_interval), np.float32)
        fmask = np.zeros((h, w, self.detect_interval), np.uint8)
        frames = np.zeros((h, w, self.detect_interval), np.uint8)
        tag_img = np.zeros((h,w,self.n), np.uint8)
        anomaly_detected = []
        path_folder_out = ('output_folders/out'+str(counter))
        os.mkdir(path_folder_out)
        for tif in files:
            movement = 0
            frame = cv2.imread(self.path + video_name + tif, cv2.IMREAD_GRAYSCALE)
            if number_frame % self.detect_interval == 0:
                fmask = self.fgbg.apply(frame)
                flow = cv2.calcOpticalFlowFarneback(old_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                tag_img = cv2.imread(self.path + tag_video + files_tag[number_frame] ,cv2.IMREAD_GRAYSCALE)
                # Calculate direction and magnitude
                height, width = flow.shape[:2]
                fx, fy = flow[:,:,0], flow[:,:,1]
                angle = ((np.arctan2(fy, fx+1) + 2*np.pi)*180)% 360
                binno = np.ceil(angle/45)
                magnitude = np.sqrt(fx*fx+fy*fy)
                binno[magnitude < mag_threshold] = 0
                bins = binno
                mag = magnitude
                found_anomaly = self.process_frame(bins, mag, fmask, tag_img, frame)
                if found_anomaly:
                    anomaly_detected.append(number_frame)
            cv2.imwrite((path_folder_out+'/frame'+str(number_frame)+'.jpg'),frame)
        
            number_frame += 1
            old_frame = frame
        counter+=1
        return anomaly_detected

if __name__ == '__main__':
    shutil.rmtree("output_folders", ignore_errors = True)
    ucsdped = 'UCSDped1'
    ucsd_test = UCSDTest('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', 10, 5, ucsdped)
    dir_test = [f for f in listdir('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/') if isdir(join('UCSD_Anomaly_Dataset.v1p2/'+ucsdped+'/Test/', f))]
    dir_test.sort()
    total_correct = 0.0
    total_should_found = 0.0
    total_found = 0.0
    
    os.mkdir('output_folders')
    for directory in dir_test:
        if not directory.endswith("gt"):
            start_time = time.time()
            
            anomaly_detected = ucsd_test.process_video(directory+'/', directory + '_gt/')
            time_video = passed_time(start_time)
            print(200.0/time_video, "frames per second")
            total_found += len(anomaly_detected)
            index_video = int(directory[-3:])
            total_correct += len(set(anomaly_detected).intersection(TestVideoFile[index_video]))
            total_should_found += len(TestVideoFile[index_video])

    precision = total_correct/total_found
    recall = total_correct/total_should_found
    f1 = 2.0*precision*recall/(precision+recall)
    print("Resultats par image :")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    print("Resultats par pixel:")
    pixel_true_positive = ucsd_test.true_positive
    pixel_false_positive = ucsd_test.false_positive
    pixel_false_negative = ucsd_test.false_negative
    pixel_total = ucsd_test.total
    fpr, tpr, threshold = roc_curve(ucsd_test.y, ucsd_test.y_pred, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print("EER by pixel")
    print(eer)
    print("RD by pixel")
    print(1-eer)
    print("True positive: ", pixel_true_positive)
    print("False positive: ", pixel_false_positive)
    precision = pixel_true_positive/(pixel_true_positive + pixel_false_positive)
    recall = pixel_true_positive/(pixel_true_positive + pixel_false_negative)
    f1 = 2.0*precision*recall/(precision+recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)    
