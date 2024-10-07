import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

def HOG_(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray, orientations=30, pixels_per_cell=(6, 6), cells_per_block=(1, 1), block_norm='L2-Hys',transform_sqrt=True)
    return hog_feature
def LBP_(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feature=local_binary_pattern(gray,8,1)
    hist_lbp, _ = np.histogram(lbp_feature, bins=int(lbp_feature.max() + 1), range=(0,int(lbp_feature.max() + 1)),density=True)
    return hist_lbp

def GLCM_(image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    glcm = np.zeros((256, 256), dtype=np.uint8)
    height, width = gray.shape
    for i in range(height - 1):
        for j in range(width - 1):
            p = gray[i, j]
            q = gray[i, j + 1]
            glcm[p, q] += 1

    glcm = glcm.astype(np.float64)
    glcm /= np.sum(glcm)
    contrast = np.sum(np.square(glcm - np.mean(glcm)))
    dissimilarity = np.sum(np.abs(glcm - np.mean(glcm)))
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
    energy = np.sum(np.square(glcm))
    correlation = np.sum(np.divide((i - np.mean(glcm)) * (j - np.mean(glcm)), np.sqrt(np.sum(np.square(i - np.mean(glcm)))) * np.sqrt(np.sum(np.square(j - np.mean(glcm))))))
    glcm_feature = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    return glcm_feature

cap=cv2.VideoCapture("C:\\Users\\ASUS\\Desktop\\FINALFINAL.mp4")
facedetector=cv2.CascadeClassifier('C:\\Users\\ASUS\\Desktop\\haarcascade_frontalface_default.xml ')
vwriter=cv2.VideoWriter('out.wmv',cv2.VideoWriter_fourcc(*'WMV1'),20,(640,480))
with open('C:\\Users\\ASUS\\Desktop\\project\\model.pkl','rb') as f:
    clf = pickle.load(f)
while(True):
    ret,frame=cap.read()
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facedetector.detectMultiScale(gray,1.1,4)
        for (x,y,w,h) in faces:
            frame_detector = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
            hog_detect = []
            lbp_detect = []
            glcm_detect = []
            hog_detect.append(HOG_(frame_detector))
            lbp_detect.append(LBP_(frame_detector))
            glcm_detect.append(GLCM_(frame_detector))
            feature = np.concatenate((hog_detect, lbp_detect, glcm_detect), axis=1)
            ans=clf.predict(feature)
            if(ans[0]):
               cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        vwriter.write(frame)
        cv2.imshow('frame',frame)
        cv2.waitKey(10)
    else:
        vwriter.release()
cap.release()
