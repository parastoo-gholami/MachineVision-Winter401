import cv2
import numpy as np
import math

class edge_detection:
    def __init__(self,img):
        self.A=img
        self.F=np.zeros((3,3))
        self.F[0,:]=1
        self.F[2,:]=-1    
    def prewitth(self):
        #ofoghi
        B=cv2.filter2D(self.A,0,self.F)
        return B
    def prewittv(self):
        #amodi
        F2=self.F.T
        C=cv2.filter2D(self.A,0,F2)
        return C
    def prewitt(self):
        A=self.prewitth()
        B=self.prewittv()
        h,w=A.shape
        for i in range(h):
            for j in range(w):
                B[i][j]= math.sqrt(pow(B[i][j],2)+pow(A[i][j],2) )
        return B
        

    
A=cv2.imread('C:\\Users\\ASUS\\Desktop\\a.jpg',0)
img=edge_detection(A)
cv2.imshow('image',A)
cv2.waitKey(0)
cv2.imshow('image',img.prewitth())
cv2.waitKey(0)
cv2.imshow('image',img.prewittv())
cv2.waitKey(0)
cv2.imshow('image',img.prewitt())
cv2.waitKey(0)

