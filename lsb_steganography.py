import cv2
import numpy as np
import random

img1 = cv2.imread('path')
cv2.imshow('img1',img1)
cv2.waitKey(0)
img2 = cv2.imread('path',flags=0)
cv2.imshow('img2',img2)
cv2.waitKey(0)
img3=img1.copy()
img1w,img1h,img1c=img1.shape
img2w,img2h=img2.shape

np.random.seed(6)#harbar in meghdar ra begirad shauffel hamin tartib sabet ra midahad
img1pxl=[]
for i in range(img1w):
    for j in range(img1h) :
        for k in range(img1c):
            img1pxl+=[(i,j,k)]
np.random.shuffle(img1pxl)
needpxl=img1pxl[:img2w * img2h * 8]

if img2w*img2h*8>img1w*img1h*img1c :
    print("secret message is too big")
    exit(1)

count = 0
for i in range(img2.shape[0]): #width
    for j in range(img2.shape[1]): #height
        v2 = format(img2[i][j], '08b') #8bit aval
        for t in v2:
            x,y,z=needpxl[count]
            count += 1
            v1 = format(img1[x][y][z], '08b')
            if (v1[-1] == 1 and t == '0') or (v1[-1] == 0 and t == '1'):
                v3 = v1[:7]+ t
                img3[x][y][z]= int(v3, 2)
                
cv2.imwrite('includemessage.png', img3)
cv2.imshow('imgnew',img3)
cv2.waitKey(0)

#decode-> estekhraj bits
secret=""
for i in range(img2w * img2h * 8):
    x,y,z=needpxl[count]
    secret+=str(img3[x][y][z]%2)
#tabdil be ax?