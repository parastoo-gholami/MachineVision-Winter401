from matplotlib import pyplot as plt
import numpy as np
import cv2


class histogram_equalizer:
    def __init__(self, img):
        self.A = img
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def equalize(self):
        self.hsv[:, :, 2] = cv2.equalizeHist(self.hsv[:, :, 2])
        img = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2RGB)
        return img


A = cv2.imread('C:\\Users\\iranian\\Desktop\\a.jpg')
img = histogram_equalizer(A)

ax1 = plt.subplot(2, 2, 1)
ax1.imshow(img.rgb)

ax2 = plt.subplot(2, 2, 2)
ax2.hist(A[:, :, 2].flatten(), 256, [0, 255], color='r')

eq = img.equalize()
ax3 = plt.subplot(2, 2, 3)
ax3.imshow(eq)

ax4 = plt.subplot(2, 2, 4)
ax4.hist(eq[:, :, :2].flatten(), 256, [0, 255], color='r')

plt.show()


