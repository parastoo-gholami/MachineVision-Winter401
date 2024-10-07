import cv2
import math
import numpy as np
import scipy.stats as st

img = cv2.imread('C:\\Users\\iranian\\Desktop\\b.jpg')


def zone(x, y):
    return 0.5 * (1 + math.cos(x * x + y * y))

SIZE = 597
img = np.zeros((SIZE, SIZE))

start = -8.2
end = 8.2
step = 0.0275

def dist_center(y, x):
    global SIZE
    center = SIZE / 2
    return math.sqrt( (x - center)**2 + (y - center)**2)

for y in range(0, SIZE):
    for x in range(0, SIZE):
        if dist_center(y, x) > 300:
            continue
        y_val = start + y * step
        x_val = start + x * step
        img[y, x] = zone(x_val, y_val)

def gkern(kernlen, nsig):

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

kernel_size = 15

lowpass_kernel_gaussian = gkern(kernel_size , 1)
lowpass_kernel_gaussian = lowpass_kernel_gaussian / lowpass_kernel_gaussian.sum()

lowpass_kernel_box = np.ones((kernel_size, kernel_size))
lowpass_kernel_box = lowpass_kernel_box / (kernel_size * kernel_size)

lowpass_image_gaussian = cv2.filter2D(img, -1, lowpass_kernel_gaussian)
lowpass_image_box = cv2.filter2D(img, -1, lowpass_kernel_box)


highpass_image_gaussian = img - lowpass_image_gaussian
highpass_image_gaussian = np.absolute(highpass_image_gaussian)

highpass_image_box = img - lowpass_image_box
highpass_image_box = np.absolute(highpass_image_box)

bandreject_image = lowpass_image_gaussian + highpass_image_box

bandpass_image = img - bandreject_image
bandpass_image = np.absolute(bandpass_image)


cv2.imshow('filter_image',bandpass_image)
cv2.waitKey(0)