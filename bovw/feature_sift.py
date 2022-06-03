import numpy as np
import cv2
import sys
import os

img = cv2.imread("cancer.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray , None) 
img = cv2.drawKeypoints(gray , kp , img , flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("seg" + "sift.jpg", img)
