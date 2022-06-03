import numpy as np
import cv2
import sys
import os


img = cv2.imread("seg.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 100, nOctaves = 4, nOctaveLayers = 3)
keypoints1,descriptor1 = surf.detectAndCompute(gray,None)
image1 = cv2.drawKeypoints(gray,keypoints1,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow(‘surf_keypoints1’,image1)
# img = cv2.drawKeypoints(gray , kp , img , flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("seg" + "surf.jpg", image1)



# surf = cv2.xfeatures2d.SURF_create(25000)
# keypoints1,descriptor1 = surf.detectAndCompute(image1,None)
# keypoints2,descriptor2 = surf.detectAndCompute(image2,None)
# print(‘descriptor1:’,descriptor1.shape,‘descriptor2’,descriptor2.shape)
# image1 = cv2.drawKeypoints(image=image1,keypoints = keypoints1,outImage=image1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# image2 = cv2.drawKeypoints(image=image2,keypoints = keypoints2,outImage=image2,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow(‘surf_keypoints1’,image1)
# cv2.imshow(‘surf_keypoints2’,image2)
# cv2.waitKey(20)