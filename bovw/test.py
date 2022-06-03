import sys
import numpy as np
import cv2
import os
from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.cluster import kmeans_plusplus
from libsvm.svmutil import *
from libsvm.svm import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

voc_cnt = 2000
# #trainset_path = 'D:\data_cross\10\train'
# testset_path = 'TestSet'
# k_range = [200, 300, 500, 700, 1000, 1500, 2000]
# cross_validation_range = [0, 85, 170, 255, 340, 425, 510, 595, 680, 765, 858]

# for i in range(1,11):
# trainset_path = "D:\data_cross\\" + str(i) + "\\" +"train"
def calcSiftFeature(img):
    image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    gray = cv2.cvtColor(image8bit, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT()
    #135210
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 550,contrastThreshold = 0.021)
    # sift = cv2.xfeatures2d.SIFT_create()
    kps, des= sift.detectAndCompute(gray, None)
    #136585
    # surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 680, nOctaves = 4, nOctaveLayers = 3)
    # surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 660)
    # kps, des = surf.detectAndCompute(gray,None)
    return des


def calcImageFeature(des, centers):
    feature = np.zeros((1,voc_cnt), dtype=np.float32)
    words,distance = vq(des, centers)
    for i in words:
        feature[0][i] += 1
    return feature


# def calcFeatureSet():
# dirs = os.listdir(trainset_path)
deses = np.zeros((0,128), dtype=np.float32)
#deses = np.zeros((0,64), dtype=np.float32)
train_npy = np.load('D:\\592\\image_classification_using_sift\\cancer\\X_train.npy')
img_cnt = 0
# for f in range(858):
#     print('extract', dir, 'sift feature')
#     # files = os.listdir(os.path.join(trainset_path, dir))


for f in range(0,858):
    img_cnt += 1
    im = train_npy[f]
    des = calcSiftFeature(im)
    if des is not None:
        deses = np.append(deses, des, axis=0)
print(img_cnt, 'images extract', deses.shape[0], 'sift features')
np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_final"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 85):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(170, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv2"  + ".npy", deses)
    


# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 170):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(255, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv3"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 255):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(340, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv4"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 340):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(425, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv5"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 425):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(510, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv6"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 510):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(595, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv7"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 595):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(680, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv8"  + ".npy", deses)




# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 680):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)
# for f in range(765, 858):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv9"  + ".npy", deses)



# deses = np.zeros((0,128), dtype=np.float32)
# img_cnt = 0
# for f in range(0, 765):
#     img_cnt += 1
#     im = train_npy[f]
#     des = calcSiftFeature(im)
#     if des is not None:
#         deses = np.append(deses, des, axis=0)

# print(img_cnt, 'images extract', deses.shape[0], 'sift features')
# np.save("D:\\592\\image_classification_using_sift\\cancer\\sift\\normal\\"+ "sift_cv10"  + ".npy", deses)






















# for k in k_range:
# # def calcVoc():
#     deses = np.load("Temp/surf_add/train_surf_add_features"+ "cross" + str(i) + ".npy")
#     #criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
#     print('begin kmeans cluster')
#     #flags = cv2.KMEANS_RANDOM_CENTERS
#     #_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
#     #centers, variance = kmeans(deses, voc_cnt, 1) 
#     # kmeans = KMeans(n_clusters=100, max_iter=300).fit(deses)
#     # centers = kmeans.cluster_centers_
#     centers, indices = kmeans_plusplus(deses, n_clusters=k)
#     print('kmeans cluster done')
#     np.save("Temp/surf_add/surf_add_kmean"+ "cross" + str(i) + "k" + str(k) + ".npy", centers)