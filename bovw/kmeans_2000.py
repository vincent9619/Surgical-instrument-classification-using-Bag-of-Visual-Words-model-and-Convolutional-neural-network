import sys
#sys.path.insert(0,'/home/zxy/opencv/opencv-2.4.13/build/lib')
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

voc_cnt = 1500
#trainset_path = 'D:\data_cross\10\train'
testset_path = 'TestSet'
k_range = [200, 300, 500, 700, 1000]

for i in range(1,11):
    # trainset_path = "D:\data_cross\\" + str(i) + "\\" +"train"
    # def calcSiftFeature(img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # sift = cv2.SIFT()
    #     #135210
    #     # sift = cv2.xfeatures2d.SIFT_create()
    #     # kps, des= sift.detectAndCompute(gray, None)
    #     #136585
    #     # surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 680, nOctaves = 4, nOctaveLayers = 3)
    #     surf = cv2.xfeatures2d.SURF_create()
    #     kps, des = surf.detectAndCompute(gray,None)
    #     return des


    # def calcImageFeature(des, centers):
    #     feature = np.zeros((1,voc_cnt), dtype=np.float32)
    #     words,distance = vq(des, centers)
    #     for i in words:
    #         feature[0][i] += 1
    #     return feature


    # # def calcFeatureSet():
    # dirs = os.listdir(trainset_path)
    # #deses = np.zeros((0,128), dtype=np.float32)
    # deses = np.zeros((0,64), dtype=np.float32)
    # img_cnt = 0
    # for dir in dirs:
    #     print('extract', dir, 'sift feature')
    #     files = os.listdir(os.path.join(trainset_path, dir))
    #     for f in files:
    #         img_cnt += 1
    #         im = cv2.imread(os.path.join(trainset_path, dir, f))
    #         des = calcSiftFeature(im)
    #         if des is not None:
    #             deses = np.append(deses, des, axis=0)
    # print(img_cnt, 'images extract', deses.shape[0], 'sift features')
    # np.save("Temp/train_surf_features"+ "cross" + str(i) + ".npy", deses)
    

    deses = np.load("Temp/train_sift_features"+ "cross" + str(i) + ".npy")
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
    print('begin kmeans cluster')
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
    #centers, variance = kmeans(deses, voc_cnt, 1) 
    # kmeans = KMeans(n_clusters=100, max_iter=300).fit(deses)
    # centers = kmeans.cluster_centers_
    centers, indices = kmeans_plusplus(deses, n_clusters=1500)
    print('kmeans cluster done')
    np.save("Temp/sift_kmean"+ "cross" + str(i) + "k1500"  + ".npy", centers)