
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
import matplotlib.pyplot as plt


voc_cnt = 1000
trainset_path = 'I:\image_classification_using_sift\TrainSet'
testset_path = 'TestSet'


def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kps, des= sift.detectAndCompute(gray, None)
    return des


def calcImageFeature(des, centers):
    feature = np.zeros((1,voc_cnt), dtype=np.float32)
    words,distance = vq(des, centers)
    for i in words:
        feature[0][i] += 1
    return feature


# def calcFeatureSet():
# dirs = os.listdir(trainset_path)
# deses = np.zeros((0,128), dtype=np.float32)
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
#print(img_cnt, 'images extract', deses.shape[0], 'sift features')
#np.save('Temp/train_sift_features.npy', deses)

K=range(50,52,1)
#K = np.arange(0.1,1,0.1)
Total_Accuracy = []
for k in K:
    # def calcVoc():
    deses = np.load('Temp/sift_train_features.npy')
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
    print('begin kmeans cluster')
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
    #centers, variance = kmeans(deses, voc_cnt, 1) 
    #kmeans = KMeans(n_clusters=100, random_state=0).fit(deses)

    # centers, indices = kmeans_plusplus(deses, n_clusters=1000)
    print('kmeans cluster done')
    # np.save('Temp/voc.npy', centers)
        
    # idf = None

    # def trainClassfier():
    dirs = os.listdir(trainset_path)
    print('trainClassfier', dirs)
    centers = np.load('Temp/sift_kmeans.npy')
    features = np.zeros((0,voc_cnt), dtype=np.float32)
    labels = np.float32([])
    dictIdx = 0

    print('begin train classfier')
    for dir in dirs:
        files = os.listdir(os.path.join(trainset_path, dir))
        
        for f in files:
            im = cv2.imread(os.path.join(trainset_path, dir, f))
            des = calcSiftFeature(im)
            feature = calcImageFeature(des, centers)
            #也可以用这种表达方式features = np.vstack((features, features))
            features = np.append(features, feature, axis=0)
            
            #np.float32做类型转换，否则变成np.float64
            labels = np.append(labels, np.float32(dictIdx))
        dictIdx += 1
    #labels = labels.flatten()
    print("第一次feature")
    print(features.shape)
    print(labels)
    nbr_occurences = np.sum((features>0)*1, axis=0)
    idf = np.array(np.log((1.0*features.shape[0]+1)/(1.0*nbr_occurences+1)), dtype=np.float32)
    features = features*idf
    features = preprocessing.normalize(features, norm='l2')

    labels = labels.reshape((-1,1))
    #labels = labels.flatten()
    print(labels.shape)
    print("第2次feature")
    print(features.shape)
    #random forest
    #clf = RandomForestClassifier(max_depth=100, random_state=50,n_estimators=100,min_samples_leaf=10)
    #clf = RandomForestClassifier(max_depth=k,random_state=69,n_estimators=150,min_samples_leaf=1,min_samples_split=2)
    clf = RandomForestClassifier(n_estimators=350, max_depth=650, min_samples_leaf=1, min_samples_split=2, random_state=k)
    clf.fit(features, labels)

    #svm = cv2.SVM()
    # svm = cv2.ml.SVM_create()
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # #svm.setP(0.5)
    # svm.setType(cv2.ml.SVM_EPS_SVR)
    # svm.setC(0.01)
    # svm.train(features, cv2.ml.ROW_SAMPLE, labels)

    #print(labels.shape, labels.dtype, labels)
    # prob  = svm_problem(labels, features)
    # param = svm_parameter('-t 0 -c 4 -b 1')
    # model = svm_train(prob, param)
    # svm.train_auto(features, labels, None, None, None)
    # svm.train(features, cv2.ml.ROW_SAMPLE, labels)

    #     svm.save("svmV2.clf")
    #     print('train classfier Done!')

    # def classify():
    #     #svm = cv2.SVM()
    #     svm = cv2.ml.SVM_create()
    #     svm.load("svmV2.clf")
    #centers = np.load('Temp/voc.npy')

    total = 0; correct = 0; dictIdx = 0
    dirt = os.listdir(testset_path)
    print('start testing')
    print('classify', dirt)
    # for dir in dirs:
    #     count = 0; crt = 0
    #     files = os.listdir(os.path.join(testset_path, dir))
    #     for f in files:
    #         count += 1
    #         im = cv2.imread(os.path.join(testset_path,dir,f))
    #         des = calcSiftFeature(im)
    #         feature = calcImageFeature(des, centers)
    #         feature = feature*idf
    #         feature = preprocessing.normalize(feature, norm='l2')
    #         #feature = np.array(feature, dtype='float32')
    #         #svm_pre = svm.predict(feature)
    #         #print(svm_pre)
    #         # print(feature.shape, feature.dtype)
    #         # svm_pre = svm.predict(feature)
    #         # print(svm_pre)
    #         # print(svm_pre[1])
    #         # if int(dictIdx) == svm.predict(feature):
    #         print(svm.predict(feature))
    #         # if dictIdx == svm.predict(feature):
    #         #     print(count)
    #         #     crt += 1
    #     print('Accuracy Class', dir, crt, '/', count, '=',float(crt)/count)
    #     total += count
    #     correct += crt
    #     dictIdx += 1
    # print('Total Accuracy ', correct, '/', total, float(correct)/total)
    for dir in dirt:
        count = 0; crt = 0
        files = os.listdir(os.path.join(testset_path, dir))
        for f in files:
            count += 1
            im = cv2.imread(os.path.join(testset_path,dir,f))
            des = calcSiftFeature(im)
            feature = calcImageFeature(des, centers)
            feature = feature*idf
            feature = preprocessing.normalize(feature, norm='l2')
            #p_label, p_acc, p_val = svm_predict(feature, model)
            #print(svm.predict(feature))
            #print(p_label)
            print(int(clf.predict(feature)))
            if dictIdx == int(clf.predict(feature)):
                crt += 1
        print('Accuracy Class', dir, crt, '/', count, '=',float(crt)/count)
        total += count
        correct += crt
        dictIdx += 1
    print('Total Accuracy ', correct, '/', total, float(correct)/total)
    Total_Accuracy.append(float(correct)/total)
        
    # calcFeatureSet()
    # calcVoc()
    # trainClassfier()

plt.plot(K,Total_Accuracy,'gx-')
plt.xlabel('k')
plt.ylabel(u'y')
plt.title(u'random_state')
plt.show()