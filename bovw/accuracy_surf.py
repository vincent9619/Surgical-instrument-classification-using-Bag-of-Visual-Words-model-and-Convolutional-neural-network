
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

record = []
voc_cnt = 2000


def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT()
    #135210
    # sift = cv2.xfeatures2d.SIFT_create()
    # kps, des= sift.detectAndCompute(gray, None)
    #136585
    # surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 680, nOctaves = 4, nOctaveLayers = 3)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 335)
    kps, des = surf.detectAndCompute(gray,None)
    return des


def calcImageFeature(des, centers):
    feature = np.zeros((1,voc_cnt), dtype=np.float32)
    words,distance = vq(des, centers)
    for i in words:
        feature[0][i] += 1
    return feature


# def calcFeatureSet():
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
# np.save('Temp/train_surf_features.npy', deses)
    
# def calcVoc():
for count in range(1,11):
    trainset_path = "D:\data_cross\\" + str(count) + "\\" +"train"
    testset_path = "D:\data_cross\\" + str(count) + "\\" +"test"
    #print(count)
    #deses = np.load("Temp/train_sift_features"+ "cross" + str(count) + ".npy")
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
    print('begin kmeans cluster')
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
    #centers, variance = kmeans(deses, voc_cnt, 1) 
    # kmeans = KMeans(n_clusters=100, max_iter=300).fit(deses)
    # centers = kmeans.cluster_centers_
    k_range = [200, 300, 500, 700, 1000, 1500, 2000]
    #k_range = [200, 300]
    norm = ["none", "zero_to_one_l2", "zero_to_one", "zero_or_one"]
    #norm = ["zero_to_one_l2", "none", "zero_to_one", "zero_or_one"]
    for nor in norm:
        for n in k_range:
            # centers, indices = kmeans_plusplus(deses, n_clusters=n)
            # print('kmeans cluster done')
            # np.save('Temp/surf'+'kmeans'+ str(n) +'.npy', centers)
                
            # idf = None

            # def trainClassfier():
            dirs = os.listdir(trainset_path)
            print('trainClassfier', dirs)
            centers = np.load("Temp/surf_double/surf_double_kmean"+ "cross" + str(count) + "k" + str(n) + ".npy")
            #centers = np.load('Temp/sift_kmeancross1k200.npy')
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
            print(features)


            if nor == "none":
                murasame_aya = 1
            elif nor == "zero_to_one_l2":
                features = preprocessing.normalize(features, norm='l2')
            elif nor == "zero_to_one":
                scaler = MinMaxScaler( )
                scaler.fit(features)
                features = scaler.transform(features)
            elif nor == "zero_or_one":
                features = np.float32(features>0)


            labels = labels.reshape((-1,1))
            #labels = labels.flatten()
            print(labels.shape)
            print("第2次feature")
            print(features.shape)

            #random forest
            # clf = RandomForestClassifier(max_depth=398,random_state=69,n_estimators=150,min_samples_leaf=1,min_samples_split=2)
            # clf = RandomForestClassifier(n_estimators=350, max_depth=650, min_samples_leaf=1, min_samples_split=2 ,random_state=51)
            # clf = RandomForestClassifier()
            # clf.fit(features, labels)
            #train_X,test_X,train_y,test_y = train_test_split(features, labels,test_size=1/5,random_state=3)

            #svc
            #clf = SVC(C=15, gamma='auto', kernel='linear')
            clf = SVC(C=15, gamma='auto', kernel='linear')
            clf.fit(features, labels)



            # total = 0; correct = 0; dictIdx = 0
            dirt = os.listdir(testset_path)
            print('start testing')
            testlabels = np.float32([])
            testId = 0
            testresults = np.float32([])
            target_names = ['6_Babcock_Tissue_Forceps', '6_Mayo_Needle_Holder', '7_Metzenbaum_Scissors', '7_Microvascular_Needle_Holder', '8_Babcock_Tissue_Forceps', '8_Mayo_Needle_Holder', '8_Microvascular_Needle_Holder', '9_DeBakey_Dissector', '9_DeBakey_Needle_Holder', '9_Metzenbaum_Scissors', 'Allis_Tissue_Forceps', 'Ball___Socket_Towel_Clips', 'Bonneys_Non_Toothed_Dissector', 'Bonneys_Toothed_Dissector', 'Crile_Artery_Forceps', 'Curved_Mayo_Scissors', 'Dressing_Scissors', 'Gillies_Toothed_Dissector', 'Lahey_Forceps', 'Littlewood_Tissue_Forceps', 'Mayo_Artery_Forceps', 'No3_BP_Handles', 'No4_BP_Handles', 'No7_BP_Handles', 'Sponge_Forceps']

            for dir in dirt:
                files = os.listdir(os.path.join(testset_path, dir))
                for f in files:
                    im = cv2.imread(os.path.join(testset_path,dir,f))
                    des = calcSiftFeature(im)
                    feature = calcImageFeature(des, centers)
                    feature = feature*idf
                    if nor == "none":
                        murasame_aya = 1
                    elif nor == "zero_to_one_l2":
                        feature = preprocessing.normalize(feature, norm='l2')
                    elif nor == "zero_to_one":
                        scaler = MinMaxScaler( )
                        scaler.fit(features)
                        feature = scaler.transform(feature)
                    elif nor == "zero_or_one":
                        feature = np.float32(feature>0)
                    #feature = preprocessing.normalize(feature, norm='l2')
                    testlabels = np.append(testlabels, np.float32(testId))
                    #p_label, p_acc, p_val = svm_predict(feature, model)
                    #print(svm.predict(feature))
                    #print(p_label)  
                    testresults = np.append(testresults, np.float32(clf.predict(feature)))
                    #print(int(clf.predict(feature)))
                testId += 1
            print("testresults")
            print(testresults)
            print(testresults.shape)
            print("testlabels")
            print(testlabels)
            print(testlabels.shape)
            print(classification_report(testlabels, testresults, target_names=target_names))
            f = open("I:\image_classification_using_sift\manual_cross\surf_double\\" + "cross" + str(count)+"surf_double_svm_15_record.txt",'a', encoding='utf-8')
            space = "********************************************************\n"
            mes = "surf" + "cross" + str(count) + " " + "k" + str(n) + " " + "norm: " + str(nor) + "\n"
            f.writelines(space)
            f.writelines(mes)
            f.writelines(classification_report(testlabels, testresults, target_names=target_names))
            f.close()



 
# calcFeatureSet()
# calcVoc()
# trainClassfier()