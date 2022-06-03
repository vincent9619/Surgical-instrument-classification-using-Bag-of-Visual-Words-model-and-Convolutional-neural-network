from sklearn.model_selection import train_test_split,cross_val_score,cross_validate # 交叉验证所需的函数
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit # 交叉验证所需的子集划分方法
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit # 分层分割
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,LeavePGroupsOut,GroupShuffleSplit # 分组分割
from sklearn.model_selection import TimeSeriesSplit # 时间序列分割
from sklearn import datasets  # 自带数据集
from sklearn import preprocessing  # 预处理模块
from sklearn.metrics import recall_score  # 模型度量
from sklearn.svm import SVC



#1742 images extract 219138 sift features
#1742 images extract 999044 sift features


 
# iris = datasets.load_iris()  # 加载数据集
# print('样本集大小：',iris.data.shape,iris.target.shape)
 
# # ===================================数据集划分,训练模型==========================
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)  # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
# print('训练集大小：',X_train.shape,y_train.shape)  # 训练集样本大小
# print('测试集大小：',X_test.shape,y_test.shape)  # 测试集样本大小
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) # 使用训练集训练模型
# print('准确率：',clf.score(X_test, y_test))  # 计算测试集的度量值（准确率）
 
 
# #  如果涉及到归一化，则在测试集上也要使用训练集模型提取的归一化函数。
# scaler = preprocessing.StandardScaler().fit(X_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
# X_train_transformed = scaler.transform(X_train)
# clf = svm.SVC(kernel='linear', C=1).fit(X_train_transformed, y_train) # 使用训练集训练模型
# X_test_transformed = scaler.transform(X_test)
# print(clf.score(X_test_transformed, y_test))  # 计算测试集的度量值（准确度）
 
# # ===================================直接调用交叉验证评估模型==========================
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)  #cv为迭代次数。
# print(scores)  # 打印输出每次迭代的度量值（准确度）
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
 
# # ===================================多种度量结果======================================
# scoring = ['precision_macro', 'recall_macro'] # precision_macro为精度，recall_macro为召回率
# scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,cv=5, return_train_score=True)
# sorted(scores.keys())
# print('测试结果：',scores)  # scores类型为字典。包含训练得分，拟合次数， score-times （得分次数）
 
 
# # ==================================K折交叉验证、留一交叉验证、留p交叉验证、随机排列交叉验证==========================================
# # k折划分子集
# kf = KFold(n_splits=2)
# for train, test in kf.split(iris.data):
#     print("k折划分：%s %s" % (train.shape, test.shape))
#     break
 
# # 留一划分子集
# loo = LeaveOneOut()
# for train, test in loo.split(iris.data):
#     print("留一划分：%s %s" % (train.shape, test.shape))
#     break
 
# # 留p划分子集
# lpo = LeavePOut(p=2)
# for train, test in loo.split(iris.data):
#     print("留p划分：%s %s" % (train.shape, test.shape))
#     break
 
# # 随机排列划分子集
# ss = ShuffleSplit(n_splits=3, test_size=0.25,random_state=0)
# for train_index, test_index in ss.split(iris.data):
#     print("随机排列划分：%s %s" % (train.shape, test.shape))
#     break
 
# # ==================================分层K折交叉验证、分层随机交叉验证==========================================
# skf = StratifiedKFold(n_splits=3)  #各个类别的比例大致和完整数据集中相同
# for train, test in skf.split(iris.data, iris.target):
#     print("分层K折划分：%s %s" % (train.shape, test.shape))
#     break
 
# skf = StratifiedShuffleSplit(n_splits=3)  # 划分中每个类的比例和完整数据集中的相同
# for train, test in skf.split(iris.data, iris.target):
#     print("分层随机划分：%s %s" % (train.shape, test.shape))
#     break
 
 
# # ==================================组 k-fold交叉验证、留一组交叉验证、留 P 组交叉验证、Group Shuffle Split==========================================
# X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
# y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
# groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
 
# # k折分组
# gkf = GroupKFold(n_splits=3)  # 训练集和测试集属于不同的组
# for train, test in gkf.split(X, y, groups=groups):
#     print("组 k-fold分割：%s %s" % (train, test))
 
# # 留一分组
# logo = LeaveOneGroupOut()
# for train, test in logo.split(X, y, groups=groups):
#     print("留一组分割：%s %s" % (train, test))
 
# # 留p分组
# lpgo = LeavePGroupsOut(n_groups=2)
# for train, test in lpgo.split(X, y, groups=groups):
#     print("留 P 组分割：%s %s" % (train, test))
 
# # 随机分组
# gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
# for train, test in gss.split(X, y, groups=groups):
#     print("随机分割：%s %s" % (train, test))
 
 
# # ==================================时间序列分割==========================================
# tscv = TimeSeriesSplit(n_splits=3)
# TimeSeriesSplit(max_train_size=None, n_splits=3)
# for train, test in tscv.split(iris.data):
#     print("时间序列分割：%s %s" % (train, test))


# coding: utf-8
'''
    抽取图像的sift特征
    -》聚类生成码本
    -》得到训练集特征向量
    -》训练svm分类器
    -》测试分类器准确率
'''
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
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 550,contrastThreshold = 0.021)
    kps, des= sift.detectAndCompute(gray, None)
    #136585
    # surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 680, nOctaves = 4, nOctaveLayers = 3)
    # surf = cv2.xfeatures2d.SURF_create()
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
            centers = np.load("Temp/sift_double/sift_double_kmean"+ "cross" + str(count) + "k" + str(n) + ".npy")
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
            clf = SVC(C=10, gamma='auto', kernel='linear')
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
            f = open("I:\image_classification_using_sift\manual_cross/sift_double\\" + "cross" + str(count)+"sift_double_svm_10_record.txt",'a', encoding='utf-8')
            space = "********************************************************\n"
            mes = "sift" + "cross" + str(count) + " " + "k" + str(n) + " " + "norm: " + str(nor) + "\n"
            f.writelines(space)
            f.writelines(mes)
            f.writelines(classification_report(testlabels, testresults, target_names=target_names))
            f.close()



 
# calcFeatureSet()
# calcVoc()
# trainClassfier()