
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

record = []
voc_cnt = 1000
trainset_path = 'I:\image_classification_using_sift\dataset_whole\drive-download-20210408T004738Z-001\general-surgery'
testset_path = 'TestSet'


def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT()
    #135210
    sift = cv2.xfeatures2d.SIFT_create()
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
deses = np.load('Temp/train_sift_features.npy')
#criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
print('begin kmeans cluster')
#flags = cv2.KMEANS_RANDOM_CENTERS
#_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
#centers, variance = kmeans(deses, voc_cnt, 1) 
# kmeans = KMeans(n_clusters=100, max_iter=300).fit(deses)
# centers = kmeans.cluster_centers_
k_range = [200, 300, 500, 700, 1000]
#k_range = [200, 300]
norm = ["none", "zero_to_one_l2", "zero_to_one", "zero_or_one"]
for nor in norm:
    cv_scores = []		#用来放每个模型的结果值
    cv_precision = []
    cv_recall = []
    for n in k_range:
        # centers, indices = kmeans_plusplus(deses, n_clusters=n)
        # print('kmeans cluster done')
        # np.save('Temp/surf'+'kmeans'+ str(n) +'.npy', centers)
            
        # idf = None

        # def trainClassfier():
        dirs = os.listdir(trainset_path)
        print('trainClassfier', dirs)
        centers = np.load('Temp/sift'+'kmeans'+ str(n) +'.npy')
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
            features=scaler.transform(features)
        elif nor == "zero_or_one":
            scaler = MinMaxScaler()
            scaler.fit(features)
            features=scaler.transform(features)
            features = np.round(features)


        # if n == 200:
        #     line_style_set = '--'
        # elif n == 300:
        #     line_style_set = '-.'
        # elif n == 500:
        #     line_style_set = '-'
        # elif n == 700:
        #     line_style_set = ':'
        # elif n == 1000:
        #     line_style_set = '-.'
        #norm
        # features = preprocessing.normalize(features, norm='l2')
        # features = np.round(features)
        # scaler = MinMaxScaler( )
        # scaler.fit(features)
        # #scaler.data_max_
        # features=scaler.transform(features)
        # features = np.round(features)
        # print(features)

        labels = labels.reshape((-1,1))
        #labels = labels.flatten()
        print(labels.shape)
        print("第2次feature")
        print(features.shape)

        #random forest
        # clf = RandomForestClassifier(max_depth=398,random_state=69,n_estimators=150,min_samples_leaf=1,min_samples_split=2)
        # clf = RandomForestClassifier(n_estimators=350, max_depth=650, min_samples_leaf=1, min_samples_split=2 ,random_state=51)
        clf = RandomForestClassifier()
       # clf.fit(features, labels)
        #train_X,test_X,train_y,test_y = train_test_split(features, labels,test_size=1/5,random_state=3)

        #svc
        # clf = SVC(C=10, gamma='auto', kernel='linear')
        #clf = SVC()
        # clf.fit(features, labels)


        
        #scores = cross_val_score(clf,features, labels,cv=8,scoring='accuracy')
        lb = LabelBinarizer()
        #features = np.array([number[0] for number in lb.fit_transform(features)])  
        labels = np.array([number[0] for number in lb.fit_transform(labels)])  
        precison = cross_val_score(clf,features, labels,cv=10,scoring='precision')
        recall =  cross_val_score(clf,features, labels,cv=10,scoring='recall')
        print(precison)
        print(recall)
        cv_precision.append(precison.mean())
        cv_recall.append(recall.mean())
        #cv_scores.append(scores.mean())
        kiroku = "\nk = " + str(n) + nor + " \n" + str(precison) + "precision mean   " +  str(precison.mean()) + str(recall) + "recall mean   " + str(recall.mean())
        record.append(kiroku)

    #     plt.plot(cv_recall,cv_precision, linestyle=line_style_set, label = "k= " + str(n))
    #     #plt.plot(k_range,cv_recall, color='g', linestyle='-.', label = 'recall')
    #     plt.legend()
    #     plt.xlabel('recall')
    #     plt.ylabel('precision')		
    #     # plt.savefig("I:\image_classification_using_sift\experiments\surf\\precision" + "norm+" + nor + ".png")	
    #     plt.savefig("I:\image_classification_using_sift\experiments\sift\\pr_curve" + "norm+" + nor + ".png")
    #     #plt.close('all')
    # plt.close('all')

    plt.plot(cv_recall,cv_precision, linestyle='-', label = nor)
    #plt.plot(k_range,cv_recall, color='g', linestyle='-.', label = 'recall')
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')		
    # plt.savefig("I:\image_classification_using_sift\experiments\surf\\precision" + "norm+" + nor + ".png")	
    plt.savefig("I:\image_classification_using_sift\experiments\sift\\pr_curve" + "norm+" + nor + ".png")
    #plt.close('all')

for i in record:
    f = open('I:\image_classification_using_sift\experiments\sift\\record_svm_sift_pr.txt','a', encoding='utf-8')
    f.writelines(str(i))
    f.close()







#scores = cross_val_score(clf, features, labels, cv=5)  #cv为迭代次数。
#print(scores)  # 打印输出每次迭代的度量值（准确度）
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）

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

# total = 0; correct = 0; dictIdx = 0
# dirt = os.listdir(testset_path)
# print('start testing')
# print('classify', dirt)
# # for dir in dirs:
# #     count = 0; crt = 0
# #     files = os.listdir(os.path.join(testset_path, dir))
# #     for f in files:
# #         count += 1
# #         im = cv2.imread(os.path.join(testset_path,dir,f))
# #         des = calcSiftFeature(im)
# #         feature = calcImageFeature(des, centers)
# #         feature = feature*idf
# #         feature = preprocessing.normalize(feature, norm='l2')
# #         #feature = np.array(feature, dtype='float32')
# #         #svm_pre = svm.predict(feature)
# #         #print(svm_pre)
# #         # print(feature.shape, feature.dtype)
# #         # svm_pre = svm.predict(feature)
# #         # print(svm_pre)
# #         # print(svm_pre[1])
# #         # if int(dictIdx) == svm.predict(feature):
# #         print(svm.predict(feature))
# #         # if dictIdx == svm.predict(feature):
# #         #     print(count)
# #         #     crt += 1
# #     print('Accuracy Class', dir, crt, '/', count, '=',float(crt)/count)
# #     total += count
# #     correct += crt
# #     dictIdx += 1
# # print('Total Accuracy ', correct, '/', total, float(correct)/total)
# for dir in dirt:
#     count = 0; crt = 0
#     files = os.listdir(os.path.join(testset_path, dir))
#     for f in files:
#         count += 1
#         im = cv2.imread(os.path.join(testset_path,dir,f))
#         des = calcSiftFeature(im)
#         feature = calcImageFeature(des, centers)
#         feature = feature*idf
#         feature = preprocessing.normalize(feature, norm='l2')
#         #p_label, p_acc, p_val = svm_predict(feature, model)
#         #print(svm.predict(feature))
#         #print(p_label)
#         print(int(clf.predict(feature)))
#         if dictIdx == int(clf.predict(feature)):
#             crt += 1
#     print('Accuracy Class', dir, crt, '/', count, '=',float(crt)/count)
#     total += count
#     correct += crt
#     dictIdx += 1
# print('Total Accuracy ', correct, '/', total, float(correct)/total)
    
# calcFeatureSet()
# calcVoc()
# trainClassfier()

# sum1 = 0
# sum2 = 0
# sum3 = 0
# sum4 = 0
# sum5 = 0
# sum6 = 0
# sum7 = 0
# sum8 = 0
# sum9 = 0
# sum10 = 0
# sum11 = 0
# sum12 = 0
# sum13 = 0
# sum14 = 0
# sum15 = 0
# sum16 = 0
# sum17 = 0
# sum18 = 0
# sum19 = 0
# sum20 = 0
# sum21 = 0
# sum22 = 0
# sum23 = 0
# sum24 = 0
# sum25 = 0