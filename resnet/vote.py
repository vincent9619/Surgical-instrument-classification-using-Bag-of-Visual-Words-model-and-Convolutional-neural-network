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
import sys
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut,LeavePGroupsOut,GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from tensorflow import keras
from keras.models import load_model
from utils import load_imgs
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import preprocess_input
# from keras.applications.resnet import ResNet101
# from ResNet101 import preprocess_input
import numpy as np
import tensorflow as tf
import os
import time
import datetime

np.set_printoptions(threshold=sys.maxsize)

record = []
voc_cnt = 8000



# def output_acc(arr1, arr2, arr3, arr4):
#     arr_result = []
#     arr1 = np.int16(arr1)
#     for i in range(len(arr1)):
#         if arr1[i] == arr2[i] and arr2[i] == arr3[i]:
#             arr_result = np.append(arr_result, arr1[i])
#         elif arr2[i] == arr3[i] and arr3[i] == arr4[i]:
#             arr_result = np.append(arr_result, arr2[i])
#         elif arr1[i] == arr2[i] and arr2[i] == arr4[i]:
#             arr_result = np.append(arr_result, arr2[i])
#         elif arr3[i] == arr1[i] or arr3[i] == arr2[i] or arr3[i] == arr4[i]:
#             arr_result = np.append(arr_result, arr3[i])
#         elif arr1[i] != arr2[i] and arr2[i] != arr3[i] and arr3[i] != arr4[i] and arr1[i] != arr4[i]:
#             arr_result = np.append(arr_result, arr3[i])
#         else:
#             arr_result = np.append(arr_result, arr2[i])
#     return arr_result


def output_acc(arr1, arr2, arr3, arr4):
    arr_result = []
    arr1 = np.int16(arr1)
    for i in range(len(arr1)):
        if arr1[i] == arr2[i] and arr2[i] == arr3[i]:
            arr_result = np.append(arr_result, arr1[i])
        elif arr2[i] == arr3[i] and arr3[i] == arr4[i]:
            arr_result = np.append(arr_result, arr2[i])
        elif arr1[i] == arr2[i] and arr2[i] == arr4[i]:
            arr_result = np.append(arr_result, arr2[i])
        elif arr1[i] != arr2[i] and arr3[i] == arr4[i]:
            arr_result = np.append(arr_result, arr3[i])
        elif arr1[i] != arr3[i] and arr2[i] == arr4[i]:
            arr_result = np.append(arr_result, arr2[i])
        elif arr1[i] != arr4[i] and arr3[i] == arr2[i]:
            arr_result = np.append(arr_result, arr3[i])
        elif arr1[i] != arr2[i] and arr3[i] == arr2[i]:
            arr_result = np.append(arr_result, arr3[i])
        elif arr2[i] != arr3[i] and arr1[i] == arr4[i]:
            arr_result = np.append(arr_result, arr1[i])
        elif arr2[i] != arr4[i] and arr3[i] == arr1[i]:
            arr_result = np.append(arr_result, arr1[i])
        elif arr3[i] != arr4[i] and arr1[i] == arr2[i]:
            arr_result = np.append(arr_result, arr1[i])
        else:
            arr_result = np.append(arr_result, arr3[i])
    return arr_result



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
# start = datetime.datetime.now()
for count in range(1,2):
    trainset_path = "D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\" +"train"
    testset_path = "D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\" +"test"
    #print(count)
    #deses = np.load("Temp/train_sift_features"+ "cross" + str(count) + ".npy")
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 2000, 1.0)
    print('begin kmeans cluster')
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #_, labels, centers = cv2.kmeans(deses, 2, None,criteria,20,flags)
    #centers, variance = kmeans(deses, voc_cnt, 1) 
    # kmeans = KMeans(n_clusters=100, max_iter=300).fit(deses)
    # centers = kmeans.cluster_centers_
    n = 8000
    #k_range = [200, 300]
    # norm = ["none", "zero_to_one_l2", "zero_to_one", "zero_or_one"]
    norm = "zero_to_one_l2"
    #norm = ["zero_to_one_l2", "none", "zero_to_one", "zero_or_one"]
    # centers, indices = kmeans_plusplus(deses, n_clusters=n)
    # print('kmeans cluster done')
    # np.save('Temp/surf'+'kmeans'+ str(n) +'.npy', centers)
        
    # idf = None

    # def trainClassfier():
    dirs = os.listdir(trainset_path)
    print('trainClassfier', dirs)
    centers = np.load("D:\\592\\resnet\\shoes_classification\\bovw\\104_sift_double"+ "cross" + str(count) + "k" + str(n) + ".npy")
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
    # ft = open("I:\image_classification_using_sift\\666\\" + "test.txt",'a', encoding='utf-8')
    # mes1 = "first feature"
    # mes2 = "second feature"
    # print("第一次feature")
    # print(features.shape)
    # print(labels)
    nbr_occurences = np.sum((features>0)*1, axis=0)
    idf = np.array(np.log((1.0*features.shape[0]+1)/(1.0*nbr_occurences+1)), dtype=np.float32)
    features = features*idf
    # print(features)
    
    # ft.writelines(mes1)
    # ft.writelines(str(features))

    
    features = preprocessing.normalize(features, norm='l2')
    

    # ft.writelines(mes2)
    # ft.writelines(str(features))
    

    labels = labels.reshape((-1,1))
    #labels = labels.flatten()
    # print(labels.shape)
    print("第2次feature")
    # print(features.shape)

    #random forest
    # clf = RandomForestClassifier(max_depth=398,random_state=69,n_estimators=150,min_samples_leaf=1,min_samples_split=2)
    # clf = RandomForestClassifier(n_estimators=350, max_depth=650, min_samples_leaf=1, min_samples_split=2 ,random_state=51)
    # clf = RandomForestClassifier()
    # clf.fit(features, labels)
    #train_X,test_X,train_y,test_y = train_test_split(features, labels,test_size=1/5,random_state=3)

    #svc
    #clf = SVC(C=15, gamma='auto', kernel='linear')
    clf = SVC(C=5, gamma='auto', kernel='linear')
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
            feature = preprocessing.normalize(feature, norm='l2')
           
            #feature = preprocessing.normalize(feature, norm='l2')
            testlabels = np.append(testlabels, np.float32(testId))
            #p_label, p_acc, p_val = svm_predict(feature, model)
            #print(svm.predict(feature))
            #print(p_label)  
            testresults = np.append(testresults, np.float32(clf.predict(feature)))
            #print(int(clf.predict(feature)))
        testId += 1
    
    # print(classification_report(testlabels, testresults, target_names=target_names))
    

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # vram limit for eficiency
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    config=tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess=tf.compat.v1.Session(config=config)
    print("初始化")
    ############# same as training phase ##############
    # define class names
    classes = ['1.1mm_K-Wire',
          '6_Babcock_Tissue_Forceps',
           '6_Mayo_Needle_Holder',
            '7_Metzenbaum_Scissors',
             '7_Microvascular_Needle_Holder',
              '8_Babcock_Tissue_Forceps' ,
              '8_Mayo_Needle_Holder' ,
              '8_Microvascular_Needle_Holder' ,
              '9_DeBakey_Dissector' ,
              '9_DeBakey_Needle_Holder',
              '9_Metzenbaum_Scissors' ,
              'Allis_Tissue_Forceps' ,
              'angled-post',
              'Ball___Socket_Towel_Clips' ,
              'Bearing_Plates_3_Hole_7_Peg_Left',
              'Bearing_Plates_3_Hole_7_Peg_Left_Narrow',
              'Bearing_Plates_3_Hole_7_Peg_Right',
              'Bearing_Plates_3_Hole_7_Peg_Right_Narrow',
              'Bearing_Plates_5_Hole_5_Peg_Left',
              'Bearing_Plates_5_Hole_5_Peg_Right',
              'Bearing_Plates_5_Hole_7_Peg_Left',  
              'Bearing_Plates_5_Hole_7_Peg_Right',
              'Bearing_Plates_7_Hole_7_Peg_Left',
              'Bearing_Plates_7_Hole_7_Peg_Right',
              'Bonneys_Non_Toothed_Dissector' ,
              'Bonneys_Toothed_Dissector',
              'chuck-no3',
              'chuck-no4',
              'Clavicle_Plate_3.5_6_Hole_Left',
              'Clavicle_Plate_3.5_6_Hole_Right',  
              'Clavicle_Plate_3.5_7_Hole_Left',
              'Clavicle_Plate_3.5_7_Hole_Right',
              'Clavicle_Plate_3.5_8_Hole_Left',
              'Clavicle_Plate_3.5_8_Hole_Right',
              'Clavicle_Plate_3.5-2.7_3_Hole_Left',
              'Clavicle_Plate_3.5-2.7_3_Hole_Right',
              'Clavicle_Plate_3.5-2.7_4_Hole_Left',  
              'Clavicle_Plate_3.5-2.7_4_Hole_Right',
              'Clavicle_Plate_3.5-2.7_5_Hole_Left',
              'Clavicle_Plate_3.5-2.7_5_Hole_Right',
              'Clavicle_Plate_3.5-2.7_6_Hole_Left',
              'Clavicle_Plate_3.5-2.7_6_Hole_Right',
              'Clavicle_Plate_3.5-2.7_7_Hole_Left',
              'Clavicle_Plate_3.5-2.7_7_Hole_Right',  
              'Clavicle_Plate_3.5-2.7_8_Hole_Left',
              'Clavicle_Plate_3.5-2.7_8_Hole_Right',
              'Clavicle_Plate_Lateral_7_Hole',
              'Clavicle_Plate_Lateral_9_Hole',
              'Clavicle_Plate_Lateral_10_Hole',
              'Clavicle_Plate_Lateral_11_Hole',
              'Clavicle_Plate_Lateral_12_Hole',  
              'Clavicle_Plate_Medial_6_Hole',
              'Clavicle_Plate_Medial_7_Hole',
              'Clavicle_Plate_Medial_8_Hole',
              'compression-distraction-rod',
              'coupling-orange-green',
              'Crile_Artery_Forceps',
              'Curved_Mayo_Scissors',
              'Depth_Measure',
              'Dorsal_Guide',  
              'Dorsal_Hook_Plate_4_Hole',
              'Dorsal_Hook_Plate_7_Hole',
              'Dressing_Scissors',
              'Drill_Sleeve_2.7_Coaxial',
              'Drill_Sleeve_2.7_Conical',
              'drill-bit-2mm',
              'drill-bit-3.2mm',
              'Fixed_Angle_Plates_3_Hole_7_Peg_Left',  
              'Fixed_Angle_Plates_3_Hole_7_Peg_Right',
              'Fixed_Angle_Plates_5_Hole_6_Peg_Right',
              'Fixed_Angle_Plates_5_Hole_7_Peg_Left',    
              'Gillies_Toothed_Dissector',
              'handle-for-guide-block',
              'hoffman-4-hole-guide-block',
              'Hook_Impactor',
              'Hook_Plate_Volar_4_Hole',  
              'Hook_Plate_Volar_7_Hole',
              'Lahey_Forceps',
              'Large_Depth_Gauge',
              'Large_Quick_Guide',
              'Littlewood_Tissue_Forceps',
              'Mayo_Artery_Forceps',
              'multi-pin-clamp-grey-orange',
              'No3_BP_Handles',
              'No4_BP_Handles',
              'No7_BP_Handles',
              'Olive_K-Wire_1.6mm',
              'Peg_Extender_Quick_Release',
              'Peg_Guides',  
              'periarticular-clamp',
              'pin-to-rod-coupling-grey-orange',
              'rod-to-rod-coupling',
              'rod-to-rod-coupling-orange-orange',
              'Small_Depth_Gauge',
              'Small_Quick_Guide',
              'spanner',
              'Sponge_Forceps',     
              'straight-post',
              'trocar-no3',  
              'trocar-no4',
              'Twist_Drill_1.8mm',
              'Twist_Drill_2.30mm',
              'Volar_Guide',
              'Volar_Impactor',
              'Wedge_Screws',
              'wrench',
            ]
    # classes = ['7_Microvascular_Needle_Holder']
    # define our image size
    target_size = (224, 224)
    ###################################################

    # load the test data
    # currently using debug data(SAME as training data, so it's cheating)
    data_x, data_y = load_imgs(datapath=testset_path, classes=classes, target_size=target_size)

    ############# same as training phase ##############
    # preprocess images for the model
    # data_x = preprocess_input(data_x)
    data_x = preprocess_input(data_x)
    # preprocess labels with scikit-learn LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(data_y)
    data_y_int = label_encoder.transform(data_y)
    ###################################################

    start4 = datetime.datetime.now()
    # load the model
    print("load model1")
    model_shoes1 = load_model("D:\\592\\resnet\\shoes_classification\\106_classes\\resnet50\\"+ str(count)+ ".h5")
    start1 = datetime.datetime.now()
    preds1 = model_shoes1.predict(data_x, batch_size=20, verbose=1)
    preds1 = np.argmax(preds1, axis=1)
    end1 = datetime.datetime.now()
    print("BOVW time")
    print((end1 - start1).seconds)


    # load the model
    print("load model2")
    model_shoes2 = load_model("D:\\592\\resnet\\shoes_classification\\106_classes\\resnet101\\"+ str(count)+ ".h5")
    start2 = datetime.datetime.now()
    preds2 = model_shoes2.predict(data_x, batch_size=20, verbose=1)
    preds2 = np.argmax(preds2, axis=1)
    end2 = datetime.datetime.now()
    print("ResNet101 time")
    print((end2 - start2).seconds)

    # load the model
    print("load model3")
    model_shoes3 = load_model("D:\\592\\resnet\\shoes_classification\\106_classes\\resnet152\\"+ str(count)+ ".h5")
    start3 = datetime.datetime.now()
    preds3 = model_shoes3.predict(data_x, batch_size=20, verbose=1)
    preds3 = np.argmax(preds3, axis=1)
    end3 = datetime.datetime.now()
    print("ResNet152 time")
    print((end3 - start3).seconds)

    # print('')
    # print(accuracy_score(y_true=data_y_int, y_pred=preds))

    # print(testresults)
    # print(preds1)
    # print(preds2)
    # print(preds3)


    # print(accuracy_score(y_true=data_y_int, y_pred=output_acc(testresults, preds1, preds2, preds3)))
    # f = open("D:\\592\\resnet\\shoes_classification\\record_vote.txt",'a', encoding='utf-8')
    # space = "********************************************************\n"
    # mes = accuracy_score(y_true=data_y_int, y_pred=output_acc(testresults, preds1, preds2, preds3))
    # f.writelines(space)
    # f.writelines(str(mes))
    # f.close()
    end4 = datetime.datetime.now()
    print("vote time")
    print((end4 - start4).seconds)

