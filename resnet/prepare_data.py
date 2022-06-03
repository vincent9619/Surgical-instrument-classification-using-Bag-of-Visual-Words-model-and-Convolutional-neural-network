import numpy as np
import os
import cv2 as cv
import sys
import shutil

# for count in range(1,9):
count = 9
trainset_path = "D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\train"
dirs = os.listdir(trainset_path)
for dir in dirs:
    files = os.listdir(os.path.join(trainset_path, dir))
    if (len(files) % 10 ) < 5:
        n = int(len(files) / 10)
    elif (len(files) % 10 ) >= 5:
        n = int(len(files) / 10) + 1
    # print(n)
    if count != 10:
        for i in range(n*count-n , n*count):
            if files[i]:
                # src = cv.imread(files[i])
                shutil.move("D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\train" + "\\" + str(dir) + "\\" + str(files[i]), "D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\test" + "\\" + str(dir))
                # print(files[i])
    if count == 10:
        for i in range(n*10-n , len(files)):
            if files[i]:
                # src = cv.imread(files[i])
                shutil.move("D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\train" + "\\" + str(dir) + "\\" + str(files[i]), "D:\\592\\resnet\\whole_data_cv\\" + str(count) + "\\test" + "\\" + str(dir))
                # print(files[i])