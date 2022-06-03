import re
import xlwt
import xlwings as xw
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

accuracy_all = []
accuracy_all1 = []
accuracy_all2 = []
accuracy_all3 = []
accuracy_all4 = []
accuracy_all5 = []
accuracy_all6 = []
accuracy_all7 = []
accuracy_all8 = []
accuracy_all9 = []
accuracy_all10 = []
accuracy_all11 = []
accuracy_all12 = []
accuracy_all13 = []
accuracy_all14 = []
accuracy_all15 = []
accuracy_all16 = []
deviation1 = []
deviation2 = []
deviation3 = []
deviation4 = []
deviation5 = []
deviation6 = []
deviation7 = []
deviation8 = []
deviation9 = []
deviation10 = []
deviation11 = []
deviation12 = []
deviation13 = []
deviation14 = []
deviation15 = []
deviation16 = []


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         #f = open(r"I:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"sift_random_forest_default_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"sift_random_forest_default_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation1.append(round(np.std(sum_sepv),3))
#     accuracy_all1.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))
# # accuracy_all.append("sift_rf_d")


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"surf_random_forest_default_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"surf_random_forest_default_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation2.append(round(np.std(sum_sepv),3))
#     accuracy_all2.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))
# # accuracy_all.append("surf_rf_d")

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"sift_svm_default_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"sift_svm_default_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation3.append(round(np.std(sum_sepv),3))
#     accuracy_all3.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"surf_svm_default_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\default\\" + "cross" + str(i)+"surf_svm_default_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation4.append(round(np.std(sum_sepv),3))
#     accuracy_all4.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_1000\\" + "cross" + str(i)+"sift_random_forest_1000_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\random_forest_1000\\" + "cross" + str(i)+"sift_random_forest_1000_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation5.append(round(np.std(sum_sepv),3))
#     accuracy_all5.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))
# # accuracy_all.append("surf_rf_1000")

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\svm15\\" + "cross" + str(i)+"sift_svm_15_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\svm15\\" + "cross" + str(i)+"sift_svm_15_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation6.append(round(np.std(sum_sepv),3))
#     accuracy_all6.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))
# # accuracy_all.append("surf_svm_15")

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_1000\\" + "cross" + str(i)+"surf_random_forest_1000_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\random_forest_1000\\" + "cross" + str(i)+"surf_random_forest_1000_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation7.append(round(np.std(sum_sepv),3))
#     accuracy_all7.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\svm15\\" + "cross" + str(i)+"surf_svm_15_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\svm15\\" + "cross" + str(i)+"surf_svm_15_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation8.append(round(np.std(sum_sepv),3))
#     accuracy_all8.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))

# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_500\\" + "cross" + str(i)+"sift_random_forest_500_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\random_forest_500\\" + "cross" + str(i)+"sift_random_forest_500_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation9.append(round(np.std(sum_sepv),3))
#     accuracy_all9.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\svm10\\" + "cross" + str(i)+"sift_svm_10_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\svm10\\" + "cross" + str(i)+"sift_svm_10_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation10.append(round(np.std(sum_sepv),3))
#     accuracy_all10.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_500\\" + "cross" + str(i)+"surf_random_forest_500_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\random_forest_500\\" + "cross" + str(i)+"surf_random_forest_500_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation11.append(round(np.std(sum_sepv),3))
#     accuracy_all11.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,28):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         # f = open(r"I:\image_classification_using_sift\manual_cross\svm10\\" + "cross" + str(i)+"surf_svm_10_record.txt",'r')
#         f = open(r"F:\image_classification_using_sift\manual_cross\svm10\\" + "cross" + str(i)+"surf_svm_10_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation12.append(round(np.std(sum_sepv),3))
#     accuracy_all12.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,20):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_300\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation13.append(round(np.std(sum_sepv),3))
#     accuracy_all13.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,20):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         f = open(r"I:\image_classification_using_sift\manual_cross\svm5\\" + "cross" + str(i)+"sift_svm_5_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation14.append(round(np.std(sum_sepv),3))
#     accuracy_all14.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,20):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         f = open(r"I:\image_classification_using_sift\manual_cross\random_forest_300\\" + "cross" + str(i)+"surf_random_forest_300_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation15.append(round(np.std(sum_sepv),3))
#     accuracy_all15.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))


# for number in range(0,20):
#     sum_sep = 0
#     sum_sepv = []
#     # sum_min = 1
#     # sum_max = 0
#     for i in range(1,11):

#         #f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_random_forest_300_record.txt",'r')
#         f = open(r"I:\image_classification_using_sift\manual_cross\svm5\\" + "cross" + str(i)+"surf_svm_5_record.txt",'r')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum_sep = float(accuracy[number]) + sum_sep
#         sum_sepv.append(float(accuracy[number]))
#         # if sum_min > float(accuracy[number]):
#         #     sum_min = float(accuracy[number])
#         # if sum_max < float(accuracy[number]):
#         #     sum_max = float(accuracy[number])
#     # deviation1.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     deviation16.append(round(np.std(sum_sepv),3))
#     accuracy_all16.append(round(sum_sep/10,3))
#     accuracy_all.append(round(sum_sep/10,3))

















# for number in range(0,20):
#     sum = 0
#     sum_min = 1
#     sum_max = 0
#     for i in range(1,11):

#         f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"sift_svm_15_record.txt",'r')
#         #f = open('I:\image_classification_using_sift\manual_cross\cross1record.txt','a', encoding='utf-8')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum = float(accuracy[number]) + sum
#         if sum_min > float(accuracy[number]):
#             sum_min = float(accuracy[number])
#         if sum_max < float(accuracy[number]):
#             sum_max = float(accuracy[number])
#     accuracy_all2.append(round(sum/10,3))
#     deviation2.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     accuracy_all.append(round(sum/10,3))


# for number in range(0,20):
#     sum = 0
#     sum_min = 1
#     sum_max = 0
#     for i in range(1,11):

#         f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"surf_random_forest_300_record.txt",'r')
#         #f = open('I:\image_classification_using_sift\manual_cross\cross1record.txt','a', encoding='utf-8')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum = float(accuracy[number]) + sum
#         if sum_min > float(accuracy[number]):
#             sum_min = float(accuracy[number])
#         if sum_max < float(accuracy[number]):
#             sum_max = float(accuracy[number])
#     accuracy_all3.append(round(sum/10,3))
#     deviation3.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     accuracy_all.append(round(sum/10,3))


# for number in range(0,20):
#     sum = 0
#     sum_min = 1
#     sum_max = 0
#     for i in range(1,11):

#         f = open(r"I:\image_classification_using_sift\manual_cross\\" + "cross" + str(i)+"surf_svm_15_record.txt",'r')
#         #f = open('I:\image_classification_using_sift\manual_cross\cross1record.txt','a', encoding='utf-8')
#         s=f.read()

#         accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
#         #print(accuracy)
#         sum = float(accuracy[number]) + sum
#         if sum_min > float(accuracy[number]):
#             sum_min = float(accuracy[number])
#         if sum_max < float(accuracy[number]):
#             sum_max = float(accuracy[number])
#     accuracy_all4.append(round(sum/10,3))
#     deviation4.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
#     accuracy_all.append(round(sum/10,3))


for number in range(0,28):
    sum = 0
    sum_min = 1
    sum_max = 0
    for i in range(1,11):

        f = open(r"I:\image_classification_using_sift\manual_cross\surf_double\\" + "cross" + str(i)+"surf_double_svm_5_record.txt",'r')
        #f = open('I:\image_classification_using_sift\manual_cross\cross1record.txt','a', encoding='utf-8')
        s=f.read()

        accuracy = re.findall( r'accuracy...........................(.*?)\ ', s)
        #print(accuracy)
        sum = float(accuracy[number]) + sum
        if sum_min > float(accuracy[number]):
            sum_min = float(accuracy[number])
        if sum_max < float(accuracy[number]):
            sum_max = float(accuracy[number])
    accuracy_all4.append(round(sum/10,3))
    deviation4.append(round(max(round(sum/10,3)-sum_min,sum_max-round(sum/10,3)),3))
    accuracy_all.append(round(sum/10,3))


def get_acc1(arr):
    get_ac = []
    get_ac.append(arr[0])
    get_ac.append(arr[1])
    get_ac.append(arr[2])
    get_ac.append(arr[3])
    get_ac.append(arr[4])
    get_ac.append(arr[5])
    get_ac.append(arr[6])
    return get_ac

def get_dev1(arr):
    get_dev = []
    get_dev.append(arr[0])
    get_dev.append(arr[1])
    get_dev.append(arr[2])
    get_dev.append(arr[3])
    get_dev.append(arr[4])
    get_dev.append(arr[5])
    get_dev.append(arr[6])
    return get_dev

def get_acc2(arr):
    get_ac = []
    get_ac.append(arr[7])
    get_ac.append(arr[8])
    get_ac.append(arr[9])
    get_ac.append(arr[10])
    get_ac.append(arr[11])
    get_ac.append(arr[12])
    get_ac.append(arr[13])
    return get_ac

def get_dev2(arr):
    get_dev = []
    get_dev.append(arr[7])
    get_dev.append(arr[8])
    get_dev.append(arr[9])
    get_dev.append(arr[10])
    get_dev.append(arr[11])
    get_dev.append(arr[12])
    get_dev.append(arr[13])
    return get_dev


def get_acc3(arr):
    get_ac = []
    get_ac.append(arr[14])
    get_ac.append(arr[15])
    get_ac.append(arr[16])
    get_ac.append(arr[17])
    get_ac.append(arr[18])
    get_ac.append(arr[19])
    get_ac.append(arr[20])
    return get_ac

def get_dev3(arr):
    get_dev = []
    get_dev.append(arr[14])
    get_dev.append(arr[15])
    get_dev.append(arr[16])
    get_dev.append(arr[17])
    get_dev.append(arr[18])
    get_dev.append(arr[19])
    get_dev.append(arr[20])
    return get_dev

def get_acc4(arr):
    get_ac = []
    get_ac.append(arr[21])
    get_ac.append(arr[22])
    get_ac.append(arr[23])
    get_ac.append(arr[24])
    get_ac.append(arr[25])
    get_ac.append(arr[26])
    get_ac.append(arr[27])
    return get_ac

def get_dev4(arr):
    get_dev = []
    get_dev.append(arr[21])
    get_dev.append(arr[22])
    get_dev.append(arr[23])
    get_dev.append(arr[24])
    get_dev.append(arr[25])
    get_dev.append(arr[26])
    get_dev.append(arr[27])
    return get_dev




# # title =  "Sift + randome forest(default)"
# # title =  "Sift + svm(default)"
# # title =  "Surf + randome forest(default)"
# # title =  "Surf + svm(default)"
# # title =  "Sift + randome forest(n_estimators = 1000)"
# # title =  "Sift + svm(c = 15)"
# # title =  "Surf + randome forest(n_estimators = 1000)"
# # title =  "Surf + svm(c = 15)"
# # title =  "Sift + randome forest(n_estimators = 500)"
# # title =  "Sift + svm(c = 10)"
# # title =  "Surf + randome forest(n_estimators = 500)"
# # title =  "Surf + svm(c = 10)"
# # title =  "Sift + randome forest(n_estimators = 300)"
# # title =  "Sift + svm(c = 5)"
# # title =  "Surf + randome forest(n_estimators = 300)"
title =  "Surf + svm(c = 5)"  


# index = np.arange(6)
plt.title(title ,fontsize=20) #设置标题名称
plt.xlabel("K value",fontsize=18)
plt.ylabel("Accuracy",fontsize=18)
x_data = ['200', '300', '500', '700', '1000', '1500', '2000']
# y_data1 = [accuracy_all1[0],accuracy_all1[1],accuracy_all1[2],accuracy_all1[3],accuracy_all1[4]]
y_data1 = get_acc1(accuracy_all4)
y_data2 = get_acc2(accuracy_all4)
y_data3 = get_acc3(accuracy_all4)
y_data4 = get_acc4(accuracy_all4)
# y_data2 = [accuracy_all1[5],accuracy_all1[6],accuracy_all1[7],accuracy_all1[8],accuracy_all1[9]]
# y_data3 = [accuracy_all1[10],accuracy_all1[11],accuracy_all1[12],accuracy_all1[13],accuracy_all1[14]]
# y_data4 = [accuracy_all1[15],accuracy_all1[16],accuracy_all1[17],accuracy_all1[18],accuracy_all1[19]]
# plt.errorbar(x_data, y_data1, yerr=[deviation1[0],deviation1[1],deviation1[2],deviation1[3],deviation1[4]],color='green', linestyle='-.',label = "none")
# plt.errorbar(x_data, y_data1, yerr=get_dev2(deviation16),color='green', linestyle='-.',label = "none")
# plt.errorbar(x_data, y_data2, yerr=get_dev2(deviation16),color='blue', linestyle='--',label = "zero_to_one_l2")
# plt.errorbar(x_data, y_data3, yerr=get_dev2(deviation16),color='black', linestyle=':',label = "zero_to_one")
# plt.errorbar(x_data, y_data4, yerr=get_dev2(deviation16),color='gray', linestyle='-',label = "zero_or_one")
# plt.legend()
# plt.show()
#plt.savefig("I:\image_classification_using_sift\\acc_curve\\acc " + title + ".png")



mean_1 = y_data1
std_1 = get_dev1(deviation4)

mean_2 = y_data2
std_2 = get_dev2(deviation4)

mean_3 = y_data3
std_3 = get_dev3(deviation4)

mean_4 = y_data4
std_4 = get_dev4(deviation4)

# def np_change(np_arr):
#     np_ar = np.array(np_arr)
#     return np_ar

# np_change(mean_1)
# np_change(std_1)
# np_change(mean_2)
# np_change(std_2)
# np_change(mean_3)
# np_change(std_3)
# np_change(mean_4)
# np_change(std_4)

mean_1 = np.array(mean_1)
mean_2 = np.array(mean_2)
mean_3 = np.array(mean_3)
mean_4 = np.array(mean_4)
std_1 = np.array(std_1)
std_2 = np.array(std_2)
std_3 = np.array(std_3)
std_4 = np.array(std_4)


x = np.arange(len(mean_1))
# plt.plot(x_data, mean_1, 'b-', label='mean_1')
# plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
# plt.plot(x_data, mean_2, 'r-', label='mean_2')
# plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
# plt.plot(x_data, mean_3, 'r-', label='mean_2')
# plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='r', alpha=0.2)
# plt.plot(x_data, mean_4, 'r-', label='mean_2')
# plt.fill_between(x, mean_4 - std_4, mean_4 + std_4, color='r', alpha=0.2)
plt.plot(x_data, mean_1,'yellow',  label='mean_1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1,color='yellow', alpha=0.7)
plt.plot(x_data, mean_2,'cyan',  label='mean_2')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2,color='cyan',  alpha=0.7)
plt.plot(x_data, mean_3,'indigo',  label='mean_2')
plt.fill_between(x, mean_3 - std_3, mean_3 + std_3,color='darkblue',  alpha=0.99)
plt.plot(x_data, mean_4,'red',  label='mean_2')
plt.fill_between(x, mean_4 - std_4, mean_4 + std_4,color='red',  alpha=0.5)
plt.legend()
plt.show()