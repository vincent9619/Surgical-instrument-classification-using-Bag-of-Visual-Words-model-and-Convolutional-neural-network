import numpy as np

t = np.load("D:\\592\\image_classification_using_sift\\sift_final_output.npy")
a_dict = t[()]

class_0 = 0
class_1 = 0
class_2 = 0
class_3 = 0

for i in range(len(a_dict)):
    if str(t[i]) == "0.0":
        class_0 = class_0 + 1
    elif str(t[i]) == "1.0":
        class_1 = class_1 + 1
    elif str(t[i]) == "2.0":
        class_2 = class_2 + 1
    elif str(t[i]) == "3.0":
        class_3 = class_3 + 1




print(class_0)
print(class_1)
print(class_2)
print(class_3)
print(len(a_dict))