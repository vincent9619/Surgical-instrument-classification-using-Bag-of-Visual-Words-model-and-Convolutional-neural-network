from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import *
import os
import matplotlib.pyplot as plt

K=range(1,2001,200)
sse_result=[]
deses = np.load('Temp/train_sift_features.npy')
for k in K:
    kmeans=KMeans(n_clusters=k)
    print(k)
    kmeans.fit(deses)
    sse_result.append(sum(np.min(cdist(deses,kmeans.cluster_centers_,'euclidean'),axis=1))/deses.shape[0])
plt.plot(K,sse_result,'gx-')
plt.xlabel('k')
plt.ylabel(u'y')
plt.title(u'best_k')
plt.show()