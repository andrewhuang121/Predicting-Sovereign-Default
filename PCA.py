import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA


X_ = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(range(2,50)))
scaler = preprocessing.StandardScaler().fit(X_)
X_scaled = scaler.transform(X_)

# pca = PCA(n_components=15)
# pca.fit(X_scaled)

pca_trafo = PCA().fit(X_scaled)
print(pca_trafo.explained_variance_ratio_.cumsum())

# X_new = pca.transform(X_scaled)

# print X_new.shape