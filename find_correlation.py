import numpy as np
from sklearn import preprocessing
import math


X_ = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(range(2,51)))
scaler = preprocessing.StandardScaler().fit(X_)
X = scaler.transform(X_)
Y = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(51))

M,N = X.shape
correlations = np.zeros(N)
for feature in range(N):
	correlations[feature] = math.fabs(np.corrcoef(X[:,feature],Y)[0][1])
indices_sorted = np.argsort(-correlations)
# print indices_sorted
columns = indices_sorted + 2
print(','.join(str(x) for x in columns))
print(np.sort(correlations)[::-1])
