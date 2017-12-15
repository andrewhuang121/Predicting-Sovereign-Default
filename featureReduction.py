import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def add_intercept(X_):
    m,n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def base_model(X):
	Y = np.zeros(X.shape[0])
	return Y

X_ = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(range(2,50)))

scaler = preprocessing.StandardScaler().fit(X_)
X_scaled = scaler.transform(X_)
X = X_scaled
# X = add_intercept(X_scaled)
Y = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(51))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


for num_features in range(2,50):
	model = LogisticRegression()
	selector = RFE(model, num_features)
	selector = selector.fit(X_train, Y_train)
	features_info = selector.support_
	feature_indices = []
	for i in range(len(features_info)):
		if features_info[i] == True:
			feature_indices.append(i+2)
	print feature_indices

	X_new = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(feature_indices))
	scaler_new = preprocessing.StandardScaler().fit(X_new)
	X_scaled_new = scaler_new.transform(X_new)

	Xnew = add_intercept(X_scaled_new)
	Ynew = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(51))

	X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(Xnew,Ynew,test_size = 0.3, random_state = 42)

	
	print "Feature indices" + str(feature_indices)
	# Calculate Base Accuracy - if model just predicted no defaults
	y_base = base_model(X_test_new)
	print "Base rate accuracy:"
	print accuracy_score(Y_test_new,y_base)

	# Actual Model:
	model_new = LogisticRegression(penalty='l2',C=1)
	model_new.fit(X_train_new,Y_train_new)
	print "Actual Accuracy:"
	print accuracy_score(Y_test_new,model_new.predict(X_test_new))
	print "True Positives:"
	print precision_score(Y_test_new,model_new.predict(X_test_new),average='binary')
	print "True Negatives:"
	print precision_score(Y_test_new,model_new.predict(X_test_new),average='binary',pos_label=0)
	print "-----------------------"





















