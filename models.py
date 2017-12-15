import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def add_intercept(X_):
    m,n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def base_model(X):
	Y = np.zeros(X.shape[0])
	return Y

def printout(X_test, Y_test, name, model, X_train=None, Y_train=None):
	print(name + '\n---------------------')
	if X_train is not None:
		print("Cross Validation Score")
		print(np.mean(cross_val_score(model, X_train, Y_train, cv=10))) 
		print()
		model.fit(X_train, Y_train)
		print("Training Accuracy:")
		print(accuracy_score(Y_train,model.predict(X_train)))
		print("Training F1 Score:")
		print(f1_score(Y_train,model.predict(X_train),average='binary'))
		print()
	print("Test Accuracy:")
	print(accuracy_score(Y_test,model.predict(X_test)))
	print("Test F1 Score:")
	print(f1_score(Y_test,model.predict(X_test),average='binary')) 
	print('---------------------\n')
	print "Confusion Matrix: "
	print confusion_matrix(Y_test, model.predict(X_test))
	print "----------------------"
	print ""


X_ = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(range(2,50)))
scaler = preprocessing.StandardScaler().fit(X_)
X = scaler.transform(X_)
Y = np.genfromtxt('data_nextyear.csv',delimiter=',',skip_header = 2,dtype=float,usecols=(51))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)

log_r = LogisticRegression(class_weight='balanced',penalty='l2')

linear_svm = SVC(kernel='linear', class_weight='balanced')

nn = MLPClassifier(hidden_layer_sizes=(25,),max_iter=1000, solver='sgd', momentum=0.95)

r_forest = RandomForestClassifier(n_estimators=100, max_features=7)

models = {"Logistic Regression":log_r, "SVM":linear_svm, "Neural Network":nn, "Random Forest":r_forest}


y_base = base_model(X_test)
print("Base rate accuracy (frequency of zeros):")
print(accuracy_score(Y_test,y_base))
print()

for name, model in models.items():
	if name == 'Logistic Regression':
		printout(add_intercept(X_test), Y_test, name, model, add_intercept(X_train), Y_train)
	else:
		printout(X_test, Y_test, name, model, X_train, Y_train)


X_addendum = np.genfromtxt('addendum_test.csv',delimiter=',', dtype=float,usecols=(range(48)))
X_addendum = scaler.transform(X_addendum)
Y_addendum = np.genfromtxt('addendum_test.csv',delimiter=',',dtype=float,usecols=(49))

for name, model in models.items():
	if name == 'Logistic Regression':
		printout(add_intercept(X_addendum), Y_addendum, name, model)
	else:
		printout(X_addendum, Y_addendum, name, model)











