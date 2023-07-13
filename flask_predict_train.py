#RF Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#load data
iris = load_iris()
X = iris.data
y = iris.target

#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, 
    random_state = 42, test_size = 0.5)

# model
clf = RandomForestClassifier(n_estimators = 10)

#train model
clf.fit(X_train, y_train)

#predict
predicted = clf.predict(X_test)

#check accuracy
print(accuracy_score(predicted, y_test))

#pickle the file for later use
import pickle

with open('/Users/feebr01/Documents/p_docker/rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl)
    
