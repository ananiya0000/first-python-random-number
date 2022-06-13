import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

datainput = pd.read_csv("mushrooms.csv", delimiter=",") # Read data from csv file

X = datainput[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor']].values

# Data Preprocessing
from sklearn import preprocessing # changes non-numerical values to numbers so that algorithms can work on them

label_capShape = preprocessing.LabelEncoder()
label_capShape.fit(['b', 'c', 'x', 'f', 'k', 's'])
X[:, 0] = label_capShape.transform(X[:, 0])

label_capSurface = preprocessing.LabelEncoder()
label_capSurface.fit(['f', 'g', 'y', 's'])
X[:, 1] = label_capSurface.transform(X[:, 1])

label_capColor = preprocessing.LabelEncoder()
label_capColor.fit(['n','b','c','g','r','p','u','e','w','y'])
X[:, 2] = label_capColor.transform(X[:, 2])

label_bruises = preprocessing.LabelEncoder()
label_bruises.fit(['t','f'])
X[:, 3] = label_bruises.transform(X[:, 3])

label_odor = preprocessing.LabelEncoder()
label_odor.fit(['a','l','c','y','f','m','n','p','s'])
X[:, 4] = label_odor.transform(X[:, 4])

y = datainput["class"]

# train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3) # We use 70% of the data to train & 30% to test

mushroomClassifier = GaussianNB() 

mushroomClassifier.fit(X_train, y_train)
predicted = mushroomClassifier.predict(X_test)

print(predicted)

print("\nNaive Bayes's Accuracy: ", metrics.accuracy_score(y_test, predicted))