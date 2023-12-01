
import pandas as pd
import numpy as np

train = pd.read_csv("file:///C:/Users/91709/PycharmProjects/dic1/train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("file:///C:/Users/91709/PycharmProjects/dic1/test_Y3wMUE5_7gLdaTN.csv")

list(train)
train.head(10)
train.dtypes
train.describe()

train.isnull().sum()
test.isnull().sum()

train.fillna(train.mean(), inplace=True)
train.isnull().sum()

test.fillna(test.mean(), inplace=True)
test.isnull().sum()

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train.isnull().sum()

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test.isnull().sum()

train['Loan_Amount_Term'] = np.log(train['Loan_Amount_Term'])

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

X = train.drop('Loan_Status', 1)
y = train['Loan_Status']

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_cv, pred_cv)
matrix = confusion_matrix(y_cv, pred_cv)
print(matrix)

from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train, y_train)
pred_cv1 = dt.predict(x_cv)
accuracy_score(y_cv, pred_cv1)
matrix1 = confusion_matrix(y_cv, pred_cv1)
print(matrix1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_cv2 = rf.predict(x_cv)
accuracy_score(y_cv, pred_cv2)
matrix2 = confusion_matrix(y_cv, pred_cv2)
print(matrix2)

from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
pred_cv3 = svm_model.predict(x_cv)
accuracy_score(y_cv, pred_cv3)
matrix3 = confusion_matrix(y_cv, pred_cv3)
print(matrix3)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
pred_cv4 = nb.predict(x_cv)
accuracy_score(y_cv, pred_cv4)
matrix4 = confusion_matrix(y_cv, pred_cv4)
print(matrix4)

from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier()
kNN.fit(x_train, y_train)
pred_cv5 = kNN.predict(x_cv)
accuracy_score(y_cv, pred_cv5)
matrix5 = confusion_matrix(y_cv, pred_cv5)
print(matrix5)

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
gbm.fit(x_train, y_train)
pred_cv6 = gbm.predict(x_cv)
accuracy_score(y_cv, pred_cv6)
matrix6 = confusion_matrix(y_cv, pred_cv6)
print(matrix6)

pred_test = nb.predict(test)

predictions = pd.DataFrame(pred_test, columns=['predictions']).to_csv('Credit_Predictions.csv')
