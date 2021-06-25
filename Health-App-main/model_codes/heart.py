#%%

import matplotlib
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#EDA
from collections import Counter
# import pandas_profiling as pp
# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')


#%%

data = pd.read_csv('../data/heart.csv')
data.head()


#%%

cols = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']       
X = df[cols]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

y_test.unique()

#%%

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#%%

m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))
disp = plot_confusion_matrix(svc, X_test, y_test, display_labels=['normal', 'diseases'], cmap=plt.cm.Blues)
disp.ax_.set_title('Support Vector Classifier:')
print('Confusion Matrix')
# plt.show()

joblib.dump(svc,"../Heart_API/heart_model.pkl")

m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(y_test,lr_predict))
disp = plot_confusion_matrix(lr, X_test, y_test, display_labels=['normal', 'diseases'], cmap=plt.cm.Blues)
disp.ax_.set_title('Logistic Regression:')
print('Confusion Matrix')
# plt.show()


#%%



m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))
disp = plot_confusion_matrix(rf, X_test, y_test, display_labels=['normal', 'diseases'], cmap=plt.cm.Blues)
disp.ax_.set_title('Random Forest:')
print('Confusion Matrix')
# plt.show()

#%%

m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))
# plt.show()


#%%

m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))
disp = plot_confusion_matrix(dt, X_test, y_test, display_labels=['normal', 'diseases'], cmap=plt.cm.Blues)
disp.ax_.set_title('DecisionTreeClassifier:')
print('Confusion Matrix')
# plt.show()

#%%







