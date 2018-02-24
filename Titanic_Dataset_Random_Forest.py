import pandas as pd 
import os 
import numpy as np
from sklearn.preprocessing import Imputer , StandardScaler , LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
os.chdir("/Users/madhavkhosla/Desktop")
#----------importing-done-----------------#
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#-------------importing-data--------------#
X_train  = pd.DataFrame([train_data['PassengerId'] , train_data['Pclass'] , train_data['Sex'],train_data['Age']]).transpose()
y_train = pd.DataFrame(train_data['Survived'])
X_test = pd.DataFrame([test_data['PassengerId'] , test_data['Pclass'] , test_data['Sex'],test_data['Age']]).transpose()
y_test = pd.DataFrame(train_data['Survived'])
#--------Label-Encoding-our-data----------#
encoder_X = LabelEncoder()
X_train['Sex'] = encoder_X.fit_transform(X_train['Sex'])
encoder_test = LabelEncoder()
X_test['Sex'] = encoder_test.fit_transform(X_test['Sex'])
#-------------cleaning-the-Data-----------#
imputer_X = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer_X = imputer_X.fit(np.array(X_train['Age']).reshape(-1,1))
X_train['Age']= imputer_X.transform(np.array(X_train['Age']).reshape(-1,1))
#-------------cleaning-testing------------#
imputer_test = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer_test = imputer_test.fit(np.array(X_test['Age']).reshape(-1,1))
X_test['Age']= imputer_test.transform(np.array(X_test['Age']).reshape(-1,1))
#----------scaling-our-data---------------#
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)
#--------making-our-model-----------------#
Classifier = DecisionTreeClassifier(criterion = 'gini' , random_state = 0)
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
#---------checking-the-accuracy------------#
cm = confusion_matrix(y_test['Survived'][:418],y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1] + cm[0][1]+cm[1][0])
