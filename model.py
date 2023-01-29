#Libraries import 

import pandas as pd
import numpy as np
import pickle 
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#Data split
train = pd.read_csv('train.csv') #Training data
test = pd.read_csv('test.csv') #Test data

#Data cleaning
train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1) #Deleting non relevant data
test = test.drop(['Name','Ticket', 'Cabin'], axis=1) #Deleting non relevant data

train.dropna(axis=0, how='any', inplace=True) #Deleting rows with missing data
test.dropna(axis=0, how='any', inplace=True) #Deleting rows with missing data

#Cambio los datos de sexos en números
train['Sex'].replace(['female','male'],[0,1],inplace=True)
test['Sex'].replace(['female','male'],[0,1],inplace=True)

#Cambio los datos de embarque en números
train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

#Data split data vs outcome
X = np.array(train.drop(['Survived'], 1))
y = np.array(train['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

pickle.dump(model, open('model.pkl','wb'))

model2 = pickle.load(open('model.pkl', 'rb'))
print(model2.predict[[]])