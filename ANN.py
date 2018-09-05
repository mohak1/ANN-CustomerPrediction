# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:17:49 2018

@author: Mohak

Classification problem to predict whether a customer leaves or stays in a bank.
"""

#importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
data = pd.read_csv('G:\Churn_Modelling.csv')

#we will include indexes from 3-12 because the other features are not at all influencing the cause
#like customerID, surname, etc will not be included.

#x = index from 3-12
#y = index 13
X = data.iloc[:,3:13].values
Y = data.iloc[:,13].values

#encoding the values
#we do not apply the lable encoder to numeric data (0 or 1) in 'Y'
#we will apply the lable encoder to 'X' because we have catagories which are strings in 'X'
#Lable encoder will encode text into numbers

#Encoding the categorical data of independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX1 = LabelEncoder()
X[:, 1] = labelEncoderX1.fit_transform(X[:, 1])

labelEncoderX2 = LabelEncoder()
X[:, 2] = labelEncoderX2.fit_transform(X[:, 2])  

onehotencoder = OneHotEncoder(categorical_features = [1])		#function to apply one hot encoding on column 1
X = onehotencoder.fit_transform(X).toarray()                 #fitting the one hot encoding on X
# we will reomve the first column from X
# because the third column is logically not required
X = X[:, 1:]


#splitting the dataset into testing and training 
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain,YTest = train_test_split(X, Y, test_size=0.2, random_state=0)
#random_state is same as random seed.

#Feature Scaling 
#to make the computation easy
#it eleminates the possibility of one ondependent variable dominating another one
#mean=0, variance=1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                               #look it up
XTrain = sc.fit_transform(XTrain)
#fit_transform performs two steps: 1)it finds the mean and variance for the given data and stored it
    #2) it transforms the given data
XTest = sc.transform(XTest)
#we use only transform() here and not fit_transform() because we have already found mean and var in the previous step
#we could have use the function .fit() to find mean and variance and then used .transform() on both 'xTrain' and 'xTest' 

#Building the ANN
import keras
#Sequential module is required to initialise the neural network
#Dense module is used to add/initialise the layers
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifierName = Sequential()
#classifierName is the object of the sequential class. It is the ANN structure, we will add layers to it.

#adding the first hidden layer
classifierName.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifierName.add(Dropout(rate = 0.1))

#Second hidden layer
classifierName.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifierName.add(Dropout(rate=0.1))
        
#Output layer
classifierName.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling the ANN
classifierName.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the model to the training set
classifierName.fit(x=XTrain, y=YTrain, batch_size=5, epochs=15)

#Evaluating the model
out = classifierName.evaluate(x=XTrain, y=YTrain, verbose=2)
print(out)

#Predicting the probability
Ypred = classifierName.predict(XTest)
Ypred = (Ypred>0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(YTest, Ypred)

#Predicting the homework question
#___________________________________________________________________________________________________________________
file = pd.read_csv('G:\Question.csv')
Xquest = data.iloc[:,3:13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX1 = LabelEncoder()
Xquest[:, 1] = labelEncoderX1.fit_transform(Xquest[:, 1])

labelEncoderX2 = LabelEncoder()
Xquest[:, 2] = labelEncoderX2.fit_transform(Xquest[:, 2])  

onehotencoder = OneHotEncoder(categorical_features = [1])		#function to apply one hot encoding on column 1
Xquest = onehotencoder.fit_transform(Xquest).toarray()                 #fitting the one hot encoding on X
# we will reomve the first column from X
# because the third column is logically not required
Xquest = Xquest[:, 1:]
ans = classifierName.predict(Xquest)
#print(ans)                     answer: 0.06135, it is safe to say that the customer is not leaving
ans = classifierName.predict(Xquest)

#another solution:
#new_pred = classifierName.predict(sc.transform(np.array(all the given values)))

#___________________________________________________________________________________________________________________

#Evaluating the model using K-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#biulding the function that will build the model and compile it. Because KerasClassifier requires it.
def createModel():
    model = Sequential()
    model.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
    model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

classifier = KerasClassifier(build_fn = createModel, epochs = 15, batch_size=5)
#accuracies = cross_val_score(estimator = classifier, X = XTrain, y = YTrain, cv = 10, n_jobs=-1)
accuracies = cross_val_score(estimator = classifier, X = XTrain, y = YTrain, cv = 10)
#estimator takes the classifier that is going to be trained
#cv is the number of folds (or k)
#n_jobs signifies how many cpus will be used to train the model because we will train the model 10 times, it will take time, -1 means all cpus.
#Getting an error while using the n_jobs thing. Might be due to some permission errors
mean = accuracies.mean()
var = accuracies.std()

#Improving the ANN
#Using Dropout
from keras.layers import Dropout

#Parameter Tuning
#using GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#biulding the function that will build the model and compile it. Because KerasClassifier requires it.
def createModel(optimizer):
    #change 'init' to 'kernel_initializer' to remove the warning.
    #warning comes because the name has been changed in the updated version
    model = Sequential()
    model.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
    model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid' ))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
#Here we will not use the KerasClassifier 
classifier = KerasClassifier(build_fn = createModel)

#GridSearch part starts
#creating the dictionary for hyper parameters
parameters = {'batch_size': [40, 32], 
              'epochs': [100, 50],
              'optimizer': ['adam', 'rmsprop']}
#creating the grid search object and specifing the parametes
gs = GridSearchCV(estimator = classifier,
                  param_grid = parameters,
                  scoring = 'accuracy',
                  cv = 10)
#fitting the grid search to the training search
#fit method will return an object so we'll give the same name to it
gs = gs.fit(X=XTrain, y=YTrain) 

#getting the best parameters and accuracy
best_param = gs.best_params_
best_acc = gs.best_score_

#saving the model
#save the model using .h5 extension
#model.save("E:\\WebarchProject\\trainedModel.h5")
