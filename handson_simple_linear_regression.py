# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:02:40 2018

@author: amitsaini
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv("Car_age_data")

#Set independent variable 
X = dataset.iloc[:,:-1].values

#set dependent variable
y=dataset.iloc[:,1].values

#spliting data into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results￼
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('KM Run vs Age (Training set)')
plt.xlabel('Car Age')
plt.ylabel('KM Run')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('KM Run vs Age (Test set)')
plt.xlabel('Car Age')
plt.ylabel('KM Run')
plt.show()

