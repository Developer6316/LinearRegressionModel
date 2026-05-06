#Author: Developer6316
#Machine Learning Model for Linear Regression
#importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

#Loading the data

df = pd.read_csv("salesdata.csv", skiprows=1, names=['tv', 'radio', 'newspaper', 'sales'])
print(df.head())

#Adding the features and output

features =  df.loc[:, 'tv':'newspaper']
output = df['sales']

#Splitting the data into training and testing sets

x_train,x_test,y_train,y_test = train_test_split(features,output,test_size=0.1,random_state=3)

#Training the model and making predictions

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("The actual sales value for the test set is:", y_test)
print("The coefficients are:", model.coef_)
print("The intercept is:", model.intercept_)
print("The predicted sales value for the test set is:", y_pred)

#To Make it Predict for a new data point
print("The predicted sales value for the new data point is:", model.predict([[100,100,100]]))

#pickle
import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)