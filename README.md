# EX-01 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![model](https://github.com/BALA291/basic-nn-model/assets/120717501/be73d037-3cdf-4bc2-872e-2afd61033ad5)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: BALAMURUGAN B
### Register Number: 212222230016
```python

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _=default()
gc= gspread.authorize(creds)
worksheet=gc.open('DLdata1').sheet1
data= worksheet.get_all_values()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1
dataset1=dataset1.astype({'Input1' : 'float'})
dataset1=dataset1.astype({'Output1' : 'float'})
dataset1.head()
X=dataset1[["Input1"]].values
Y=dataset1[["Output1"]].values
X_train,X_test,Y_train,Y_test=(train_test_split(X,Y,test_size=0.33,random_state=20))
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1= Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
history=ai_brain.fit(X_train1,Y_train,epochs = 3500)
ai_brain.summary()
loss=pd.DataFrame(ai_brain.history.history)
loss.plot()
X_test=Scaler.transform(X_test)
X_a=[[21]]
X_a_1=Scaler.transform(X_a)
ai_brain.predict(X_a_1)
```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
