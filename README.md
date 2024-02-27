# EX-01 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression types of problems. In this exercise, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.In general regression types of problems gives real numbers as a output. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the required libraries like tensorflow and sklearn.Before building the model,using sklearn data preprocessing and cleaning should be done to the dataset.now split the data into training and testing data.create a neural network model with five input neurons and one hidden layer which consist of ten neurons with activation layer relu and with their nodes in them. Now we will fit out dataset and then predict the value.

## Neural Network Model
![neuron model](https://github.com/BALA291/basic-nn-model/assets/120717501/99db3e2c-f02f-4fd9-938f-e7f7e21ff9bb)


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
#IMPORTING REQUIRED LIBRARIES AND READ DATASET
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

#ALLOCATE 'X' AND 'Y' VALUES
X=dataset1[["Input1"]].values
Y=dataset1[["Output1"]].values

#SPLIT THE DATASET INTO TRAINING AND TESTING DATA
X_train,X_test,Y_train,Y_test=(train_test_split(X,Y,test_size=0.33,random_state=20))

#DATA PREPROCESSING USING MINMAX SCALER
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1= Scaler.transform(X_train)

#CREATING THE MODEL
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

#TESTING DATA
X_test=Scaler.transform(X_test)
X_b=[[30]]
X_b_1=Scaler.transform(X_b)
ai_brain.predict(X_b_1)

X_c=[[72]]
X_c_1=Scaler.transform(X_c)
ai_brain.predict(X_c_1)
```
## Dataset Information

![DATASET](https://github.com/BALA291/basic-nn-model/assets/120717501/71702328-e1a8-449f-94ab-35e3fbd37a49)

## OUTPUT

### Training Loss Vs Iteration Plot
![PLOT](https://github.com/BALA291/basic-nn-model/assets/120717501/67fcebc5-2ac7-4f9c-93d4-c502fa208554)

### Test Data Root Mean Squared Error
![Screenshot 2024-02-27 222934](https://github.com/BALA291/basic-nn-model/assets/120717501/4709e773-1f98-427e-a7c6-73ac60110129)


### New Sample Data Prediction
![predicted](https://github.com/BALA291/basic-nn-model/assets/120717501/5f69fab4-c008-4c64-858a-e67ec8629eb9)

## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
