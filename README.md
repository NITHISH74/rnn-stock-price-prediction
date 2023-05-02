# Stock Price Prediction

## AIM:

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset:
Make a Recurrent Neural Network model for predicting stock price using a training and testing dataset. The model will learn from the training dataset sample and then the testing data is pushed for testing the model accuracy for checking its accuracy. The Dataset has features of market open price, closing price, high and low price for each day.


## Neural Network Model:
![image](https://user-images.githubusercontent.com/94164665/235063155-215f6b16-816d-494f-a6b9-a5d69cd90c42.png)
## DESIGN STEPS:

### STEP 1:
Import the necessary tensorflow modules.


### STEP 2:
Load the stock dataset.

### STEP 3:
Fit the model and then predict.


## PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

df_train = pd.read_csv('trainset.csv')
df_train.head(60)

train_set = df_train.iloc[:,1:2].values
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)


X_train

X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train1

model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)
                    ])

model.compile(optimizer='Adam', loss='mae')

model.fit(X_train1,y_train,epochs=100,batch_size=32)

df_test=pd.read_csv("testset.csv")
test_set = df_test.iloc[:,1:2].values

dataset_total = pd.concat((df_train['Open'],df_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)

```


## OUTPUT:

### True Stock Price, Predicted Stock Price vs Time:
![image](https://user-images.githubusercontent.com/94164665/235061311-041e753d-1c9a-4203-bab1-2f0f6427ebea.png)


### Mean Square Error:
![image](https://user-images.githubusercontent.com/94164665/235061546-6d0c603f-507c-40c3-bceb-a683fcbca369.png)


## RESULT:
A Recurrent Neural Network model for stock price prediction is developed.
