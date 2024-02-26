import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import tensorflow as tf
import keras
import numpy as np
#Importing the Libraries
import pandas as pd
import matplotlib. pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn import linear_model
from matplotlib.ticker import MaxNLocator
from keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K


# Managing data ##########################################################################################
df = pd.read_csv("./MSFT Stock History Jan 30, 1990 - Jan 30, 2024.csv", header=None, na_values=['null'],parse_dates=True, infer_datetime_format=True, index_col=[0], names= ["Date","Open","High","Low","Close","Adj Close","Volume"] )


startDate = '2019-01-30'
endDate = '2024-01-30'
# filters data based on dates
stockData = df[ (df.index > startDate) & (df.index < endDate) ]

#convert data to float
stockData["Open"] = stockData["Open"].astype(float)
stockData["Close"] = stockData["Close"].astype(float)
stockData["High"] = stockData["High"].astype(float)
stockData["Low"] = stockData["Low"].astype(float)
stockData["Adj Close"] = stockData["Adj Close"].astype(float)
stockData["Volume"] = stockData["Volume"].astype(float)
stockData.index = pd.to_datetime(stockData.index, format= "%Y-%m-%d")

stockData["Tomorrow"] = stockData["Close"].shift(-1)
stockData["Target"] = (stockData["Tomorrow"] > stockData["Close"]).astype(int) # stock went up the next day
stockData = stockData[ (stockData.index > "1990-01-01")].copy()




# Training the ML Model #####################################################################################################
# Citation: https://www.youtube.com/watch?v=1O_BenficgE

    #Deals with data for ML Model ###########################################################################################
# create a new dataframe with only the 'close column'
data = stockData["Close"]
# convert the dataframe to a numpy array change the array to a 2D array instead of a 1D
dataset = data.values.reshape(-1, 1)

trainingDataLen = math.ceil( len(dataset) * 0.8)
print(f"# of rows of data: {trainingDataLen}")

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataset)
print(scaledData)

#create the training data set
#create the scaled training data set
trainData = scaledData[0:trainingDataLen, :]
#split the data into xTrain and yTrain data sets

xTrain = [] # independent variables
yTrain = [] # dependent variables

for i in range(60, len(trainData)):
    xTrain.append(trainData[i-60:i, 0])
    yTrain.append(trainData[i, 0])
    if i <= 60:
        print(xTrain)
        print(yTrain)
        print()

#convert xTrain and yTrain to numpy arrays to train LSTM model
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
#Reshape the data from 2D to 3D for LSTM
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
print(xTrain.shape)

    #Deals with model its self ####################################################################################################
# models architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(xTrain, yTrain, batch_size=1, epochs=1)

#create testing dataset
#create new array containing scaled values from index 1543 to 2003
testData = scaledData[trainingDataLen-60:, :]
#create the data set xTest and yTest
xTest = []
yTest = dataset[trainingDataLen:, :] # value want model to predict
for i in range(60, len(testData)):
    xTest.append(testData[i-60:i, 0])

#convert the data to numpy array
xTest = np.array(xTest)

#Reshape the data 2D -> 3D
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# get the model predicted price values
predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions) #predictions to contain same values as yTest dataset

# Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean(predictions - yTest)**2 )
print("rsme: " + str(rmse))
print(stockData.head())
print("data")
print(data)
#plot the data
train = pd.DataFrame()
train = stockData["Close"][:trainingDataLen].to_frame()
valid = pd.DataFrame()
valid = stockData["Close"][trainingDataLen:].to_frame()
valid['Predictions'] = predictions
#visulize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price Stock', fontsize=18 )
plt.plot(train["Close"])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc= 'lower right')
plt.show()

validAndPredict = pd.DataFrame()
validAndPredict["Actual"] = valid["Close"]
validAndPredict["Predictions"] = valid["Predictions"]
# Show the valid and predicted price
print(valid.columns)
print(valid.head())
print(validAndPredict.head())
