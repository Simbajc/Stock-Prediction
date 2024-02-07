import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import tensorflow as tf
import numpy as np
#Importing the Libraries
import pandas as PD
# import NumPy as np
# %matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras. Models import Sequential
from keras. Layers import Dense
import keras. Backend as K
from Keras. Callbacks import EarlyStopping
from Keras. Optimisers import Adam
from Keras. Models import load_model
from Keras. Layers import LSTM
from Keras. utils.vis_utils import plot_model

























df = pd.read_csv("./MSFT Stock History Jan 30, 2019 - Jan 30, 2024.csv",  names=["Date","Open", "High", "Low", "Close", "Adj Close",  "Volume"]).iloc[1:,:]
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


startDate = '2021-01-30'
endDate = '2024-01-30'
# filters data based on dates
stockData = df[df['Date'].dt.strftime('%Y-%m-%d').between(startDate, endDate)]
stockData["High"] = pd.to_numeric(stockData["High"])
stockData["Low"] = pd.to_numeric(stockData["Low"])
stockData["Close"] = pd.to_numeric(stockData["Close"])
stockData["Open"] = pd.to_numeric(stockData["Open"])
stockData["Adj Close"] = pd.to_numeric(stockData["Adj Close"])


# Formatting stock graph ##############################################################################################################
# x axis data
priceDate = stockData['Date']

# y-axises data
closingPrice = stockData["Close"]
OpeningPrice = stockData["Open"] 


# chart label and style
plt.plot_date(priceDate, closingPrice, linestyle='--', color='r')
plt.title("Microsoft Stock History", fontweight="bold")
plt.ylabel('Stock Price')
plt.xlabel('Date of price')


# Formatting the x-axis
plt.xlabel('Date of price')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))


# plt.figure(figsize=(15,10))
plt.plot(priceDate, stockData["High"], label='High')
plt.plot(priceDate, stockData["Low"], label='Low')
plt.legend()

# change the amount of ticks (for every other)
plt.yticks(np.arange(0, max(stockData["High"]), 10))
plt.xticks(rotation=30)



stockData["Adj Close"].plot()
plt.show()







