# firstly we import all the libraries required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# then we import the training set
# RNN is not going to validate on the test set
# test set does not exist in the training of the RNN
# then we will introduce the test set in the end so it can make predictions of the future
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# create training set, including input data of the network
# need to make array, thus we put 1:2, so that it makes a numpy array of 1 column
# .values transforms it into a numpy array
training_set = dataset_train.iloc[:, 1:2].values

# then we ned to apply feature scaling
# standardization and normalization, for this RNN, it's more relevant to use 
# normalization (x-xmin)/(xmax-xmin), sigmoid function at output in RNN recommend
# normalization, use min max scaler class in preprocessing module
# feature_range makes every values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# keep original dataset before applying sc
training_set_scaled = sc.fit_transform(training_set)

# going to create datat structure that the RNN needs to remember before predicting
# called the number of time steps, wrong number of time steps lead to nonsense or 
# overfitting
# now create a data structure with 60 time steps and 1 output
# at each time t, the RNN is going to see the 60 previous time steps and itself
# try to predict the next output, at time t+1
# 1 time step leads to overfitting, model wasn't learning anythin
# 20 time steps weren't able to learn the trends
# 60 is 60 previous financial days, 3 months to predict the stock price the next day
# create x_train which will be the inputs of the neural network, y_train is the output
# basically each financial day X_train will include 60 previous stock price
# y_train is the one next day, but need to do it for every time t
X_train = []
y_train = []

# use for loop to populate the list, to start doing this we need the first 60 first
# or the 60th data set we will start our for loop, need to get range from i-60 to i
# which contains 60 produced stock price at time t
# append the values of the numpy array to the X_train
# memorising previous time steps to predict time t+1
# for y_train we use i because it is lower bound and it is inclusive
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
# since X_train and y_train are lists, we need to convert them to np arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# we need to reshape the data such that it has more dimension to the data
# which is the unit: the number of predictors/indicators (in this case it's the
# open google stock price), then we can add more predictors, but we're not going
# to add more here
# reshape function adds a dimension to the numpy array
# we need to reshape into a 3D stack of arrays
# third parameter is the number of indicators (dependent vars)
# first dimension corresponds to the number of stock prices (no of rows)
# second dimension is the number of time_steps
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# now we build the RNN, a stacked LSTM with several LSTM layers
# add regularization, use most useful RNN
# first we import all the libraries
# Dropout is to add dropout regularization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# now we initialize this RNN as a sequence of layers
# we call it regressor because we expect a continous stream of values and therefore
# we're doing some regression: continous output prediction
# classifier is for predicting a category or class
regressor = Sequential()

# add first LSTM layer and some Dropout regularization to avoid overfitting
# first argument is units, how many cells or memory units in this LSTM model, its basically the neurons
# we can increase dimensionality by increasing no of neurons as well, 50 neurons we'll be able to predict the ups and downs
# return sequences is true because we have stacked LSTM, default is true
# input_shape is the shape of the input in X_train, in 3D refer to keras lib
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# add dropout rate in dropout regularization step
# 20% of neuron will be dropped during training
regressor.add(Dropout(0.2))

# then we add second layer and some dropout regularization as well
# don't need to add input_shape for second layer onwards
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# add the third LSTM layer, keep 50 neurons, and keep true because we're adding another layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# fourth LSTM layer
# since this is the last LSTM layer, return_sequence is False, remove parameter altogether
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# adding output layer for RNN, since the layer is a normal classic layer, create with dense class
# since output is only 1, units = 1, 
regressor.add(Dense(units = 1))

# now we need to compile the RNN and fit the neural network to the training set
# for RNN, an RMSprop is recommended instead of adam, its an advanced stochastic
# gradient descent network, but we'll just use adam here since it's always a good choice
# since it's a regression problem, we have to use the mean squared error for the loss
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# then we need to fit the training set to the regressor model
# now make connection from model with training set
# batch_size is the number of batches it will be divided into
# after which weights will be updated, every 32 we update the losses (batch deep learning)
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# then we will make predictions and visualizing the results
# now we get the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# now we need to get the predicted stock price from the RNN of Jan 2017
# since we're using 60 previous test results, then we need to apply it to jan 2017 as well
# need both the training and test set, this is why it's quite complicated
# we need to concatenate the test set into the training set
# we cannot straight concatenate, we have to apply the fit_transform to scale the concatenation
# so we can get the scaled stock price but we'll change the actual test values
# actual test values just leave it as they are
# concatenate the original data frames
# we'll then scale it to get the predictions, don't change the test values
# leading to the most relevant result
# let's concatenate the original data frames now
# concatenate the lines along the vertical axis
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# second step is to get the input, or the 60 previous days
# please note the lower and upper bound (i-60) and (i-40)
# len total - len test set is the index of the first member of dataset_test
# then convert to np array with .values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# next step we need to locate the inputs and reshape it such that it is in the format
# of a 3D array with parameters that can be inputted into keras
inputs = inputs.reshape(-1, 1)

# then we need to make it such that it can fit into the RNN, make it the right format
# before that we need to scale the inputs first
# we only need to transform, don't need to fit as it's already fitted to the prev dataset
inputs = sc.transform(inputs)

# create inputs for the test sets
X_test = []

# test set only contains 20
# we will get the 60 previous inputs for each of the stock prices of jan 2017
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    
# now we put into an np array
X_test = np.array(X_test)

# now we transform it into a 3D format
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# now we predict the test values
predicted_stock_price = regressor.predict(X_test)

# then we have to inverse our scaling to get the actual values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the results for the pred vs real stock price
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# some observations:
# model cannot react to spikes in stock prices
# but it can react to smooth changes (small increments)
# this is correct for down or up