# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#------------------------------------------------------------------------------
# Function to load and process data with additional features
#------------------------------------------------------------------------------
def load_process_data(company, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Volume'],
                split_method='date', split_ratio=0.8, split_date=None, 
                scale_features=True, save_local=True, data_dir='./stock_data/'):
    
    """
    Load and process stock data with multiple features.

    Parameters:
    company (str): Stock symbol (such as 'CBA.AX' in this case)
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    features (list): List of features to include from the dataset
    split_method (str):  How to split data - 'date' or 'ratio'
    split_ratio (float): Ratio for train/ test split (0.0 to 1.0)
    split_date (str): Specific date to split when using 'date' method
    scale_features (bool): Whether to scale the feature columns
    save_local (bool): Whether to save/ load data locally
    data_dir (str): Directory to save/ load data

    Returns:
    dict: Dictionary containing processed data and scalers
    """
    
    # Create a directory locally if it doesn't exist
    if save_local and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename
    filename = f"{company}_{start_date}_{end_date}.pkl"
    filepath = os.path.join(data_dir, filename)

    # Check if data already exists locally
    if save_local and os.path.exists(filepath):
        print("Loading saved data from local directory...")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # Download data from Yahoo finance
    print("Downloading data from Yahoo Finance...")
    data = yf.download(company, start_date, end_date)

    # Verify if data is loaded
    if data.empty:
        raise ValueError(f"No data found for {company} from {start_date} to {end_date}")

    # Processing missing values (NaN)
    # Replace NaN with previous value - forward fill
    data = data.ffill()
    # Update remaining NaN values at beginning - back fill
    data = data.bfill()
    data = data[features]  # Select features

    # Split data into train and test data sets
    if split_method == 'date':
        # use a specific date to split
        if split_date is None:
            # if no split date provided, calculate based on ratio
            split_index = int(len(data) * split_ratio)
            split_date = data.index[split_index]
        else:
            split_date = pd.to_datetime(split_date)
        
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]

    elif split_method == 'ratio':
        # split ratio (80% train, 20% test)
        split_index = int(len(data) * split_ratio)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
    else:
        raise ValueError("split_method must be 'date' or 'ratio'")
    
    # Initialise dict to store scalers
    feature_scalers = {}

    # Scale the features if requested
    if scale_features:
        print("Scaling features...")
        for feature in features:
            # create a scaler for this feature
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # fit scaler on training data only
            train_values = train_data[feature].values.reshape(-1, 1)
            scaler.fit(train_values)

            # transform both training and test data
            train_data[feature] = scaler.transform(train_values).flatten()
            test_values = test_data[feature].values.reshape(-1, 1)
            test_data[feature] = scaler.transform(test_values).flatten()

            # store scaler for future access
            feature_scalers[feature] = scaler

    # Prepare the data to return
    processed_data = {
        'train_data': train_data,
        'test_data': test_data,
        'feature_scalers': feature_scalers,
        'features': features
    }

    # Save data locally
    if save_local:
        print("Saving data locally for future use...")
        with open(filepath, 'wb') as f:
            pickle.dump(processed_data, f)
    
    return processed_data

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# Load and process the data parameters
processed_data = load_process_data(
    company=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    features=['Open', 'High', 'Low', 'Close', 'Volume'],
    split_method='date',
    split_date='2023-01-01',
    scale_features=True,
    save_local=True
)

# Extract data
train_data = processed_data['train_data']
feature_scalers = processed_data['feature_scalers']

# Get the scaler for the Close price
PRICE_VALUE = "Close"
close_scaler = feature_scalers[PRICE_VALUE]

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
# Prepare the scaled data for the LSTM model
scaled_data = train_data[PRICE_VALUE].values

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) 

# Prediction of the next closing value of the stock price
# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data_new = yf.download(COMPANY, TEST_START, TEST_END)

# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data_new[PRICE_VALUE].values

# Prepare test data similar to original approach
# Get the full dataset (training + test) for creating sequences
full_data = yf.download(COMPANY, TRAIN_START, TEST_END)
full_prices = full_data[PRICE_VALUE].values

# Scale the full dataset using the same scaler
model_inputs = full_prices[len(full_prices) - len(test_data_new) - PREDICTION_DAYS:]
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = close_scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = close_scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------

real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = close_scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??