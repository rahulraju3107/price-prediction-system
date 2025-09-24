import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import pickle
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional

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
# Function to create configurable deep learning models
#------------------------------------------------------------------------------
def create_model(sequence_length, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
                loss="mean_squared_error", optimizer="adam", bidirectional=False, steps_ahead=1):
    """
    Create a configurable deep learning model for time series prediction.
    
    This function creates a Sequential model with configurable layers, cell types,
    number of units, dropout rates, and other hyperparameters. It supports LSTM,
    GRU, and SimpleRNN cells, and can optionally make them bidirectional.
    
    Parameters:
    sequence_length (int): Length of input sequences (number of time steps)
    n_features (int): Number of features in each time step
    units (int or list): Number of units in each layer. If int, all layers use the same.
                         If list, must match n_layers length.
    cell (class): Type of recurrent cell to use (LSTM, GRU, or SimpleRNN)
    n_layers (int): Number of recurrent layers to create
    dropout (float): Dropout rate to apply after each layer (0.0 to 1.0)
    loss (str): Loss function for model compilation
    optimizer (str): Optimizer for model compilation
    bidirectional (bool): Whether to wrap layers in Bidirectional wrapper
    steps_ahead (int): Number of steps to predict (1 for single step, >1 for multistep)
    
    Returns:
    model: Compiled Keras Sequential model
    """
    
    # Create the Sequential model
    model = Sequential()
    
    # Handle units parameter - convert to list if it's an integer
    if isinstance(units, int):
        units = [units] * n_layers
    elif len(units) != n_layers:
        raise ValueError("Length of units list must match n_layers")
    
    # Add layers based on parameters
    for i in range(n_layers):
        # First layer needs input shape specification
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units[i], return_sequences=True), 
                                       input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units[i], return_sequences=True, 
                              input_shape=(sequence_length, n_features)))
        # Last layer should not return sequences (unless it's the only layer)
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units[i], return_sequences=False)))
            else:
                model.add(cell(units[i], return_sequences=False))
        # Hidden layers should return sequences
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units[i], return_sequences=True)))
            else:
                model.add(cell(units[i], return_sequences=True))
        
        # Add dropout after each layer
        model.add(Dropout(dropout))
    
    # Add final dense layer for prediction - output size depends on steps_ahead
    model.add(Dense(units=steps_ahead))
    
    # Compile the model with specified optimizer and loss
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

#------------------------------------------------------------------------------
# Function to create training data for different prediction types
#------------------------------------------------------------------------------
def create_training_data(scaled_data, prediction_days, steps_ahead=1, multivariate=False):
    """
    Create training data for different prediction problems.
    
    Parameters:
    scaled_data: The scaled data array
    prediction_days (int): Number of days to look back
    steps_ahead (int): Number of days to predict into the future (1 for single step)
    multivariate (bool): Whether to create multivariate data
    
    Returns:
    x_train, y_train: Training data arrays
    """
    x_train = []
    y_train = []
    
    if multivariate:
        # For multivariate prediction
        for x in range(prediction_days, len(scaled_data) - steps_ahead + 1):
            x_train.append(scaled_data[x-prediction_days:x])
            if steps_ahead == 1:
                y_train.append(scaled_data[x, 3])  # 3 is the index for Close price
            else:
                y_train.append(scaled_data[x:x+steps_ahead, 3])  # Predict next k days of Close prices
    else:
        # For univariate prediction
        for x in range(prediction_days, len(scaled_data) - steps_ahead + 1):
            x_train.append(scaled_data[x-prediction_days:x])
            y_train.append(scaled_data[x:x+steps_ahead])  # Predict next k days
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    if not multivariate:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train

#------------------------------------------------------------------------------
# Function to display stock data using candlestick chart
#------------------------------------------------------------------------------
def plot_candlestick_chart(data, company, n_days=1, chart_style='charles', 
                         title_suffix="", figsize=(12, 8)):
    """
    Display stock market data using a candlestick chart.
    
    Parameters:
    data (DataFrame): DataFrame containing OHLC data (Open, High, Low, Close) with DatetimeIndex
    company (str): Company stock symbol for chart title
    n_days (int): Number of trading days each candle should represent (default=1 for daily candles)
    chart_style (str): Style of the chart (e.g., 'charles', 'yahoo', 'nightclouds', 'binance')
    title_suffix (str): Additional text to add to chart title
    figsize (tuple): Figure size as (width, height)
    
    Returns:
    None: Displays the chart
    
    Note: This function requires the 'mplfinance' library to be installed.
    """
    
    # If n_days > 1, we need to resample the data to create candles for n-day periods
    if n_days > 1:
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Create a numeric index for grouping (0, 1, 2, 3, ...)
        numeric_index = np.arange(len(data))
        
        # Create a new column to group every n_days rows using the numeric index
        data['group'] = numeric_index // n_days  # Integer division to create groups
        
        # Aggregate data for each group to form new candles
        # Open = first open price in the group
        # High = maximum high price in the group
        # Low = minimum low price in the group  
        # Close = last close price in the group
        # Volume = sum of volumes in the group (if volume column exists)
        
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        
        # Add Volume to aggregation if it exists in the data
        if 'Volume' in data.columns:
            agg_dict['Volume'] = 'sum'
            
        # Group by the 'group' column and aggregate
        resampled_data = data.groupby('group').agg(agg_dict)
        
        # Create new datetime index - use the last date in each group as the candle date
        # Fixed: Removed include_groups parameter for compatibility with older pandas versions
        new_index = data.groupby('group').apply(lambda x: x.index[-1])
        resampled_data.index = new_index
        
        # Use the resampled data for plotting
        plot_data = resampled_data
        period_text = f"({n_days}-Day Periods)"
    else:
        # Use original data for daily candles
        plot_data = data
        period_text = "(Daily)"
    
    # Create the candlestick chart using mplfinance
    mpf.plot(plot_data,                    # The DataFrame with OHLC data
             type='candle',                # Chart type: 'candle' for candlestick
             style=chart_style,            # Visual style of the chart
             title=f'{company} Stock Price {period_text} {title_suffix}',  # Chart title
             ylabel='Price ($)',           # Y-axis label
             ylabel_lower='Volume',        # Y-axis label for volume subplot (if shown)
             volume='Volume' in plot_data.columns,  # Show volume subplot if Volume column exists
             figsize=figsize,                # Size of the figure
             show_nontrading=False)        # Don't show gaps for non-trading days
    
    # Display the chart
    plt.show()

#------------------------------------------------------------------------------
# Function to display stock data using boxplot chart for moving windows
#------------------------------------------------------------------------------
def plot_boxplot_chart(data, company, window_size=5, column='Close', 
                      chart_title_suffix="", figsize=(12, 8)):
    """
    Display stock market data using boxplot chart for moving windows.
    
    Parameters:
    data (DataFrame): DataFrame containing stock data with DatetimeIndex
    company (str): Company stock symbol for chart title
    window_size (int): Size of the moving window in trading days (default=5 for weekly)
    column (str): Column name to create boxplots for (default='Close')
    chart_title_suffix (str): Additional text to add to chart title
    figsize (tuple): Figure size as (width, height)
    
    Returns:
    None: Displays the chart
    
    Note: Boxplots show the distribution of prices within each moving window,
    including median, quartiles, and outliers.
    """
    
    # Make a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Create a numeric index for grouping (0, 1, 2, 3, ...)
    numeric_index = np.arange(len(data_copy))
    
    # Create a column to identify which window each row belongs to
    # We'll create groups of 'window_size' consecutive days using numeric index
    data_copy['window_group'] = numeric_index // window_size
    
    # Create a list to store data for each window
    window_data = []
    window_labels = []
    
    # Group the data by window_group and create boxplot data for each window
    grouped = data_copy.groupby('window_group')
    
    for name, group in grouped:
        # Extract the values for the specified column
        values = group[column].values
        window_data.append(values)
        
        # Create a label for this window (use the last date in the window)
        last_date = group.index[-1].strftime('%Y-%m-%d')
        window_labels.append(last_date)
    
    # Create the boxplot
    plt.figure(figsize=figsize)
    
    # Create boxplot: each box represents the distribution of prices in one window
    box_plot = plt.boxplot(window_data, 
                          labels=window_labels,  # X-axis labels
                          patch_artist=True,     # Fill boxes with color
                          showfliers=True)       # Show outliers
    
    # Customize the appearance
    plt.title(f'{company} {column} Price - {window_size}-Day Moving Window {chart_title_suffix}')
    plt.xlabel('Window End Date')
    plt.ylabel(f'{column} Price ($)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.grid(True, alpha=0.3)  # Add light grid
    
    # Color the boxes (optional)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for i, box in enumerate(box_plot['boxes']):
        box.set_facecolor(colors[i % len(colors)])
    
    # Display the chart
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

#------------------------------------------------------------------------------
# Load Data
#------------------------------------------------------------------------------
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
# Prepare Data for Different Prediction Types
#------------------------------------------------------------------------------
PREDICTION_DAYS = 60

# Univariate single step prediction
print("Setting up univariate single step prediction...")
scaled_data = train_data[PRICE_VALUE].values
x_train_uni, y_train_uni = create_training_data(scaled_data, PREDICTION_DAYS, steps_ahead=1, multivariate=False)

# Multistep prediction
print("Setting up multistep prediction...")
STEPS_AHEAD_MS = 5  # Predict 5 days ahead
x_train_multistep, y_train_multistep = create_training_data(scaled_data, PREDICTION_DAYS, steps_ahead=STEPS_AHEAD_MS, multivariate=False)

# Multivariate prediction
print("Setting up multivariate prediction...")
scaled_multivariate_data = train_data.values  # All features
x_train_multivariate, y_train_multivariate = create_training_data(scaled_multivariate_data, PREDICTION_DAYS, steps_ahead=1, multivariate=True)

# Multivariate multistep prediction
print("Setting up multivariate multistep prediction...")
STEPS_AHEAD_MV_MS = 3  # Predict 3 days ahead
x_train_mv_ms, y_train_mv_ms = create_training_data(scaled_multivariate_data, PREDICTION_DAYS, steps_ahead=STEPS_AHEAD_MV_MS, multivariate=True)

#------------------------------------------------------------------------------
# Build and Train Models
#------------------------------------------------------------------------------
# Univariate single step model
print("Training Original_LSTM...")
model = create_model(
    sequence_length=PREDICTION_DAYS, 
    n_features=1, 
    units=50, 
    cell=LSTM, 
    n_layers=3, 
    dropout=0.2,
    optimizer='adam',
    loss='mean_squared_error',
    steps_ahead=1
)

model.fit(x_train_uni, y_train_uni, epochs=10, batch_size=32)

# Multistep model
print("Training Multistep_Model...")
multistep_model = create_model(
    sequence_length=PREDICTION_DAYS, 
    n_features=1, 
    steps_ahead=STEPS_AHEAD_MS,
    units=50, 
    cell=LSTM, 
    n_layers=2, 
    dropout=0.2
)

multistep_model.fit(x_train_multistep, y_train_multistep, epochs=10, batch_size=32)

# Multivariate model
print("Training Multivariate_Model...")
multivariate_model = create_model(
    sequence_length=PREDICTION_DAYS, 
    n_features=len(train_data.columns),  # Number of features
    units=50, 
    cell=LSTM, 
    n_layers=2, 
    dropout=0.2,
    steps_ahead=1
)

multivariate_model.fit(x_train_multivariate, y_train_multivariate, epochs=10, batch_size=32)

# Multivariate multistep model
print("Training Multivariate_Multistep_Model...")
mv_ms_model = create_model(
    sequence_length=PREDICTION_DAYS, 
    n_features=len(train_data.columns),  # Number of features
    steps_ahead=STEPS_AHEAD_MV_MS,
    units=50, 
    cell=LSTM, 
    n_layers=2, 
    dropout=0.2
)

mv_ms_model.fit(x_train_mv_ms, y_train_mv_ms, epochs=10, batch_size=32)

#------------------------------------------------------------------------------
# Experiment with different model configurations
#------------------------------------------------------------------------------
# Define different model configurations to test
model_configs = [
    {
        'name': 'GRU_Model',
        'cell': GRU,
        'units': [50, 50, 50],
        'n_layers': 3,
        'dropout': 0.2,
        'steps_ahead': 1
    },
    {
        'name': 'RNN_Model',
        'cell': SimpleRNN,
        'units': [50, 50],
        'n_layers': 2,
        'dropout': 0.3,
        'steps_ahead': 1
    },
    {
        'name': 'Bidirectional_LSTM',
        'cell': LSTM,
        'units': [50, 50],
        'n_layers': 2,
        'dropout': 0.2,
        'bidirectional': True,
        'steps_ahead': 1
    }
]

# Store models for later comparison
models = {'Original_LSTM': model}  # Add the original model

# Train and evaluate different models
for config in model_configs:
    print(f"\nTraining {config['name']}...")
    
    # Create model with current configuration
    current_model = create_model(
        sequence_length=PREDICTION_DAYS,
        n_features=1,
        units=config['units'],
        cell=config['cell'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        bidirectional=config.get('bidirectional', False),
        steps_ahead=config['steps_ahead']
    )
    
    # Train the model
    current_model.fit(x_train_uni, y_train_uni, epochs=10, batch_size=32, verbose=1)
    
    # Store the model
    models[config['name']] = current_model

# Add multistep, multivariate, and combined models to the models dictionary
models['Multistep_Model'] = multistep_model
models['Multivariate_Model'] = multivariate_model
models['Multivariate_Multistep_Model'] = mv_ms_model

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data_new = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data_new[PRICE_VALUE].values

# Prepare test data similar to original approach
full_data = yf.download(COMPANY, TRAIN_START, TEST_END)
full_prices = full_data[PRICE_VALUE].values

model_inputs = full_prices[len(full_prices) - len(test_data_new) - PREDICTION_DAYS:]
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = close_scaler.transform(model_inputs)

#------------------------------------------------------------------------------
# Make predictions on test data for all models
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Dictionary to store predictions from all models
all_predictions = {}

# Get predictions from all models
for model_name, model_obj in models.items():
    if 'Multistep' in model_name or 'Multivariate' in model_name:
        # Skip multistep and multivariate models for this test since they have different input/output shapes
        continue
    else:
        predicted_prices = model_obj.predict(x_test)
        predicted_prices = close_scaler.inverse_transform(predicted_prices)
        all_predictions[model_name] = predicted_prices

#------------------------------------------------------------------------------
# Plot the test predictions for all models
#------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")

# Plot predictions from all models
colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, (model_name, predictions) in enumerate(all_predictions.items()):
    plt.plot(predictions, color=colors[i % len(colors)], label=f"Predicted {model_name}")

plt.title(f"{COMPANY} Share Price - Model Comparison")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Plot Candlestick Chart
#------------------------------------------------------------------------------
# Get data for candlestick chart (using test period for demonstration)
candlestick_data = yf.download(COMPANY, TEST_START, TEST_END)

# CLEAN THE DATA BEFORE PLOTTING
# Check if data is empty
if candlestick_data.empty:
    raise ValueError(f"No data found for {COMPANY} from {TEST_START} to {TEST_END}")

# Handle MultiIndex columns (common with yfinance)
# If columns are MultiIndex, convert to regular index
if isinstance(candlestick_data.columns, pd.MultiIndex):
    # Flatten the MultiIndex columns
    candlestick_data.columns = ['_'.join(col).strip() for col in candlestick_data.columns.values]
    # If there's only one ticker, remove the ticker suffix
    candlestick_data.columns = [col.split('_')[0] if '_' in col else col for col in candlestick_data.columns]

# Forward fill to replace NaN values with previous values
candlestick_data = candlestick_data.ffill()

# Backward fill to handle any remaining NaN values at the beginning
candlestick_data = candlestick_data.bfill()

# Ensure all required columns are numeric
required_columns = ['Open', 'High', 'Low', 'Close']
for col in required_columns:
    if col in candlestick_data.columns:
        # Make sure we're working with a Series
        if isinstance(candlestick_data[col], pd.Series):
            candlestick_data[col] = pd.to_numeric(candlestick_data[col], errors='coerce')
        else:
            # If it's not a Series, convert to Series first
            candlestick_data[col] = pd.to_numeric(pd.Series(candlestick_data[col]), errors='coerce')
        
# If Volume column exists, ensure it's numeric too
if 'Volume' in candlestick_data.columns:
    if isinstance(candlestick_data['Volume'], pd.Series):
        candlestick_data['Volume'] = pd.to_numeric(candlestick_data['Volume'], errors='coerce')
    else:
        candlestick_data['Volume'] = pd.to_numeric(pd.Series(candlestick_data['Volume']), errors='coerce')
    # Fill any remaining NaN in Volume with 0 (since volume can't be negative)
    candlestick_data['Volume'] = candlestick_data['Volume'].fillna(0)

# Now plot the cleaned data
print("\nDisplaying Daily Candlestick Chart...")
plot_candlestick_chart(candlestick_data, COMPANY, n_days=1, 
                      title_suffix="(Test Period)")

# Plot 3-day candlestick chart
print("\nDisplaying 3-Day Candlestick Chart...")
plot_candlestick_chart(candlestick_data, COMPANY, n_days=3, 
                      title_suffix="(Test Period)")

#------------------------------------------------------------------------------
# Plot Boxplot Chart
#------------------------------------------------------------------------------
# Plot boxplot chart with 5-day moving windows
print("\nDisplaying 5-Day Moving Window Boxplot Chart...")
plot_boxplot_chart(candlestick_data, COMPANY, window_size=5, column='Close',
                  chart_title_suffix="(Test Period)")

# Plot boxplot chart with 10-day moving windows
print("\nDisplaying 10-Day Moving Window Boxplot Chart...")
plot_boxplot_chart(candlestick_data, COMPANY, window_size=10, column='Close',
                  chart_title_suffix="(Test Period)")

#------------------------------------------------------------------------------
# Predict next day using the original model
#------------------------------------------------------------------------------
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = close_scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

#------------------------------------------------------------------------------
# Test multistep prediction
#------------------------------------------------------------------------------
print("\nTesting multistep prediction...")
# Prepare data for multistep prediction
test_data_multistep = yf.download(COMPANY, TEST_START, TEST_END)
full_data_multistep = yf.download(COMPANY, TRAIN_START, TEST_END)
full_prices_multistep = full_data_multistep[PRICE_VALUE].values

model_inputs_multistep = full_prices_multistep[len(full_prices_multistep) - len(test_data_multistep) - PREDICTION_DAYS:]
model_inputs_multistep = model_inputs_multistep.reshape(-1, 1)
model_inputs_multistep = close_scaler.transform(model_inputs_multistep)

x_test_multistep = []
for x in range(PREDICTION_DAYS, len(model_inputs_multistep) - STEPS_AHEAD_MS + 1):
    x_test_multistep.append(model_inputs_multistep[x - PREDICTION_DAYS:x, 0])

x_test_multistep = np.array(x_test_multistep)
x_test_multistep = np.reshape(x_test_multistep, (x_test_multistep.shape[0], x_test_multistep.shape[1], 1))

# Get multistep predictions
multistep_predictions = multistep_model.predict(x_test_multistep)

# Handle inverse transform for multistep predictions
# The prediction output should already be in shape (batch_size, steps_ahead)
if multistep_predictions.ndim == 2 and multistep_predictions.shape[1] == STEPS_AHEAD_MS:
    # Already in the right shape (batch_size, steps_ahead), so process each step
    reshaped_predictions = multistep_predictions.reshape(-1, 1)
    inverse_predictions = close_scaler.inverse_transform(reshaped_predictions)
    multistep_predictions = inverse_predictions.reshape(-1, STEPS_AHEAD_MS)
else:
    # If shape is different, handle accordingly
    multistep_predictions = close_scaler.inverse_transform(multistep_predictions)

#------------------------------------------------------------------------------
# Test multivariate prediction
#------------------------------------------------------------------------------
print("\nTesting multivariate prediction...")
# Prepare test data for multivariate prediction
full_data_multivariate = yf.download(COMPANY, TRAIN_START, TEST_END)
full_data_multivariate = full_data_multivariate[['Open', 'High', 'Low', 'Close', 'Volume']]

# Scale the data using the same scalers
scaled_test_multivariate = full_data_multivariate.copy()
for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if feature in feature_scalers:
        scaler = feature_scalers[feature]
        scaled_test_multivariate[feature] = scaler.transform(scaled_test_multivariate[feature].values.reshape(-1, 1)).flatten()

# Get the last PREDICTION_DAYS of multivariate data
last_sequence = scaled_test_multivariate.tail(PREDICTION_DAYS).values
last_sequence = last_sequence.reshape(1, PREDICTION_DAYS, -1)

# Predict using the multivariate model
multivariate_prediction = multivariate_model.predict(last_sequence)
# Inverse transform using close_scaler
multivariate_prediction = close_scaler.inverse_transform(multivariate_prediction.reshape(-1, 1)).flatten()

print(f"Multivariate prediction: {multivariate_prediction[0]}")

#------------------------------------------------------------------------------
# Test multivariate multistep prediction
#------------------------------------------------------------------------------
print("\nTesting multivariate multistep prediction...")
# Prepare test data for multivariate multistep prediction
last_sequence_mv_ms = scaled_test_multivariate.tail(PREDICTION_DAYS).values
last_sequence_mv_ms = last_sequence_mv_ms.reshape(1, PREDICTION_DAYS, -1)

# Predict using the multivariate multistep model
mv_ms_prediction = mv_ms_model.predict(last_sequence_mv_ms)

# Handle inverse transform for multivariate multistep predictions
if mv_ms_prediction.ndim == 2 and mv_ms_prediction.shape[1] == STEPS_AHEAD_MV_MS:
    # Already in the right shape (batch_size, steps_ahead)
    reshaped_predictions = mv_ms_prediction.reshape(-1, 1)
    inverse_predictions = close_scaler.inverse_transform(reshaped_predictions)
    mv_ms_prediction = inverse_predictions.reshape(-1, STEPS_AHEAD_MV_MS)
else:
    # Reshape and inverse transform
    mv_ms_prediction = mv_ms_prediction.reshape(-1, STEPS_AHEAD_MV_MS)
    reshaped_predictions = mv_ms_prediction.reshape(-1, 1)
    inverse_predictions = close_scaler.inverse_transform(reshaped_predictions)
    mv_ms_prediction = inverse_predictions.reshape(-1, STEPS_AHEAD_MV_MS)

print(f"Multivariate multistep prediction for {STEPS_AHEAD_MV_MS} days:")
print(mv_ms_prediction[0])