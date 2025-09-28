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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#------------------------------------------------------------------------------
# Function to load and process data with additional features
#------------------------------------------------------------------------------
def load_process_data(company, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Volume'],
                split_method='date', split_ratio=0.8, split_date=None, 
                scale_features=True, save_local=True, data_dir='./stock_data/'):
    
    if save_local and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    filename = f"{company}_{start_date}_{end_date}.pkl"
    filepath = os.path.join(data_dir, filename)

    if save_local and os.path.exists(filepath):
        print("Loading saved data from local directory...")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    print("Downloading data from Yahoo Finance...")
    data = yf.download(company, start_date, end_date)

    if data.empty:
        raise ValueError(f"No data found for {company} from {start_date} to {end_date}")

    data = data.ffill().bfill()
    data = data[features]

    if split_method == 'date':
        if split_date is None:
            split_index = int(len(data) * split_ratio)
            split_date = data.index[split_index]
        else:
            split_date = pd.to_datetime(split_date)
        
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]

    elif split_method == 'ratio':
        split_index = int(len(data) * split_ratio)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
    else:
        raise ValueError("split_method must be 'date' or 'ratio'")
    
    feature_scalers = {}

    if scale_features:
        print("Scaling features...")
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_values = train_data[feature].values.reshape(-1, 1)
            scaler.fit(train_values)
            train_data[feature] = scaler.transform(train_values).flatten()
            test_values = test_data[feature].values.reshape(-1, 1)
            test_data[feature] = scaler.transform(test_values).flatten()
            feature_scalers[feature] = scaler

    processed_data = {
        'train_data': train_data,
        'test_data': test_data,
        'feature_scalers': feature_scalers,
        'features': features
    }

    if save_local:
        print("Saving data locally for future use...")
        with open(filepath, 'wb') as f:
            pickle.dump(processed_data, f)
    
    return processed_data

#------------------------------------------------------------------------------
# Enhanced function to create configurable deep learning models
#------------------------------------------------------------------------------
def create_model(sequence_length, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
                loss="mean_squared_error", optimizer="adam", bidirectional=False,
                activation='tanh', recurrent_activation='sigmoid'):
    
    model = Sequential()
    
    if isinstance(units, int):
        units = [units] * n_layers
    elif len(units) != n_layers:
        raise ValueError("Length of units list must match n_layers")
    
    for i in range(n_layers):
        is_first_layer = i == 0
        is_last_layer = i == n_layers - 1
        
        # Configure return_sequences
        return_sequences = not is_last_layer or n_layers == 1
        
        # Configure cell arguments based on cell type
        cell_args = {
            'units': units[i],
            'activation': activation,
            'return_sequences': return_sequences
        }
        
        # Add specific arguments for LSTM/GRU
        if cell in [LSTM, GRU]:
            cell_args['recurrent_activation'] = recurrent_activation
        
        # Create the layer with input shape for first layer
        if is_first_layer:
            if bidirectional:
                layer = Bidirectional(cell(**cell_args), input_shape=(sequence_length, n_features))
            else:
                layer = cell(**cell_args, input_shape=(sequence_length, n_features))
        else:
            if bidirectional:
                layer = Bidirectional(cell(**cell_args))
            else:
                layer = cell(**cell_args)
        
        # Add layer to model
        model.add(layer)
        
        # Add dropout after each layer except the last one
        if not is_last_layer:
            model.add(Dropout(dropout))
    
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    return model

#------------------------------------------------------------------------------
# Function to prepare training data
#------------------------------------------------------------------------------
def prepare_training_data(data, target_column, sequence_length):
    """
    Prepare training data sequences for time series prediction.
    
    Parameters:
    data (DataFrame): Input data
    target_column (str): Column to predict
    sequence_length (int): Number of time steps to use for prediction
    
    Returns:
    tuple: (x_train, y_train) arrays
    """
    scaled_data = data[target_column].values
    
    x_data = []
    y_data = []
    
    for x in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[x-sequence_length:x])
        y_data.append(scaled_data[x])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data, y_data

#------------------------------------------------------------------------------
# Function to evaluate model performance
#------------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, scaler, model_name=""):
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    model: Trained Keras model
    x_test: Test features
    y_test: True test values
    scaler: Scaler for inverse transformation
    model_name (str): Name of the model for display
    
    Returns:
    dict: Dictionary containing evaluation metrics
    """
    predictions = model.predict(x_test)
    
    # Inverse transform predictions and actual values
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    rmse = np.sqrt(mse)
    
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions_inv.flatten(),
        'actual': y_test_inv.flatten()
    }
    
    print(f"\n{model_name} Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return metrics

#------------------------------------------------------------------------------
# Function to plot model comparisons
#------------------------------------------------------------------------------
def plot_model_comparisons(metrics_dict, actual_prices, company):
    """
    Plot comparison of predictions from different models.
    
    Parameters:
    metrics_dict (dict): Dictionary containing metrics for each model
    actual_prices (array): Actual stock prices
    company (str): Company name for title
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: All predictions
    plt.subplot(2, 1, 1)
    plt.plot(actual_prices, color="black", linewidth=2, label="Actual Price")
    
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        plt.plot(metrics['predictions'], color=colors[i % len(colors)], 
                linestyle='--', label=f"{model_name} (RMSE: {metrics['rmse']:.2f})")
    
    plt.title(f"{company} Share Price - Model Comparison")
    plt.xlabel("Time")
    plt.ylabel("Share Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Metrics comparison
    plt.subplot(2, 1, 2)
    model_names = list(metrics_dict.keys())
    rmse_values = [metrics_dict[name]['rmse'] for name in model_names]
    mape_values = [metrics_dict[name]['mape'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.7)
    plt.bar(x + width/2, mape_values, width, label='MAPE (%)', alpha=0.7)
    
    plt.xlabel('Models')
    plt.ylabel('Error Values')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# Function to prepare test data
#------------------------------------------------------------------------------
def prepare_test_data(company, train_start, test_start, test_end, prediction_days, scaler, price_column='Close'):
    """
    Prepare test data for model evaluation.
    
    Parameters:
    company (str): Stock symbol
    train_start (str): Training start date
    test_start (str): Test start date
    test_end (str): Test end date
    prediction_days (int): Number of days for prediction sequence
    scaler: Fitted scaler object
    price_column (str): Column name for price data
    
    Returns:
    tuple: (x_test, y_test, actual_prices) arrays
    """
    # Download test data
    test_data = yf.download(company, test_start, test_end)
    actual_prices = test_data[price_column].values
    
    # Download full data range for creating sequences
    full_data = yf.download(company, train_start, test_end)
    full_prices = full_data[price_column].values
    
    # Prepare model inputs
    model_inputs = full_prices[len(full_prices) - len(test_data) - prediction_days:]
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    # Create test sequences
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_test, actual_prices, test_data

#------------------------------------------------------------------------------
# Function to display stock data using candlestick chart
#------------------------------------------------------------------------------
def plot_candlestick_chart(data, company, n_days=1, chart_style='charles', 
                         title_suffix="", figsize=(12, 8)):
    
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
# Function to clean data for charting
#------------------------------------------------------------------------------
def clean_data_for_charting(data):
    """
    Clean and prepare data for charting functions.
    
    Parameters:
    data (DataFrame): Raw data from yfinance
    
    Returns:
    DataFrame: Cleaned data ready for charting
    """
    # Make a copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Handle MultiIndex columns (common with yfinance)
    if isinstance(cleaned_data.columns, pd.MultiIndex):
        # Flatten the MultiIndex columns
        cleaned_data.columns = ['_'.join(col).strip() for col in cleaned_data.columns.values]
        # If there's only one ticker, remove the ticker suffix
        cleaned_data.columns = [col.split('_')[0] if '_' in col else col for col in cleaned_data.columns]
    
    # Forward fill to replace NaN values with previous values
    cleaned_data = cleaned_data.ffill()
    
    # Backward fill to handle any remaining NaN values at the beginning
    cleaned_data = cleaned_data.bfill()
    
    # Ensure all required columns are numeric
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col in cleaned_data.columns:
            # Make sure we're working with a Series
            if isinstance(cleaned_data[col], pd.Series):
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            else:
                # If it's not a Series, convert to Series first
                cleaned_data[col] = pd.to_numeric(pd.Series(cleaned_data[col]), errors='coerce')
            
    # If Volume column exists, ensure it's numeric too
    if 'Volume' in cleaned_data.columns:
        if isinstance(cleaned_data['Volume'], pd.Series):
            cleaned_data['Volume'] = pd.to_numeric(cleaned_data['Volume'], errors='coerce')
        else:
            cleaned_data['Volume'] = pd.to_numeric(pd.Series(cleaned_data['Volume']), errors='coerce')
        # Fill any remaining NaN in Volume with 0 (since volume can't be negative)
        cleaned_data['Volume'] = cleaned_data['Volume'].fillna(0)
    
    return cleaned_data

#------------------------------------------------------------------------------
# Fixed function for error analysis visualization
#------------------------------------------------------------------------------
# Replace the plot_error_analysis function with this fixed version:
def plot_error_analysis(actual_prices, predictions, model_name, company):
    """
    Plot comprehensive error analysis for model predictions.
    
    Parameters:
    actual_prices (array): Actual stock prices
    predictions (array): Model predictions
    model_name (str): Name of the model
    company (str): Company stock symbol
    """
    # Calculate errors
    errors = actual_prices - predictions
    
    # Ensure errors is a 1D array
    errors = np.array(errors).flatten()
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Predictions vs actual
    plt.subplot(2, 2, 1)
    plt.plot(actual_prices, color='black', label='Actual Prices', linewidth=2)
    plt.plot(predictions, color='red', label=f'{model_name} Predictions', linestyle='--')
    plt.title(f'Best Model: {model_name} - Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    plt.subplot(2, 2, 2)
    plt.plot(errors, color='blue', label='Prediction Errors')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.title(f'Prediction Errors - {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(2, 2, 3)
    # Convert errors to proper format for histogram
    errors_flat = errors.flatten() if errors.ndim > 1 else errors
    
    # Use proper histogram function call
    n, bins, patches = plt.hist(errors_flat, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns comparison - FIXED VERSION
    plt.subplot(2, 2, 4)
    # Ensure arrays are 1D and compatible for division
    actual_flat = actual_prices.flatten() if actual_prices.ndim > 1 else actual_prices
    pred_flat = predictions.flatten() if predictions.ndim > 1 else predictions
    
    # Calculate returns with proper array shapes
    if len(actual_flat) > 1 and len(pred_flat) > 1:
        actual_returns = np.diff(actual_flat) / actual_flat[:-1]
        predicted_returns = np.diff(pred_flat) / pred_flat[:-1]
        
        cumulative_actual = np.cumsum(actual_returns)
        cumulative_predicted = np.cumsum(predicted_returns)
        
        plt.plot(cumulative_actual, label='Actual Cumulative Returns', color='black')
        plt.plot(cumulative_predicted, label='Predicted Cumulative Returns', color='red', linestyle='--')
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor returns calculation', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Cumulative Returns Comparison')
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------
def main():
    # Configuration
    COMPANY = 'CBA.AX'
    TRAIN_START = '2020-01-01'
    TRAIN_END = '2023-08-01'
    TEST_START = '2023-08-02'
    TEST_END = '2024-07-02'
    PREDICTION_DAYS = 60
    PRICE_VALUE = "Close"
    
    # Load and process data
    print("Loading and processing data...")
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
    
    train_data = processed_data['train_data']
    feature_scalers = processed_data['feature_scalers']
    close_scaler = feature_scalers[PRICE_VALUE]
    
    # Prepare training data
    print("Preparing training data...")
    x_train, y_train = prepare_training_data(train_data, PRICE_VALUE, PREDICTION_DAYS)
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Define model configurations for experimentation
    model_configs = [
        {
            'name': 'LSTM_3Layer',
            'cell': LSTM,
            'units': [50, 50, 50],
            'n_layers': 3,
            'dropout': 0.2,
            'epochs': 25,
            'batch_size': 32
        },
        {
            'name': 'GRU_2Layer',
            'cell': GRU,
            'units': [64, 32],
            'n_layers': 2,
            'dropout': 0.3,
            'epochs': 20,
            'batch_size': 32
        },
        {
            'name': 'SimpleRNN_2Layer',
            'cell': SimpleRNN,
            'units': [50, 50],
            'n_layers': 2,
            'dropout': 0.2,
            'epochs': 30,
            'batch_size': 64
        },
        {
            'name': 'Bidirectional_LSTM',
            'cell': LSTM,
            'units': [50, 50],
            'n_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'epochs': 25,
            'batch_size': 32
        },
        {
            'name': 'Deep_LSTM',
            'cell': LSTM,
            'units': [100, 80, 60, 40],
            'n_layers': 4,
            'dropout': 0.3,
            'epochs': 30,
            'batch_size': 16
        }
    ]
    
    # Prepare test data once for all models
    print("Preparing test data...")
    x_test, actual_prices, test_data = prepare_test_data(
        COMPANY, TRAIN_START, TEST_START, TEST_END, 
        PREDICTION_DAYS, close_scaler, PRICE_VALUE
    )
    
    # Train and evaluate models
    models = {}
    metrics_results = {}
    
    print("\nTraining and evaluating models...")
    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Training {config['name']}...")
        print(f"{'='*50}")
        
        # Create model
        model = create_model(
            sequence_length=PREDICTION_DAYS,
            n_features=1,
            units=config['units'],
            cell=config['cell'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            bidirectional=config.get('bidirectional', False),
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        print(f"Model architecture: {config['name']}")
        model.summary()
        
        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.1,
            verbose=1,
            shuffle=False  # Important for time series data
        )
        
        models[config['name']] = model
        
        # Evaluate model
        metrics = evaluate_model(model, x_test, actual_prices, close_scaler, config['name'])
        metrics_results[config['name']] = metrics
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{config["name"]} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{config["name"]} - Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Compare all models
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    plot_model_comparisons(metrics_results, actual_prices, COMPANY)
    
    # Find best model
    best_model_name = min(metrics_results.keys(), 
                         key=lambda x: metrics_results[x]['rmse'])
    best_model = models[best_model_name]
    best_metrics = metrics_results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best RMSE: {best_metrics['rmse']:.4f}")
    print(f"Best MAPE: {best_metrics['mape']:.2f}%")
    
    # Make prediction for next day using best model
    real_data = x_test[-1:]  # Use the last sequence from test data
    prediction = best_model.predict(real_data)
    prediction = close_scaler.inverse_transform(prediction)
    
    print(f"\nNext Day Price Prediction using {best_model_name}: ${prediction[0][0]:.2f}")
    
    # Display final comparison table
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
    print(f"{'-'*80}")
    for model_name, metrics in metrics_results.items():
        print(f"{model_name:<20} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['mape']:<10.2f}%")
    
    #------------------------------------------------------------------------------
    # Candlestick and Boxplot Charts
    #------------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VISUALIZATION CHARTS")
    print(f"{'='*60}")
    
    # Get data for candlestick and boxplot charts
    candlestick_data = yf.download(COMPANY, TEST_START, TEST_END)
    
    # Clean the data for charting
    cleaned_data = clean_data_for_charting(candlestick_data)
    
    # Plot Candlestick Charts
    print("\nDisplaying Daily Candlestick Chart...")
    plot_candlestick_chart(cleaned_data, COMPANY, n_days=1, 
                          title_suffix="(Test Period)")
    
    print("\nDisplaying 3-Day Candlestick Chart...")
    plot_candlestick_chart(cleaned_data, COMPANY, n_days=3, 
                          title_suffix="(Test Period)")
    
    # Plot Boxplot Charts
    print("\nDisplaying 5-Day Moving Window Boxplot Chart...")
    plot_boxplot_chart(cleaned_data, COMPANY, window_size=5, column='Close',
                      chart_title_suffix="(Test Period)")
    
    print("\nDisplaying 10-Day Moving Window Boxplot Chart...")
    plot_boxplot_chart(cleaned_data, COMPANY, window_size=10, column='Close',
                      chart_title_suffix="(Test Period)")
    
    # Additional visualization: Error analysis using best model
    print(f"\nDisplaying Error Analysis for Best Model: {best_model_name}")
    best_predictions = metrics_results[best_model_name]['predictions']
    plot_error_analysis(actual_prices, best_predictions, best_model_name, COMPANY)

if __name__ == "__main__":
    main()