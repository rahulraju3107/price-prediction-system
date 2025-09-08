# Stock Price Prediction System

## Overview

This project is developed for COS30018 Intelligent Systems at Swinburne University and uses an LSTM neural network to predict stock prices. It fetches data via yfinance, trains a model with train.py, and evaluates predictions with test.py. Outputs include performance metrics and plots of actual vs. predicted prices. See the C1_Setup directory for the detailed setup report.

## Links

- [C1_Setup](./C1_Setup)
- [C2_DataProcessing1](./C2_DataProcessing1)
- [C3_DataProcessing2](./C3_DataProcessing2)

## Setup and Running

Follow these steps to run the project:

1. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib pandas scikit-learn tensorflow yfinance
   ```

2. **Run the Program**:
   - Train the model:
     ```bash
     python train.py
     ```
   - Test and evaluate predictions:
     ```bash
     python test.py
     ```

3. **Check Outputs**:
   - Model weights are saved in the `results/` directory.
   - Prediction data is saved in the `csv-results/` directory.
   - A plot comparing actual vs. predicted prices is displayed.
