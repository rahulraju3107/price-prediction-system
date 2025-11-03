# Stock Price Prediction System

## Overview

This project is developed for **COS30018 Intelligent Systems** at Swinburne University. It predicts stock prices using different recurrent neural network (RNN) architectures such as LSTM, GRU, and SimpleRNN.

Key features include:
- A configurable model builder (create_model()) that supports multiple architectures, variable layers/units, dropout, bidirectionality, and custom optimizers/ loss functions.
- Controlled experiments comparing LSTM, GRU, SimpleRNN, and Bidirectional LSTM models.
- Evaluation of training performance, prediction accuracy, and trade-offs between speed and complexity.
- Advanced prediction problems including multistep, multivariate, and combined multivariate multistep prediction.

## Links

- [C1_Setup](./C1_Setup)
- [C2_DataProcessing1](./C2_DataProcessing1)
- [C3_DataProcessing2](./C3_DataProcessing2)
- [C4_MachineLearning1](./C4_MachineLearning1)
- [C5_MachineLearning2](./C5_MachineLearning2)
- [C6_MachineLearning3](./C6_MachineLearning3)
- [C7_Extension](./C7_Extension)

## Setup and Running

Follow these steps to run the project:

1. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib pandas scikit-learn tensorflow yfinance mplfinance pickle
   ```

2. **Training the model**:
   - Train the model:
     ```bash
     python train.py
     ```
   - Test and evaluate predictions:
     ```bash
     python test.py
     ```
3. **Run Program**:
     ```bash
     python stock_prediction.py 
     ```

5. **Outputs**:
   - Model weights are saved in the `results/` directory.
   - Prediction data is saved in the `csv-results/` directory.
   - Plot comparing actual vs. predicted prices is displayed.
   - Candlestick Chart
   - Boxplot Chart
