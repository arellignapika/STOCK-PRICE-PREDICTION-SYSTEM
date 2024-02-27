# STOCK-PRICE-PREDICTION-SYSTEM


This project involves building a Long Short-Term Memory (LSTM) neural network to predict stocks closing prices using historical data. The model is implemented using TensorFlow's Keras API.

## Overview

- The dataset used for this project: TCS (Tata Consultancy Services) , CIPLA , ASIANPAINTS , WIPRO , TITAN  stock prices.
- The model predicts stock prices based on historical closing prices.
- LSTM (Long Short-Term Memory) architecture is used for sequence modeling.

## Description

The `LSTM.py` script performs the following steps:

- Loads stock price data from a CSV file.
- Visualizes the historical closing prices.
- Applies Min-Max scaling to the price data.
- Prepares training and test datasets.
- Defines and trains an LSTM model on the training data.
- Predicts stock prices on the test dataset.
- Visualizes the predicted prices alongside actual prices.
- Calculates and prints the Root Mean Squared Error (RMSE) of the predictions.

## Dependencies

To run this script, you'll need the following Python libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow

## Installation

Ensure you have Python installed on your system. If you don't have the required libraries, you can install them using pip:

pip install numpy pandas matplotlib scikit-learn tensorflow

## Results
- TCS STOCK CLOSING DAYPRICE PREDICTION GRAPH AND PREDICTION PRICE COMPARED TO STOCK MARKET CLOSING DAY PRICE.
![TCS_PREDICTION_GRAPH](./images/tcs_graph.png "Example Image")
![TCS_PREDICTION_values](./images/tcs_prediction.png "Example Image")
- CIPLA STOCK CLOSING DAY PRICE PREDICTION GRAPH AND PREDICTION PRICE VALUES COMPARED TO STOCK MARKET CLOSING DAY PRICE.
![CIPLA_PREDICTION_GRAPH](./images/ciplagraph.png "Example Image")
![CIPLA_PREDICTION_VALUES](./images/cipla_prediction.png "Example Image")


### Model Performance

- The trained LSTM model demonstrates accurate predictions on the TCS and cipla stock dataset.
- The model has been tested on five different datasets i.e TCS (Tata Consultancy Services) , CIPLA , ASIANPAINTS , WIPRO , TITAN  stock prices and it consistently provides good predictions.




