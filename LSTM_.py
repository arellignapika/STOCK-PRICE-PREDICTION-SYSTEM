
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load and Preprocess Data
stock_data = pd.read_csv("./Stock_price_dataset/TCS.csv")
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index("Date", inplace=True)
print("the stock price data of TCS ")
print(stock_data)

# Visualize Historical Close Prices
plt.figure(figsize=(16, 6))
plt.title('Historical Close Prices')
plt.plot(stock_data['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (INR)', fontsize=18)
plt.show()

# Feature Scaling
close_prices = stock_data.filter(['Close'])
scaled_close_prices = MinMaxScaler(feature_range=(0, 1)).fit_transform(close_prices.values)

# Prepare Training Dataset
train_len = int(np.ceil(len(close_prices) * 0.95))
train_data = scaled_close_prices[0:train_len, :]

# Reshape Data for LSTM
X_train, y_train = [], []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(X_train)
        print(y_train)
        print()

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build and Compile LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
history = model.fit(X_train, y_train, batch_size=32, epochs=50)

# Prepare Test Data
test_data = scaled_close_prices[train_len - 60:, :]
X_test = []

# Reshape Test Data for Prediction
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make Predictions and Inverse Scaling
predictions = model.predict(X_test)
predictions = MinMaxScaler(feature_range=(0, 1)).fit(close_prices.values).inverse_transform(predictions)

training_set = close_prices[:train_len]
validation_set = close_prices[train_len:]
validation_set['Predictions'] = predictions

# Evaluate RMSE
rmse = np.sqrt(np.mean(((predictions - validation_set['Close'].values) ** 2)))
print("Root Mean Squared Error (RMSE):", rmse)

# Plot Predictions
plt.figure(figsize=(16, 6))
plt.title('Stock Price Prediction Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (INR)', fontsize=18)
plt.plot(training_set['Close'], label='Training Data')
plt.plot(validation_set[['Close', 'Predictions']], label=['Actual Prices', 'Predictions'])
plt.legend(loc='lower right')
plt.show()

# Display validation set
print(validation_set)
