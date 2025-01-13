#%% md
# # Dependencies
#%%
import requests

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

import yfinance as yf
from datetime import date

%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

tqdm.pandas()
#%%
# !where python
#%% md
# # Get Data
#%%
end_date = date.today().strftime("%Y-%m-%d")
print(end_date)
start_date = '2021-02-01'

df = yf.download('BNB-USD', start=start_date, end=end_date)

# Inspect the data
df.head()
#%%
df.info()
#%%
df.to_csv('../data/BNB-USD.csv', index=False) # save data
#%%
close_data = df.Close
# close_data
# close_data.tail()
#%%
close_data.info()
#%% md
# We rename the column and index for easy analysis
#%%
close_data.rename(columns={'BNB-USD': 'close'}, inplace=True)
close_data.index.name = 'date'

#%% md
# # Visualization
#%%
# a. Trend Plot
plt.figure(figsize=(20, 5))
plt.plot(close_data.index, close_data['close'], label='BNB-USD Price', color='blue')
plt.title('BNB-USD Price Trend')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
#%%
# b. Histogram of Prices
plt.figure(figsize=(8, 6))
sns.histplot(close_data['close'], bins=30, kde=True, color='blue')
plt.title('BNB-USD Price Distribution')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.grid()
plt.show()
#%%
# c. Box Plot to Visualize Outliers
plt.figure(figsize=(8, 6))
sns.boxplot(y=close_data['close'], color='green')
plt.title('BNB-USD Price Box Plot')
plt.ylabel('Close Price')
plt.grid()
plt.show()
#%%
# d. Rolling Average and Volatility
close_data['rolling_mean'] = close_data['close'].rolling(window=30).mean()  # 30-day rolling mean
close_data['rolling_std'] = close_data['close'].rolling(window=30).std()    # 30-day rolling std

plt.figure(figsize=(20, 5))
plt.plot(close_data.index, close_data['close'], label='Close Price', color='blue')
plt.plot(close_data.index, close_data['rolling_mean'], label='30-Day Rolling Mean', color='red')
plt.fill_between(close_data.index,
                 close_data['rolling_mean'] - close_data['rolling_std'],
                 close_data['rolling_mean'] + close_data['rolling_std'],
                 color='gray', alpha=0.2, label='Rolling Std Dev')
plt.title('BNB-USD Price with Rolling Mean and Std Dev')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
#%%
close_data.info()
#%% md
# # Feature Engineering
#%%
# Feature Engineering
def create_features(data):
    """
    Create lag and rolling features for the time series data.
    """
    data['lag_1'] = data['close'].shift(1)
    data['lag_2'] = data['close'].shift(2)
    data['lag_3'] = data['close'].shift(3)
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['close'].rolling(window=3).std()
    data['rolling_mean_7'] = data['close'].rolling(window=7).mean()
    return data

close_data = create_features(close_data)

# Drop rows with NaN values due to feature creation
close_data.dropna(inplace=True)

# Define features (X) and target (y)
X = close_data[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_7']]
y = close_data['close'].values

#%% md
# # Train XGBoost model
#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
print(X_train.head(1))

# Convert to DMatrix (XGBoost-specific format)
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
test_dmatrix = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'eval_metric': 'rmse'
}
#%%
# Train the model
xgb_model = xgb.train(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=100,
    evals=[(train_dmatrix, 'train'), (test_dmatrix, 'eval')],
    early_stopping_rounds=10
)
#%%
# Make predictions
y_pred = xgb_model.predict(test_dmatrix)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
#%%
# Plot actual vs predicted
plt.figure(figsize=(20, 5))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('XGBoost Daily Close Price Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
#%% md
# # Hyperparameter tuning: XGBoost
#%%
# Define a function for hyperparameter tuning
def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning for XGBoost using GridSearchCV.
    """
    # Create an XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_regressor,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Return the best model and parameters
    return grid_search.best_estimator_, grid_search.best_params_
#%%
# Perform hyperparameter tuning
best_model, best_params = tune_xgboost(X_train, y_train)

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f"Best Parameters: {best_params}")
print(f"RMSE with Best Parameters: {rmse_best}")

# Compare with the default model
default_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
default_model.fit(X_train, y_train)
y_pred_default = default_model.predict(X_test)
rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
print(f"RMSE with Default Parameters: {rmse_default}")

# Compare results
print(f"Improvement in RMSE: {rmse_default - rmse_best}")
#%%
# Plot the predictions of the tuned model
plt.figure(figsize=(20, 5))
plt.plot(y_test, label='Actual', color='blue')
# plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_best, label='Predicted (Tuned)', color='red')
plt.title('XGBoost Daily Close Price Prediction (Tuned)')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
#%% md
# # LSTM
#%%
# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM input (samples, time_steps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#%%
# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # Take the last time step output
        return predictions
#%%
# Hyperparameter Tuning (GridSearchCV won't work directly with PyTorch, we will manually try different parameters)
def train_lstm_model(hidden_layer_size, num_epochs=100, batch_size=32, learning_rate=0.001):
    model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=hidden_layer_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(X_train_tensor)
        loss = loss_function(output.squeeze(), y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred.numpy()))
    return rmse
#%%
# Hyperparameter Tuning: Trying different configurations manually
hidden_layer_sizes = [100, 200, 300, 450, 600]
learning_rates = [ 0.01, 0.1, 0.25, 0.5, 0.65]
num_epochs = 50

best_rmse = float('inf')
best_params = {}

for hidden_layer_size in hidden_layer_sizes:
    for learning_rate in learning_rates:
        print(f"Training with hidden layer size: {hidden_layer_size}, learning rate: {learning_rate}")
        rmse = train_lstm_model(hidden_layer_size, num_epochs, learning_rate=learning_rate)
        print(f"RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'hidden_layer_size': hidden_layer_size, 'learning_rate': learning_rate}

print(f"Best Hyperparameters: {best_params}")
print(f"Best RMSE: {best_rmse}")
#%%
# Final Model with Best Hyperparameters
final_model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=best_params['hidden_layer_size'])
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

# Train the final model
final_model.train()
for epoch in range(num_epochs):
    final_model.train()
    optimizer.zero_grad()
    output = final_model(X_train_tensor)
    loss = loss_function(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate the final model
final_model.eval()
with torch.no_grad():
    y_pred_final = final_model(X_test_tensor).squeeze()

# Calculate RMSE for the final model
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final.numpy()))
print(f"Final RMSE: {final_rmse}")

# Plot actual vs predicted
plt.figure(figsize=(20, 5))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred_final.numpy(), label='Predicted', color='green')
plt.title('LSTM Daily Close Price Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
#%% md
# # Comparison plot
#%%
# Plot actual vs predicted
plt.figure(figsize=(20, 5))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred_best, label='XGBoost', color='red')
plt.plot(y_pred_final.numpy(), label='LSTM', color='green')
plt.title('Daily Close Price Prediction: Model Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
#%% md
# # Save Model
#%% md
# From the above and the RSME value, we see that the XGBoost model is the best. We save the model and deploy it later.
#%%
# Save the best XGBoost model to a file
best_model.save_model('../model/xgb_model.json')
print("Model saved as 'xgb_model.json'")

#%% md
# # Predict BNB Price
#%%
url = 'http://localhost:9100/bnbpriceprediction'
#%%
price_attributes = {
         "lag_1" : 320.1,
         "lag_2" : 315.5,
         "lag_3" : 310.2,
         "rolling_mean_3" : 315.27,
         "rolling_std_3" : 5.04,
        "rolling_mean_7" : 318.54,
     }
#%%
features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3", "rolling_mean_7"]
X_new = np.array([[price_attributes[feature] for feature in features]])
X_new
#%%
model = '../model/xgb_model.json'
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model)
print("Model loaded successfully")
#%%
# Predict using the loaded model
new_prediction = xgb_model.predict(X_new)
print(new_prediction[0])
#%%
import requests
response = requests.post(url, json=price_attributes).json()
print(response)

#%%
# import numpy as np
# import xgboost as xgb
#
# # User input
# price_attributes = {
#     "lag_1": 320.1,
#     "lag_2": 315.5,
#     "lag_3": 310.2,
#     "rolling_mean_3": 315.27,
#     "rolling_std_3": 5.04,
#     "rolling_mean_7": 318.54,
# }
#
# # Step 1: Ensure features are in the correct order
# # This must match the training data feature order
# features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3", "rolling_mean_7"]
#
# # Convert the dictionary values into a list in the correct order
# X_new = np.array([[price_attributes[feature] for feature in features]])
#
# # Step 2: Load the trained XGBoost model
# model = '../model/xgb_model.json'
# xgb_model = xgb.XGBRegressor()
# xgb_model.load_model(model)
# print("Model loaded successfully")
#
# # Step 3: Predict the target value
# # new_dmatrix = xgb.DMatrix(X_new)
# predicted_price = xgb_model.predict(X_new)
#
# # Step 4: Output the result
# print(f"Predicted price: {predicted_price[0]:.2f}")

#%%
