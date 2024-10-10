# Stock-Price-Predicition

This repository contains a Python project for predicting stock prices using machine learning techniques. The project employs various regression models and performs data preprocessing, feature engineering, and model evaluation to forecast stock prices.

Stock price prediction is a challenging task due to the complex and volatile nature of the financial markets. This project demonstrates how to use historical stock data to build a predictive model using Python. The model predicts future stock prices by analyzing historical trends and applying machine learning algorithms.

Data Preprocessing
The raw stock data is preprocessed to handle missing values, and the date column is converted to a datetime format. The dataset is then sorted by date to maintain the sequence of events.

Feature Engineering
Several features are engineered to enhance the predictive power of the model:

Lag Features: Previous stock prices (Close_lag_X).
Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
Rolling Standard Deviation and Rolling Volume Sum.
Volume Change and Volume Ratio.
Relative Strength Index (RSI).
Bollinger Bands.
On-Balance Volume (OBV).
These features capture trends and patterns in the stock data that are useful for prediction.

The project uses a linear regression model as a baseline to predict stock prices. The steps include:

Splitting the dataset into training and testing sets.
Imputing missing values.
Scaling the features.
Training the model on the training data.
Evaluating the model's performance using Mean Squared Error (MSE).
The evaluation metrics provide insight into how well the model performs on both training and testing data.

The project outputs the following metrics to assess the model's performance:

Training Set Mean Squared Error (MSE) 
Testing Set Mean Squared Error (MSE)

These metrics help gauge the accuracy of the predictions and the model's generalizability.


