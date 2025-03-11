# Stock Price Prediction Using Machine Learning

This project focuses on predicting future stock prices for NASDAQ (NDAQ) using machine learning techniques implemented in Python. The primary goal is to develop a regression-based machine learning model capable of analyzing historical stock data and providing reliable predictions for future stock prices. The project demonstrates how data preprocessing, feature engineering, and regression algorithms can be combined to build a simple yet effective predictive model.

Project Overview
The project leverages historical stock price data from NASDAQ (NDAQ) to train and evaluate a machine learning regression model. Using Python and common data science libraries, the project walks through the complete data science workflow: data preprocessing, feature engineering, model training, and evaluation. The core model used in this implementation is a baseline Linear Regression algorithm, which serves as an introduction to more advanced machine learning techniques that can be explored in future iterations.

Dataset Details
The dataset used in this project is named NDAQ_data.csv and contains historical stock prices for the NASDAQ exchange. The time range of the dataset spans from [INSERT START DATE] to [INSERT END DATE], and includes key features such as Date, Open, High, Low, Close, and Volume. The data has been cleaned to handle missing values, and date columns have been converted into appropriate datetime formats to preserve the chronological order of events.

Methodology and Feature Engineering
The raw stock price data underwent preprocessing to ensure quality and consistency. Missing values were addressed, and the data was sorted by date to maintain chronological integrity. Extensive feature engineering was performed to enhance the dataset’s predictive power. Specifically, lag features were created to incorporate previous closing prices into the prediction process. Additionally, technical indicators commonly used in financial analysis were computed, including Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Rolling Standard Deviations, Rolling Volume Sums, Relative Strength Index (RSI), Bollinger Bands, and On-Balance Volume (OBV). These engineered features were selected to capture trends, volatility, and momentum within the stock data, providing valuable signals for the regression model.

After feature engineering, the dataset was split into training and testing sets to evaluate the model's ability to generalize. A Linear Regression model was then trained on the processed data, and its predictions were assessed using standard regression metrics.

Model Results and Evaluation
The Linear Regression model produced consistent evaluation metrics on both the training and testing datasets. The Mean Squared Error (MSE) for both datasets was 0.07076469854616008, and the Root Mean Squared Error (RMSE) was 0.2660163501481818. These results indicate that the model performs consistently across different subsets of the data and does not exhibit signs of overfitting. In addition to MSE and RMSE, the Mean Absolute Error (MAE) was calculated, providing a direct interpretation of the model’s average prediction error. The Mean Absolute Error (MAE) was 0.189837. This score reflects the average absolute difference between the actual stock prices and the predicted prices, further validating the model’s performance.

The model’s predictions were compared against actual stock prices, and the results are summarized in a sample from the output dataframe. For example, the actual stock price for index 76 was 32.15, while the predicted price was 32.189034. Similarly, for index 1026, the actual price was 70.73, and the model predicted 70.980264. Additional examples include index 43 with an actual price of 29.18 versus a predicted price of 28.962124, index 666 with 52.62 versus 52.445774, and index 529 with 49.52 versus 49.539626. These predictions demonstrate the model’s capability to produce estimates that closely align with actual market values.

Project Structure and Organization
The project directory is structured to ensure clarity and scalability. The data folder contains the input dataset, specifically NDAQ_data.csv. The notebooks directory includes the Jupyter Notebook StockMarketPredictionV1_0.ipynb, which presents the full analysis and implementation. A src directory is designated for future development, where reusable Python scripts can be placed. Additionally, a reports folder has been created to store output files, including visualizations such as predicted versus actual price charts. The README.md file and requirements.txt file reside in the root directory for easy access and installation guidance.

Installation and Environment Setup
To run this project locally, the repository can be cloned using the command git clone https://github.com/chazbrahma/Stock-Price-Prediction.git. After navigating into the project directory, dependencies should be installed. These dependencies include Python libraries such as pandas, numpy, matplotlib, scikit-learn, and ta. A requirements.txt file should be provided in the root directory to simplify installation. Once the environment is configured, the Jupyter Notebook can be opened from the notebooks directory and executed to reproduce the results.

Usage Instructions
The primary usage of the project is through the Jupyter Notebook StockMarketPredictionV1_0.ipynb. This notebook includes all steps from data loading, preprocessing, feature engineering, model training, evaluation, and visualization of results. Users can modify the notebook to load different datasets, experiment with different technical indicators, or implement alternative regression models for comparison.

Future Enhancements
Future work on this project could include the implementation of more advanced machine learning models such as Long Short-Term Memory (LSTM) networks, Gradient Boosting Machines (XGBoost), or AutoRegressive Integrated Moving Average (ARIMA) models to capture complex patterns in the stock price data. Additional features could include live data streaming and prediction by integrating APIs such as yFinance or Alpaca. The project could also be deployed as an interactive web application using Streamlit, providing users with a dashboard to input custom stock tickers and receive predictions in real-time. Hyperparameter tuning using tools like GridSearchCV or Optuna could further optimize model performance.

Author Information
This project was developed by Chazin Brahma. For additional projects and contributions, visit my GitHub at https://github.com/chazbrahma. Professional inquiries and networking are welcome on LinkedIn at https://www.linkedin.com/in/chazin-brahma-684197292/.
