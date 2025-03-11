# Stock Price Prediction Using Machine Learning

This project focuses on predicting future stock prices for NASDAQ (NDAQ) using machine learning techniques. The primary objective is to analyze historical stock data and apply regression models, along with advanced feature engineering, to create a predictive framework. The entire workflow is implemented in Python and demonstrated within a Jupyter Notebook titled StockMarketPredictionV1_0.ipynb.

Project Overview
The aim of this project is to forecast the future stock prices of the NASDAQ (NDAQ) stock by utilizing historical data and machine learning regression techniques. The methodology includes data preprocessing, engineering of relevant features derived from technical indicators, model training using a baseline linear regression model, and the evaluation of the model’s predictive performance. This project demonstrates a step-by-step approach to handling raw stock market data and converting it into actionable insights through predictive modeling.

Features and Methodology
The dataset used in this project, NDAQ_data.csv, contains historical stock prices of NASDAQ. The time range covered by this data extends from [INSERT START DATE] to [INSERT END DATE] (you'll need to fill in the dates from your dataset). The dataset includes standard stock attributes such as Date, Open, High, Low, Close, and Volume.

The project begins with data preprocessing. The raw data undergoes cleansing where missing values are imputed, and the date column is converted into a proper datetime format. The dataset is then sorted in chronological order to ensure the integrity of the time series analysis.

Following the preprocessing, extensive feature engineering is performed to enrich the dataset. Several lag features are created to capture the influence of prior closing prices on future predictions. Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) are calculated to smooth out price trends. Additionally, measures of volatility such as Rolling Standard Deviation and Rolling Volume Sum are incorporated. Advanced technical indicators like Relative Strength Index (RSI), Bollinger Bands, and On-Balance Volume (OBV) are also generated. These features are specifically chosen to encapsulate trends, momentum, and volatility in the data, providing the model with more predictive power.

Once the feature engineering is complete, the data is prepared for modeling. The dataset is split into training and testing sets, ensuring the model can be validated on unseen data. Feature scaling and imputation are applied to ensure consistency. A linear regression model is then trained on the training set. This model is used as a baseline to establish the feasibility of predicting future stock prices using relatively simple machine learning techniques.

The evaluation of the model is conducted using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on both the training and testing datasets. The results demonstrate consistency between the training and testing sets. Specifically, the model achieves a Mean Squared Error (MSE) of 0.07076469854616008 and a Root Mean Squared Error (RMSE) of 0.2660163501481818 on both datasets. This suggests that the linear regression model generalizes reasonably well but also highlights the potential for improvement through more advanced algorithms such as LSTM, XGBoost, or ARIMA.

Project Structure and Files
The project is organized into several folders for clarity and maintainability. The data folder contains the input dataset, specifically NDAQ_data.csv. The notebooks folder contains the Jupyter Notebook, StockMarketPredictionV1_0.ipynb, which includes the entire project code, analysis, and results. A src folder is intended for future use, where modular Python scripts such as data_preprocessing.py, feature_engineering.py, and model_training.py can be placed to improve code organization and reusability. The reports folder is designed to store output files, including plots and figures. For example, an actual_vs_predicted.png visualization can be stored here to demonstrate the model's performance visually. The root directory contains the README.md file (this file) and a requirements.txt file that lists the necessary Python dependencies.

Installation and Setup
To run this project locally, first clone the repository using the command git clone https://github.com/chazbrahma/Stock-Price-Prediction.git and navigate into the project directory. The project dependencies can be installed via a requirements.txt file, which should include packages such as pandas, numpy, matplotlib, scikit-learn, and ta. Once the environment is set up, you can open and run the Jupyter Notebook by navigating to the notebooks folder and executing jupyter notebook StockMarketPredictionV1_0.ipynb.

Usage
The Jupyter Notebook guides you through the entire process, from data loading and preprocessing to feature engineering and model evaluation. Users can adapt the notebook to load different datasets, adjust feature engineering parameters, or implement additional models for comparison. To run the notebook, make sure your working directory is set correctly, and the necessary libraries are installed.

Results
The model evaluation on the NASDAQ stock (NDAQ) demonstrates a Mean Squared Error (MSE) of 0.07076469854616008 and a Root Mean Squared Error (RMSE) of 0.2660163501481818 on both the training and testing datasets. These results indicate that the model performs consistently, but as a baseline model, it leaves room for further optimization and enhancement.

Future Work
Future improvements to this project include implementing more complex models such as LSTM or XGBoost, which are better suited for capturing nonlinear relationships in time series data. There is also the potential to integrate live data fetching through APIs like yFinance or Alpaca and to deploy the predictive model using Streamlit for an interactive user interface. Hyperparameter tuning using GridSearchCV or Optuna is another logical next step to improve model performance. Additionally, creating visualizations such as predicted versus actual price plots would enhance the project's clarity and user engagement.

Author
This project was created by Chazin Brahma. You can find more of my work on GitHub at https://github.com/chazbrahma or connect with me on LinkedIn at https://www.linkedin.com/in/chazin-brahma-684197292/.
