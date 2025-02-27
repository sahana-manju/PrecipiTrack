# Rainfall Prediction System

This project utilizes a hybrid model consisting of **LSTM (Long Short-Term Memory)**, **Ridge Regression**, and **XGBoost** to predict rainfall measurements in millimeters, given radar measurements in a region. This hybrid approach has shown to reduce the prediction error by **10%** compared to existing standalone models.

## Overview

The system aims to forecast rainfall based on radar measurements, with an improved accuracy thanks to the integration of multiple machine learning models. The dataset used for training and testing is sourced from the Kaggle competition [How Much Did It Rain II?](https://www.kaggle.com/competitions/how-much-did-it-rain-ii/data).

### Models Used:
1. **LSTM** (Long Short-Term Memory): A type of recurrent neural network (RNN) ideal for sequence prediction, such as time-series data.
2. **Ridge Regression**: A linear regression model with L2 regularization that helps in dealing with multicollinearity and overfitting.
3. **XGBoost**: A powerful gradient boosting model that is effective for prediction tasks, especially when there is structured data.

### Performance:
- The hybrid model approach has resulted in a **10% reduction in error** compared to the standalone models.

## Dataset
- The dataset was downloaded from [Kaggle: How Much Did It Rain II?](https://www.kaggle.com/competitions/how-much-did-it-rain-ii/data).
- The dataset contains radar measurement data used to predict rainfall in the form of millimeters.

## Training Code
- The code for training the models is split into three separate notebooks:
  1. **lstm_rain.ipynb** - LSTM model training.
  2. **ridge_rain.ipynb** - Ridge Regression model training.
  3. **xgb_rain.ipynb** - XGBoost model training.

## Steps to Run the App

Follow the steps below to set up and run the app:

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_folder>
