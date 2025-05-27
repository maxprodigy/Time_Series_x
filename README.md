# PM2.5 Forecasting with Deep Learning  
A Time Series Challenge with Kaggle Leaderboard Optimization

## Project Overview

This repository contains a complete solution to a time series forecasting challenge focused on predicting PM2.5 air pollution levels in Beijing. The project was designed as a Kaggle-style competition with evaluation based on RMSE (Root Mean Square Error). The objective was to develop a deep learning model capable of generalizing well on unseen data and achieving an RMSE below 4000.

## Problem Statement

Air pollution, particularly PM2.5, is a significant public health and environmental concern. This project aims to predict hourly PM2.5 levels using historical meteorological data, enabling intentional urban planning and public health interventions.

## Objectives

- Forecast PM2.5 values using LSTM/GRU-based deep learning models.
- Implement robust preprocessing and sequence modeling.
- Submit formatted predictions of exactly 13,148 rows to Kaggle.
- Improve model performance through 15+ experiments.
- Analyze results critically and reflect on modeling decisions.

## Methodology

### 1. Data Overview

- **Train Set**: Includes hourly readings of PM2.5, temperature, dew point, pressure, wind speed/direction.
- **Test Set**: Similar to train set, but missing the target `pm2.5`.
- **Submission Format**: Requires 13,148 rows with columns `row ID` and `pm2.5`.

### 2. Preprocessing Steps

- Time-based interpolation and backward fill for missing PM2.5 values.
- MinMaxScaler used for normalizing all input features.
- Sequence creation using 48 time steps.
- Target smoothing applied by averaging the next 3 future values.

### 3. My Model Architecture

A hybrid RNN combining LSTM and GRU layers:

- LSTM(128) with return_sequences=True
- GRU(64)
- Dropout(0.3) for regularization
- Dense(1) for final output
- Custom RMSLE loss function

The model was trained for 25 epochs using a batch size of 64 and the Adam optimizer with a learning rate scheduler.

### 4. Tools and Environment

- Python, TensorFlow/Keras
- Google Colab with GPU support
- Matplotlib for visualization

## Experiments and Results

I made a total of 15 submissions to Kaggle with varying configurations of architecture, dropout, targets, and loss functions. Below is a summary:

| #  | Submission Name     | Layers                        | Batch Size | Dropout | Smoothing | Loss Function | Public RMSE |
|----|---------------------|-------------------------------|------------|---------|-----------|----------------|-------------|
| 1  | subm_fixed77.csv    | LSTM(128) + GRU(64)           | 64         | 0.3     | 3-step    | RMSLE          | **4178.63** |
| 2  | subm_fix3.csv       | LSTM(64) only                 | 64         | 0.2     | No        | MSE            | 5215.53     |
| 3  | subm_fix.csv        | LSTM(128) only                | 64         | 0       | No        | MSE            | 5767.42     |
| 4  | subm_fix4.csv       | LSTM(64) only                 | 32         | 0.2     | No        | MSE            | 5642.31     |
| 5  | subm_fix2.csv       | LSTM(128) only                | 64         | 0.3     | No        | MSE            | 5596.47     |
| 6  | subm_fix11.csv      | GRU(128) only                 | 64         | 0.3     | No        | MSE            | 6162.39     |
| 7  | subm_fix13.csv      | LSTM(128) only                | 64         | 0       | No        | RMSLE          | 6030.81     |
| 8  | subm_fix18.csv      | GRU(128) + LSTM(64)           | 64         | 0.2     | Yes       | MSE            | 5738.70     |
| 9  | subm_fix17.csv      | LSTM(128) only                | 64         | 0.3     | Yes       | RMSLE          | 9479.97     |
| 10 | submission x.csv    | LSTM(64) only                 | 32         | 0       | No        | MSE            | 19546.53    |
| 11 | test_dropout.csv    | LSTM(128) only                | 64         | 0.5     | Yes       | RMSLE          | 6003.77     |
| 12 | test_gruonly.csv    | GRU(128) only                 | 32         | 0.3     | No        | MSE            | 5820.21     |
| 13 | test_lossmse.csv    | LSTM(128) only                | 64         | 0.3     | Yes       | MSE            | 6109.45     |
| 14 | hybrid_3layer.csv   | LSTM(128) + GRU(64) + LSTM(32)| 64         | 0.2     | Yes       | MSE            | 5901.33     |
| 15 | final_aug.csv       | LSTM(128) + GRU(64)           | 64         | 0.3     | Yes       | RMSLE          | 4992.17     |


## Visualizations

Two key plots are included in the `Diagrams/` directory:
- `Time series.png`: Raw PM2.5 concentration over time.
- `Loss plot.png`: Training loss curve showing convergence.

## Challenges and Resolutions

- **Submission Row Mismatch**: Several submissions failed due to incorrect row length. Fixed by adding a padding layer for the initial `seq_length` steps.
- **Overfitting**: Managed through dropout and augmentation.
- **Vanishing Gradients**: Addressed using GRU layers and smoothing targets.

## Key Achievements

- Achieved a leaderboard score of 4178.63.
- Built a reusable training system with proper formatting safeguards.
- Designed and tested 15+ model variations.
- Documented model behavior through results and visualizations.
