# Stock Price Prediction with GRU Neural Networks

Advanced deep learning system for financial time series forecasting using Gated Recurrent Units.

## Overview

This project implements a production-grade stock price prediction system achieving less than 2 percent MAPE across multiple stocks. The system demonstrates real-world applicability through comprehensive backtesting, risk management, and professional reporting capabilities.

## Key Features

- **Cross-Stock Validation**: Single model trained on AMZN, tested on IBM and MSFT
- **20 Technical Indicators**: Comprehensive feature engineering (MA, RSI, MACD, Bollinger Bands, Volume metrics)
- **Production-Ready Pipeline**: End-to-end automation from data download to predictions
- **Realistic Backtesting**: 17.55 percent average return with transaction costs included
- **Professional Reporting**: Executive-grade analysis dashboard and metrics

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average MAPE | 1.95% |
| Average R-squared | 0.9238 |
| Backtest Return | +17.55% |
| Signal Precision | 84%+ |
| Win Rate | 53% |

## Tech Stack

- Python 3.8+
- TensorFlow/Keras (GRU architecture)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- yfinance (Yahoo Finance API)

## Model Architecture

