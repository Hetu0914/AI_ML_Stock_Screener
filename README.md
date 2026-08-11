# AI/ML Stock Market Screening and Analysis System

An AI/ML-based stock market screening and analysis application that combines SMMA crossover analysis with a Random Forest machine learning model to evaluate potential stock signals.

## Project Overview

This project detects SMMA (20) and SMMA (120) crossover signals and uses historical market data to train a Random Forest classifier.

The trained model evaluates current market features and generates a probability-based decision:

- BUY
- WATCH
- NO TRADE

The system also includes historical backtesting and model performance analysis.

## Key Features

- NSE stock market screening
- SMMA 20 and SMMA 120 indicators
- Automatic SMMA crossover detection
- Historical crossover analysis
- Random Forest machine learning model
- Profitability prediction
- Probability-based trading decisions
- High-confidence signal detection
- Historical backtesting
- Feature importance analysis
- Interactive Streamlit dashboard
- CSV result generation

## System Workflow

```text
Market Data
     ↓
Technical Indicators
     ↓
SMMA 20 / SMMA 120
     ↓
Crossover Detection
     ↓
Feature Engineering
     ↓
Random Forest Model
     ↓
Probability Prediction
     ↓
BUY / WATCH / NO TRADE
     ↓
Backtesting & Performance Analysis

## 🚀 Live Dashboard

[Open AI/ML Stock Screener Dashboard](https://stock-screener-hetvi.streamlit.app/)
