# 🎬 Cinema Audience Forecasting System

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to forecast daily cinema audience turnout using historical booking and visit data.

The system integrates online bookings (BookNow), offline ticket sales (CinePOS), and theater metadata to predict future audience demand using time-series feature engineering and Gradient Boosting.

---

## 🎯 Problem Statement

Cinema operators need accurate demand forecasting to:

- Optimize staffing
- Improve marketing strategies
- Manage ticket pricing
- Plan movie release schedules
- Reduce over/under capacity issues

This project predicts daily `audience_count` per theater using historical behavior and booking trends.

---

## 🏗️ Project Pipeline

Raw Data  
   ↓  
01_merge_data.py  
   ↓  
base_master_data.csv  
   ↓  
02_feature_engineering.py  
   ↓  
master_cinema_data.csv  
   ↓  
03_train_model.py  
   ↓  
Model Evaluation  
   ↓  
04_predict.py  
   ↓  
Future Forecast Output  

---

## 📂 Repository Structure

```
Cinema-Audience-Forecasting/
│
├── 01_merge_data.py          # Raw data merging
├── 02_feature_engineering.py # Time-series & demand features
├── 03_train_model.py         # Model training & evaluation
├── 04_predict.py             # Future forecasting
├── requirements.txt
└── README.md
```

---

## 📊 Feature Engineering

### ⏳ Time Features
- Month
- Day of Week
- Weekend flag
- Quarter
- Friday release indicator
- Cyclic encoding (sin/cos for seasonality)

### 📈 Lag Features (No Data Leakage)
- Previous show audience
- 2-show lag
- 3-show lag

### 📊 Rolling Statistics
- Rolling mean (3 & 7)
- Rolling standard deviation
- Rolling max
- Volatility signals

### 🎟 Booking & Demand Features
- Online ticket ratio
- Booking intensity
- Online dominance
- Total tickets sold
- Days since last show

These features help capture:
- Momentum
- Seasonal patterns
- Demand spikes
- Behavioral shifts

---

## 🤖 Model Used

**Gradient Boosting Regressor**

Why Gradient Boosting?

- Handles non-linear relationships
- Works well with structured tabular data
- Robust to feature scaling
- Strong performance in demand forecasting

---

## 📈 Model Evaluation

- Train/Test Split: Time-based (80/20)
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - R² Score

Time-based validation ensures no future data leakage.

---

## 🔮 Forecasting Logic

The system trains on historical data and generates next-day forecasts per theater.

Output format:

book_theater_id | predicted_next_day_audience

---

## 🧠 Key ML Concepts Demonstrated

- End-to-end ML pipeline design
- Time-series feature engineering
- Lag & rolling statistics
- Data leakage prevention
- Gradient boosting regression
- Feature importance analysis
- Production-style code organization

---

## 🚀 Future Improvements

- Hyperparameter tuning (RandomizedSearchCV)
- Stacking ensemble models
- XGBoost / LightGBM integration
- Holiday/event detection
- Deployment via Streamlit API
- Model serialization with joblib

---

## 🧑‍💻 Author

Avinash Kumar  
B.S. Data Science  
Machine Learning & Forecasting Enthusiast  

---

## 📌 Note

Dataset not included due to size constraints.  
Available upon request.
