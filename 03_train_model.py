# =====================================================
# 03_train_model.py
# Purpose:
# Train forecasting model on engineered features
# Evaluate performance using time-based split
# =====================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =====================================================
# 1️⃣ LOAD FEATURE DATA
# =====================================================

df = pd.read_csv("master_cinema_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically (VERY IMPORTANT for forecasting)
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# 2️⃣ SELECT FEATURES
# =====================================================

feature_cols = [
    # Booking features
    "online_ticket_sold",
    "offline_ticket_sold",
    "online_booking_count",
    "offline_booking_count",
    "total_tickets_sold",
    "online_ratio",
    "booking_intensity",
    "online_dominance",

    # Time features
    "month",
    "day_of_week",
    "is_weekend",
    "quarter",
    "is_friday",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",

    # Lag features
    "prev_show_audience",
    "prev_2show",
    "prev_3show",

    # Rolling features
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_max_7",

    # Trend
    "trend_3",
    "momentum_3",
    "volatility_7",
    "days_since_last"
]

# Keep only available columns (safety check)
feature_cols = [f for f in feature_cols if f in df.columns]

X = df[feature_cols]
y = df["audience_count"]

# =====================================================
# 3️⃣ TIME-BASED TRAIN TEST SPLIT
# =====================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =====================================================
# 4️⃣ TRAIN MODEL
# =====================================================

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# 5️⃣ EVALUATE
# =====================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===================================")
print("Model Performance")
print("===================================")
print(f"MAE: {mae:.2f}")
print(f"R²:  {r2:.4f}")

# =====================================================
# 6️⃣ FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Optional plot
plt.figure(figsize=(8, 6))
plt.barh(importance["feature"].head(10),
         importance["importance"].head(10))
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show()
