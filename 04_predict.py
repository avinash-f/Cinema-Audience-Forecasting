# =====================================================
# 04_predict.py
# Purpose:
# Train final model on full data and
# generate next-day audience forecast
# =====================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# =====================================================
# 1️⃣ LOAD FEATURE DATA
# =====================================================

df = pd.read_csv("master_cinema_data.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["book_theater_id", "date"])

# =====================================================
# 2️⃣ SELECT FEATURES
# =====================================================

feature_cols = [
    "online_ticket_sold",
    "offline_ticket_sold",
    "online_booking_count",
    "offline_booking_count",
    "total_tickets_sold",
    "online_ratio",
    "booking_intensity",
    "online_dominance",
    "month",
    "day_of_week",
    "is_weekend",
    "quarter",
    "is_friday",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "prev_show_audience",
    "prev_2show",
    "prev_3show",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_max_7",
    "trend_3",
    "momentum_3",
    "volatility_7",
    "days_since_last"
]

feature_cols = [f for f in feature_cols if f in df.columns]

X = df[feature_cols]
y = df["audience_count"]

# =====================================================
# 3️⃣ TRAIN FINAL MODEL ON FULL DATA
# =====================================================

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

model.fit(X, y)

# =====================================================
# 4️⃣ GENERATE NEXT-DAY FORECAST (DEMO)
# =====================================================

latest_rows = (
    df
    .sort_values("date")
    .groupby("book_theater_id")
    .tail(1)
)

X_future = latest_rows[feature_cols]

predictions = model.predict(X_future)
predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({
    "book_theater_id": latest_rows["book_theater_id"].values,
    "predicted_next_day_audience": predictions.round().astype(int)
})

submission.to_csv("submission.csv", index=False)

print("✅ Forecast generated successfully!")
print(submission.head())
