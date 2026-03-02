# =====================================================
# 02_feature_engineering.py
# Purpose:
# Create advanced time-series & demand features
# from base_master_data.csv
# Output: master_cinema_data.csv
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# 1️⃣ LOAD BASE DATA
# =====================================================

master = pd.read_csv("base_master_data.csv")
master["date"] = pd.to_datetime(master["date"])

# Sort properly (CRITICAL for time-series)
master = master.sort_values(
    ["book_theater_id", "date"]
).reset_index(drop=True)

grp = master.groupby("book_theater_id")

# =====================================================
# 2️⃣ TIME FEATURES
# =====================================================

master["month"] = master["date"].dt.month
master["day_of_week"] = master["date"].dt.dayofweek
master["is_weekend"] = master["day_of_week"].isin([5, 6]).astype(int)
master["quarter"] = master["date"].dt.quarter
master["is_friday"] = (master["day_of_week"] == 4).astype(int)

# Cyclic encoding
month = master["month"] - 1
master["month_sin"] = np.sin(2 * np.pi * month / 12)
master["month_cos"] = np.cos(2 * np.pi * month / 12)

dow = master["day_of_week"]
master["dow_sin"] = np.sin(2 * np.pi * dow / 7)
master["dow_cos"] = np.cos(2 * np.pi * dow / 7)

# =====================================================
# 3️⃣ DEMAND FEATURES
# =====================================================

master["total_tickets_sold"] = (
    master["online_ticket_sold"] +
    master["offline_ticket_sold"]
)

master["online_ratio"] = (
    master["online_ticket_sold"] /
    (master["total_tickets_sold"] + 1)
)

master["booking_intensity"] = (
    master["total_tickets_sold"] /
    (master["online_booking_count"] +
     master["offline_booking_count"] + 1)
)

master["online_dominance"] = (
    master["online_ticket_sold"] /
    (master["offline_ticket_sold"] + 1)
)

# =====================================================
# 4️⃣ LAG FEATURES (NO DATA LEAKAGE)
# =====================================================

master["prev_show_audience"] = grp["audience_count"].shift(1)
master["prev_2show"] = grp["audience_count"].shift(2)
master["prev_3show"] = grp["audience_count"].shift(3)

# =====================================================
# 5️⃣ ROLLING FEATURES
# =====================================================

master["rolling_mean_3"] = (
    grp["audience_count"].shift(1).rolling(3).mean()
)

master["rolling_mean_7"] = (
    grp["audience_count"].shift(1).rolling(7).mean()
)

master["rolling_std_7"] = (
    grp["audience_count"].shift(1).rolling(7).std()
)

master["rolling_max_7"] = (
    grp["audience_count"].shift(1).rolling(7).max()
)

# =====================================================
# 6️⃣ TREND & MOMENTUM
# =====================================================

master["trend_3"] = (
    master["prev_show_audience"] -
    master["rolling_mean_3"]
)

master["momentum_3"] = (
    master["prev_show_audience"] -
    master["prev_3show"]
)

master["volatility_7"] = (
    master["rolling_max_7"] -
    master["rolling_mean_7"]
)

master["days_since_last"] = (
    grp["date"].diff().dt.days
)

# =====================================================
# 7️⃣ SAFE FILL
# =====================================================

fill_cols = [
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

for col in fill_cols:
    master[col] = master[col].fillna(0)

# =====================================================
# 8️⃣ ENCODE THEATER TYPE
# =====================================================

if "theater_type" in master.columns:
    master["theater_type"] = master["theater_type"].fillna("other")
    master["theater_type_code"] = (
        master["theater_type"]
        .astype("category")
        .cat.codes
    )

# =====================================================
# 9️⃣ SAVE FINAL DATASET
# =====================================================

master.to_csv("master_cinema_data.csv", index=False)

print("✅ Feature engineering completed!")
print("Final shape:", master.shape)
