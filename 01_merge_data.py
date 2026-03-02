# =====================================================
# 01_merge_data.py
# Purpose:
# Merge raw BookNow + CinePOS datasets into
# ONE clean base dataset (no feature engineering)
# =====================================================

import pandas as pd

# =====================================================
# 1️⃣ LOAD RAW DATA
# =====================================================

booknow_visits = pd.read_csv("booknow_visits.csv")
booknow_booking = pd.read_csv("booknow_booking.csv")
booknow_theaters = pd.read_csv("booknow_theaters.csv")

cinepos_booking = pd.read_csv("cinePOS_booking.csv")
cinepos_theaters = pd.read_csv("cinePOS_theaters.csv")

date_info = pd.read_csv("date_info.csv")
mapping = pd.read_csv("movie_theater_id_relation.csv")

# =====================================================
# 2️⃣ PREPARE VISITS (Audience Count)
# =====================================================

booknow_visits["date"] = pd.to_datetime(booknow_visits["show_date"])
booknow_visits["audience_count"] = pd.to_numeric(
    booknow_visits["audience_count"], errors="coerce"
)

booknow_visits = booknow_visits[
    booknow_visits["audience_count"] >= 0
]

# Collapse to ONE row per theater per date
visits = (
    booknow_visits
    .groupby(["book_theater_id", "date"], as_index=False)
    .agg(audience_count=("audience_count", "sum"))
)

# =====================================================
# 3️⃣ PREPARE DATE INFO
# =====================================================

date_info["date"] = pd.to_datetime(date_info["show_date"])

date_info = (
    date_info
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="first")
    .drop(columns=["show_date"])
)

# =====================================================
# 4️⃣ PREPARE BOOKNOW BOOKINGS (ONLINE)
# =====================================================

booknow_booking["date"] = pd.to_datetime(
    booknow_booking["show_datetime"]
).dt.normalize()

booknow_booking["tickets_booked"] = pd.to_numeric(
    booknow_booking["tickets_booked"], errors="coerce"
)

booknow_booking = booknow_booking[
    booknow_booking["tickets_booked"] > 0
]

booknow_agg = (
    booknow_booking
    .groupby(["book_theater_id", "date"], as_index=False)
    .agg(
        online_ticket_sold=("tickets_booked", "sum"),
        online_booking_count=("tickets_booked", "count")
    )
)

# =====================================================
# 5️⃣ PREPARE CINEPOS BOOKINGS (OFFLINE)
# =====================================================

cinepos_booking["date"] = pd.to_datetime(
    cinepos_booking["show_datetime"]
).dt.normalize()

cinepos_booking["tickets_sold"] = pd.to_numeric(
    cinepos_booking["tickets_sold"], errors="coerce"
)

cinepos_booking = cinepos_booking[
    cinepos_booking["tickets_sold"] > 0
]

cinepos_agg = (
    cinepos_booking
    .groupby(["cine_theater_id", "date"], as_index=False)
    .agg(
        offline_ticket_sold=("tickets_sold", "sum"),
        offline_booking_count=("tickets_sold", "count")
    )
)

# =====================================================
# 6️⃣ MAP CINEPOS TO BOOKNOW IDS
# =====================================================

booknow_agg = booknow_agg.merge(
    mapping, on="book_theater_id", how="left"
)

cinepos_agg = cinepos_agg.merge(
    mapping, on="cine_theater_id", how="left"
)

# =====================================================
# 7️⃣ MERGE ONLINE + OFFLINE BOOKINGS
# =====================================================

master_booking = pd.merge(
    booknow_agg,
    cinepos_agg,
    on=["book_theater_id", "date"],
    how="outer"
)

for col in [
    "online_ticket_sold", "online_booking_count",
    "offline_ticket_sold", "offline_booking_count"
]:
    master_booking[col] = master_booking[col].fillna(0).astype(int)

# =====================================================
# 8️⃣ BUILD BASE MASTER DATASET
# =====================================================

master = visits.merge(date_info, on="date", how="left")

master = master.merge(
    master_booking,
    on=["book_theater_id", "date"],
    how="left"
)

master[[
    "online_ticket_sold", "online_booking_count",
    "offline_ticket_sold", "offline_booking_count"
]] = master[[
    "online_ticket_sold", "online_booking_count",
    "offline_ticket_sold", "offline_booking_count"
]].fillna(0).astype(int)

# =====================================================
# 9️⃣ FINAL SORT & SAVE
# =====================================================

master = master.sort_values(
    ["book_theater_id", "date"]
).reset_index(drop=True)

master.to_csv("base_master_data.csv", index=False)

print("✅ Base master dataset created successfully!")
print("Shape:", master.shape)
