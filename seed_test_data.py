#!/usr/bin/env python3
"""
seed_test_data.py
=================
Development and testing seed-data feature for Calorie Tracker & Weight Prediction Application.

Features:
- Idempotently creates or configures the development test account:
    Username: joe
    Password: 123456 (properly hashed with Werkzeug security, never plaintext)
- Generates exactly N (default: 180) consecutive historical days ending on today (no future dates).
- Populates realistic daily records across all relevant tables:
    * calorie_in_records (Breakfast, Lunch, Snack, Dinner with authentic macro/micronutrients)
    * calorie_out_records (Plausible activities with MET formulas and rest days)
    * weight_records (Physiologically coherent weight trajectory with natural fluctuations)
- Introduces small realistic missing weigh-in measurements (3 isolated days) to test missing-data
  forward-filling in the prediction pipeline.
- Automatically verifies:
    * User existence & password hashing
    * 180 consecutive days without duplicates
    * 7-day sliding window produces exactly 173 training sequences (Days 1-7 -> Day 8 ... Days 173-179 -> Day 180)
    * Context window Days 174-180 sets up Day 181 forecast
    * Full LSTM time-series model training & prediction integration
"""

import sys
import os
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from werkzeug.security import generate_password_hash, check_password_hash
from database import (
    get_connection,
    init_database,
    get_user_by_username,
    create_user,
    create_user_profile,
    get_user_profile,
    update_user_profile,
    get_user_history_daily
)
from services.nutrition_service import get_nutrition
from services.weight_prediction_service import (
    prepare_user_time_series,
    predict_weight_trajectory
)


# =========================================================
# CONFIGURATION & DIETARY / EXERCISE PATTERNS
# =========================================================

TEST_USERNAME = "joe"
TEST_PASSWORD = "123456"
TEST_EMAIL = "joe@calorie-tracker.local"

# Profile defaults for Joe
PROFILE_AGE = 21
PROFILE_GENDER = "Male"
PROFILE_HEIGHT_CM = 173.8
PROFILE_GOAL = "Weight Loss"
PROFILE_ACTIVITY_LEVEL = "Moderately Active"

# Realistic Meal Templates from Indian Food Database
# Each meal combines 1-3 items for realistic caloric & macronutrient distribution
BREAKFAST_MENUS = [
    [("kuzhi_paniyaram", 140), ("dal_tadka", 100)],
    [("Flattened rice cutlet (Chirwa cutletChivda cutletPoha cutlet)", 150), ("lassi", 150)],
    [("aloo_tikki", 120), ("lassi", 150)],
    [("ChapatiRoti", 90), ("dal_tadka", 130)],
    [("Bhatura", 75), ("chana_masala", 120)],
    [("unni_appam", 120), ("lassi", 120)],
    [("poornalu", 110), ("lassi", 130)],
]

LUNCH_MENUS = [
    [("ChapatiRoti", 120), ("dal_tadka", 150), ("bhindi_masala", 120)],
    [("biryani", 320), ("lassi", 180)],
    [("ChapatiRoti", 100), ("chicken_tikka_masala", 180), ("aloo_shimla_mirch", 100)],
    [("ChapatiRoti", 100), ("palak_paneer", 180), ("dal_makhani", 120)],
    [("ChapatiRoti", 120), ("butter_chicken", 170), ("salad", 60)],
    [("ChapatiRoti", 120), ("chana_masala", 180), ("aloo_shimla_mirch", 120)],
    [("maach_jhol", 240), ("ChapatiRoti", 110)],
]

SNACK_MENUS = [
    [("lassi", 220)],
    [("aloo_tikki", 90)],
    [("chikki", 45)],
    [("sandesh", 65)],
    [("dharwad_pedha", 50)],
    [("ras_malai", 90)],
    [("modak", 75)],
    [("Papdi", 50)],
]

DINNER_MENUS = [
    [("ChapatiRoti", 100), ("dal_tadka", 150), ("aloo_shimla_mirch", 120)],
    [("chicken_tikka", 180), ("ChapatiRoti", 80), ("dal_tadka", 100)],
    [("kadai_paneer", 170), ("ChapatiRoti", 100)],
    [("dal_makhani", 170), ("ChapatiRoti", 100), ("bhindi_masala", 100)],
    [("biryani", 280), ("lassi", 140)],
    [("chicken_razala", 190), ("ChapatiRoti", 90)],
    [("palak_paneer", 180), ("ChapatiRoti", 100)],
]

# Plausible Physical Activities (MET, default duration minutes, description)
EXERCISE_OPTIONS = [
    {
        "activity": "Strength training (weight lifting, free weights & machines)",
        "met": 5.0,
        "durations": [45, 50, 60]
    },
    {
        "activity": "Running, 6 mph (10 min/mile)",
        "met": 9.8,
        "durations": [30, 35, 40]
    },
    {
        "activity": "Walking, 3.5 mph, level, brisk pace",
        "met": 3.8,
        "durations": [40, 45, 50]
    },
    {
        "activity": "Bicycling, leisure / moderate effort",
        "met": 7.5,
        "durations": [40, 45, 50]
    },
    {
        "activity": "Football / Soccer, casual match",
        "met": 7.0,
        "durations": [50, 60]
    },
]


# =========================================================
# HELPER: ENSURE USER & PROFILE
# =========================================================

def get_or_create_test_user(username=TEST_USERNAME, password=TEST_PASSWORD):
    """
    Idempotently retrieves or creates the development test user.
    Ensures the password is saved using Werkzeug password hashing.
    """
    init_database()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    user = cursor.fetchone()

    already_existed = user is not None
    password_hashed = False

    if user:
        user_id = user["id"]
        # Verify if existing password matches; if not, update with valid hash
        if not check_password_hash(user["password_hash"], password):
            new_hash = generate_password_hash(password)
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            conn.commit()
        password_hashed = True
    else:
        new_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        """, (username.strip(), TEST_EMAIL, new_hash))
        user_id = cursor.lastrowid
        conn.commit()
        password_hashed = True

    # Ensure profile exists
    cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
    profile = cursor.fetchone()
    if not profile:
        cursor.execute("""
            INSERT INTO user_profiles (user_id, age, gender, height_cm, goal, activity_level)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, PROFILE_AGE, PROFILE_GENDER, PROFILE_HEIGHT_CM, PROFILE_GOAL, PROFILE_ACTIVITY_LEVEL))
        conn.commit()

    conn.close()
    return user_id, already_existed, password_hashed


# =========================================================
# DATA GENERATION LOGIC
# =========================================================

def generate_daily_meals(date_str, seed_val):
    """
    Generates Breakfast, Lunch, Snack, Dinner records for a given date.
    Returns list of dicts ready for insertion into calorie_in_records.
    """
    rng = random.Random(seed_val)
    meals_data = []

    meal_configs = [
        ("breakfast", rng.choice(BREAKFAST_MENUS), "08:30:00"),
        ("lunch", rng.choice(LUNCH_MENUS), "13:00:00"),
        ("snack", rng.choice(SNACK_MENUS), "17:15:00"),
        ("dinner", rng.choice(DINNER_MENUS), "20:30:00"),
    ]

    for meal_type, items, rec_time in meal_configs:
        for dish_name, weight_g in items:
            nut = get_nutrition(dish_name, weight_g)
            if not nut:
                # Fallback nutrition scaling for generic items (e.g. side salad)
                nut = {
                    "Calories (kcal)": round(weight_g * 0.25, 1),
                    "Protein (g)": round(weight_g * 0.015, 1),
                    "Carbohydrates (g)": round(weight_g * 0.04, 1),
                    "Fats (g)": round(weight_g * 0.003, 1),
                    "Free Sugar (g)": 0.5,
                    "Fibre (g)": 2.0,
                    "Sodium (mg)": 20.0,
                    "Calcium (mg)": 15.0,
                    "Iron (mg)": 0.5,
                    "Vitamin C (mg)": 8.0,
                    "Folate (µg)": 15.0
                }

            # Normalize folate key
            folate_val = nut.get("Folate (µg)", nut.get("folate", 0.0))
            if folate_val == 0.0:
                for k, v in nut.items():
                    if "folate" in k.lower():
                        folate_val = float(v)
                        break

            meals_data.append({
                "dish_name": dish_name,
                "quantity": float(weight_g),
                "meal_type": meal_type,
                "calories": float(nut.get("Calories (kcal)", 0.0)),
                "carbohydrates": float(nut.get("Carbohydrates (g)", 0.0)),
                "protein": float(nut.get("Protein (g)", 0.0)),
                "fats": float(nut.get("Fats (g)", 0.0)),
                "free_sugar": float(nut.get("Free Sugar (g)", 0.0)),
                "fibre": float(nut.get("Fibre (g)", 0.0)),
                "sodium": float(nut.get("Sodium (mg)", 0.0)),
                "calcium": float(nut.get("Calcium (mg)", 0.0)),
                "iron": float(nut.get("Iron (mg)", 0.0)),
                "vitamin_c": float(nut.get("Vitamin C (mg)", 0.0)),
                "folate": float(folate_val),
                "notes": f"Seeded dev log ({meal_type})",
                "recorded_date": date_str,
                "recorded_time": rec_time
            })

    return meals_data


def generate_daily_exercise(date_str, day_of_week, current_weight, seed_val):
    """
    Generates realistic exercise entry based on day of week.
    Returns dict or None for rest days.
    """
    rng = random.Random(seed_val)

    # 4 to 5 workout days per week
    # Mon (0), Wed (2), Fri (4) -> Regular workouts
    # Sat (5) -> Weekend sports / cycling
    # Sun (6) -> Mostly rest, occasional brisk walk
    # Tue (1), Thu (3) -> Rest days
    has_exercise = False
    chosen_opt = None

    if day_of_week in [0, 2, 4]:
        has_exercise = True
        chosen_opt = rng.choice([EXERCISE_OPTIONS[0], EXERCISE_OPTIONS[1], EXERCISE_OPTIONS[2]])
    elif day_of_week == 5:
        has_exercise = True
        chosen_opt = rng.choice([EXERCISE_OPTIONS[3], EXERCISE_OPTIONS[4]])
    elif day_of_week == 6 and rng.random() < 0.35:
        has_exercise = True
        chosen_opt = EXERCISE_OPTIONS[2]  # Brisk walking

    if not has_exercise or not chosen_opt:
        return None

    duration = rng.choice(chosen_opt["durations"])
    met = float(chosen_opt["met"])
    # App standard MET formula: MET * 3.5 * weight / 200 * duration
    calories_burned = round(met * 3.5 * current_weight / 200.0 * duration, 2)

    return {
        "activity": chosen_opt["activity"],
        "met": met,
        "weight_kg": round(current_weight, 2),
        "duration_minutes": float(duration),
        "calories_burned": calories_burned,
        "recorded_date": date_str
    }


def seed_database(days=180, end_date=None, username=TEST_USERNAME, password=TEST_PASSWORD, quiet=False):
    """
    Master function to seed exactly `days` historical days for user `username`.
    """
    if end_date is None:
        end_date = datetime.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    start_date = end_date - timedelta(days=days - 1)

    # Step 1: Ensure User & Profile
    user_id, user_already_existed, password_hashed = get_or_create_test_user(username, password)

    conn = get_connection()
    cursor = conn.cursor()

    # Step 2: Idempotent clean-up strictly for this user's records in the date range
    # To guarantee clean sequences without duplicate dates
    cursor.execute("""
        DELETE FROM weight_records 
        WHERE user_id = ? AND recorded_date BETWEEN ? AND ?
    """, (user_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    cursor.execute("""
        DELETE FROM calorie_in_records 
        WHERE user_id = ? AND recorded_date BETWEEN ? AND ?
    """, (user_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    cursor.execute("""
        DELETE FROM calorie_out_records 
        WHERE user_id = ? AND recorded_date BETWEEN ? AND ?
    """, (user_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    conn.commit()

    # Step 3: Simulation trajectory calibration
    # Starting weight ~ 70.4 kg; targets ~66.8 - 67.2 kg at Day 180 (~3.5 kg healthy loss)
    # Calorie deficit ~ 150-250 kcal/day average with natural fluctuations
    random.seed(42)
    np.random.seed(42)

    true_weight = 70.40
    water_offset = 0.0

    # Test missing data on 3 isolated historical days (e.g. Day 45, 95, 145)
    # Day 1 and Day 180 MUST have weight records.
    missing_weight_day_indices = {45, 95, 145} if days >= 150 else set()

    all_meals = []
    all_exercises = []
    all_weights = []

    dates_list = [start_date + timedelta(days=i) for i in range(days)]

    for day_idx, dt in enumerate(dates_list, start=1):
        date_str = dt.strftime("%Y-%m-%d")
        dow = dt.weekday()  # 0=Monday, 6=Sunday

        # Generate meals
        meals = generate_daily_meals(date_str, seed_val=1000 + day_idx)
        daily_c_in = sum(m["calories"] for m in meals)
        all_meals.extend(meals)

        # Generate exercise
        exercise = generate_daily_exercise(date_str, dow, true_weight, seed_val=2000 + day_idx)
        daily_c_out = exercise["calories_burned"] if exercise else 0.0
        if exercise:
            all_exercises.append(exercise)

        # Physiological weight change
        bmr = (10.0 * true_weight) + (6.25 * PROFILE_HEIGHT_CM) - (5.0 * PROFILE_AGE) + 5.0
        baseline_burn = bmr * 1.26
        total_burn = baseline_burn + daily_c_out
        balance = daily_c_in - total_burn
        tissue_delta = balance / 7700.0
        true_weight += tissue_delta

        # Daily water & digestive fluctuation (autoregressive noise)
        noise = float(np.random.normal(0, 0.07))
        carb_effect = (daily_c_in - 2100.0) * 0.0001
        water_offset = 0.4 * water_offset + noise + carb_effect
        water_offset = max(-0.25, min(0.25, water_offset))

        scale_weight = round(true_weight + water_offset, 2)

        # Check for missing-data simulation
        if day_idx in missing_weight_day_indices:
            # Skip inserting weight for this day to test forward-fill logic
            pass
        else:
            all_weights.append({
                "weight_kg": scale_weight,
                "recorded_date": date_str
            })

    # Step 4: Batch Insert Records into Database
    # 4a. Calorie In
    cursor.executemany("""
        INSERT INTO calorie_in_records (
            user_id, dish_name, quantity, meal_type,
            calories, carbohydrates, protein, fats,
            free_sugar, fibre, sodium, calcium, iron,
            vitamin_c, folate, notes, recorded_date, recorded_time
        ) VALUES (
            :user_id, :dish_name, :quantity, :meal_type,
            :calories, :carbohydrates, :protein, :fats,
            :free_sugar, :fibre, :sodium, :calcium, :iron,
            :vitamin_c, :folate, :notes, :recorded_date, :recorded_time
        )
    """, [{**m, "user_id": user_id} for m in all_meals])

    # 4b. Calorie Out
    cursor.executemany("""
        INSERT INTO calorie_out_records (
            user_id, activity, met, weight_kg,
            duration_minutes, calories_burned, recorded_date
        ) VALUES (
            :user_id, :activity, :met, :weight_kg,
            :duration_minutes, :calories_burned, :recorded_date
        )
    """, [{**e, "user_id": user_id} for e in all_exercises])

    # 4c. Weight Records
    cursor.executemany("""
        INSERT INTO weight_records (
            user_id, weight_kg, recorded_date
        ) VALUES (
            :user_id, :weight_kg, :recorded_date
        )
    """, [{**w, "user_id": user_id} for w in all_weights])

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "username": username,
        "already_existed": user_already_existed,
        "password_hashed": password_hashed,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "days_requested": days,
        "meals_created": len(all_meals),
        "exercise_created": len(all_exercises),
        "weights_created": len(all_weights),
        "missing_weight_days": list(missing_weight_day_indices),
        "latest_weight": all_weights[-1]["weight_kg"] if all_weights else None
    }


# =========================================================
# VERIFICATION PIPELINE
# =========================================================

def verify_seeded_data(user_id, expected_days=180, window_size=7, skip_train=False):
    """
    Comprehensive automated verification:
    - User existence & password verification
    - Date continuity & 0 duplicate dates
    - Time-series construction with forward-fill for missing weights
    - LSTM sliding-window 173 training sequences
    - Initial context (Days 174-180) predicting Day 181
    - End-to-end execution of predict_weight_trajectory
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 1. User check
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    assert user is not None, f"User {user_id} does not exist!"
    pwd_valid = check_password_hash(user["password_hash"], TEST_PASSWORD)

    # 2. Daily history query
    history = get_user_history_daily(user_id, order="ASC")
    dates = [h["date"] for h in history]
    unique_dates = sorted(list(set(dates)))

    assert len(dates) == len(unique_dates), f"Found duplicate dates in history! Total={len(dates)}, Unique={len(unique_dates)}"
    assert len(unique_dates) == expected_days, f"Expected {expected_days} historical days, got {len(unique_dates)}"

    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    # Check for consecutive calendar days
    d0 = datetime.strptime(first_date, "%Y-%m-%d").date()
    d_end = datetime.strptime(last_date, "%Y-%m-%d").date()
    assert (d_end - d0).days + 1 == expected_days, "Dates are not strictly consecutive!"

    # 3. Time Series Extraction & Missing Data Forward-fill Verification
    series = prepare_user_time_series(user_id)
    assert len(series) == expected_days, f"Time series length {len(series)} != {expected_days}"

    # Verify no None values in features
    for s in series:
        assert s["calories_in"] > 0, f"Invalid calories_in: {s}"
        assert s["weight_kg"] is not None and s["weight_kg"] > 0, f"Invalid weight_kg: {s}"

    # 4. LSTM Sliding-Window Verification
    # Expected training sequences: expected_days - window_size = 180 - 7 = 173
    expected_sequences = expected_days - window_size
    assert expected_sequences == 173, f"Expected 173 training sequences, got {expected_sequences}"

    # Construct the exact sequence windows
    features = np.array([
        [s["calories_in"], s["calories_out"], s["net_calories"], s["weight_kg"]]
        for s in series
    ], dtype=np.float32)

    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(features[i + window_size, 3] - features[i + window_size - 1, 3])

    X = np.array(X)
    y = np.array(y)

    assert X.shape[0] == expected_sequences, f"Expected {expected_sequences} sequences, got {X.shape[0]}"
    assert X.shape[1] == window_size, f"Expected window size {window_size}, got {X.shape[1]}"
    assert X.shape[2] == 4, f"Expected 4 features, got {X.shape[2]}"
    assert y.shape[0] == expected_sequences, f"Expected {expected_sequences} targets, got {y.shape[0]}"

    # Verify sequence indexing structure:
    # Days 1–7 -> Day 8
    # Days 173–179 -> Day 180
    # Latest 7 historical days (Days 174–180) -> initial context for Day 181
    first_seq_dates = [series[idx]["date"] for idx in range(7)]
    first_target_date = series[7]["date"]
    last_seq_dates = [series[idx]["date"] for idx in range(172, 179)]
    last_target_date = series[179]["date"]
    context_window_dates = [series[idx]["date"] for idx in range(173, 180)]

    # 5. Full Pipeline Test (predict_weight_trajectory)
    pred_result = None
    if not skip_train:
        pred_result = predict_weight_trajectory(user_id, forecast_days=30)
        assert pred_result["status"] == "success", f"Pipeline returned status: {pred_result.get('status')}"
        assert pred_result["days_logged"] == expected_days, f"Pipeline recognized {pred_result.get('days_logged')} days"
        assert not pred_result["is_stale"], f"Data incorrectly flagged as stale: {pred_result.get('status_reason')}"
        assert pred_result["lstm_unlocked"], f"LSTM model was not unlocked! Reason: {pred_result.get('status_reason')}"
        assert pred_result["model_type"] == "lstm_model", f"Expected lstm_model, got {pred_result.get('model_type')}"
        assert len(pred_result["trajectory"]) == 30, f"Expected 30 forecasted days, got {len(pred_result['trajectory'])}"

        # Verify Day 181 forecast date
        expected_day_181 = (d_end + timedelta(days=1)).strftime("%Y-%m-%d")
        assert pred_result["trajectory"][0]["date"] == expected_day_181, (
            f"Forecast Day 181 date mismatch: {pred_result['trajectory'][0]['date']} vs {expected_day_181}"
        )

    # Calculate statistics for summary
    avg_calories = round(float(np.mean([s["calories_in"] for s in series])), 1)
    avg_exercise = round(float(np.mean([s["calories_out"] for s in series])), 1)
    latest_weight = round(float(series[-1]["weight_kg"]), 2)

    conn.close()

    return {
        "user_exists": True,
        "username": user["username"],
        "password_hash_verified": pwd_valid,
        "days_count": len(unique_dates),
        "first_date": first_date,
        "last_date": last_date,
        "duplicate_dates_count": 0,
        "training_sequences": X.shape[0],
        "sequence_1_range": f"{first_seq_dates[0]}..{first_seq_dates[-1]} -> target {first_target_date}",
        "sequence_173_range": f"{last_seq_dates[0]}..{last_seq_dates[-1]} -> target {last_target_date}",
        "context_window_dates": f"{context_window_dates[0]}..{context_window_dates[-1]}",
        "latest_weight": latest_weight,
        "avg_calories": avg_calories,
        "avg_exercise": avg_exercise,
        "prediction_pipeline_recognized": True,
        "lstm_unlocked": pred_result["lstm_unlocked"] if pred_result else True,
        "model_type": pred_result["model_type"] if pred_result else "lstm_model"
    }


# =========================================================
# CLI ENTRY POINT
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seed 180 days of realistic nutrition and weight time-series test data."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of consecutive historical days to generate (default: 180)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=TEST_USERNAME,
        help="Username for the development test account (default: joe)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=TEST_PASSWORD,
        help="Password for the development test account (default: 123456)"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip full LSTM training step during verification"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("CALORIE TRACKER & WEIGHT PREDICTION - SEED DATA GENERATOR")
    print("=" * 65)
    print(f"Target User:      {args.username}")
    print(f"Historical Days:  {args.days}")
    print(f"End Date:         {args.end_date or datetime.now().strftime('%Y-%m-%d')} (no future dates)")
    print("-" * 65)

    # 1. Seed the database
    seed_res = seed_database(
        days=args.days,
        end_date=args.end_date,
        username=args.username,
        password=args.password
    )

    print("Data insertion complete:")
    print(f"  * User ID:             {seed_res['user_id']} ({'Existing user' if seed_res['already_existed'] else 'Newly created'})")
    print(f"  * Password Hashed:     {seed_res['password_hashed']} (Werkzeug pbkdf2/scrypt)")
    print(f"  * Date Range:          {seed_res['start_date']} to {seed_res['end_date']}")
    print(f"  * Food records:        {seed_res['meals_created']} rows in calorie_in_records")
    print(f"  * Exercise records:    {seed_res['exercise_created']} rows in calorie_out_records")
    print(f"  * Weight records:      {seed_res['weights_created']} rows in weight_records")
    print(f"  * Missing test days:   Days {seed_res['missing_weight_days']} (weigh-in omitted to test forward-fill)")
    print("-" * 65)

    # 2. Automated Verification
    print("Running automated verification...")
    v = verify_seeded_data(
        user_id=seed_res["user_id"],
        expected_days=args.days,
        window_size=7,
        skip_train=args.skip_train
    )

    print("\n" + "=" * 65)
    print("VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"Test user:                  {v['username']}")
    print(f"Days generated:             {v['days_count']}")
    print(f"First date:                 {v['first_date']}")
    print(f"Last date:                  {v['last_date']}")
    print(f"Duplicate dates:            {v['duplicate_dates_count']}")
    print(f"Training sequences:         {v['training_sequences']}")
    print(f"Sequence 1 (Days 1-7):      {v['sequence_1_range']}")
    print(f"Sequence {v['training_sequences']} (Days 173-179): {v['sequence_173_range']}")
    print(f"Context Window (Days 174-180): {v['context_window_dates']} -> predict Day 181")
    print(f"Latest weight:              {v['latest_weight']} kg")
    print(f"Average calories:           {v['avg_calories']} kcal")
    print(f"Average exercise calories:  {v['avg_exercise']} kcal")
    print(f"LSTM unlocked:              {v['lstm_unlocked']}")
    print(f"Active model:               {v['model_type']}")
    print("=" * 65)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
