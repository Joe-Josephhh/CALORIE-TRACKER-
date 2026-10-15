#!/usr/bin/env python3
"""
scripts/seed_cohorts.py
=======================
Seeds realistic evaluation cohorts into calorie_tracker.db for user groups
that currently have insufficient or zero evaluable users in the existing database.

Cohorts created (without modifying any existing user records):
- Group A (< 20 days):
    * alex_a16: 16 days (eligible: >= 14 days, tests the 14-19 day window)
- Group C (30-40 days):
    * priya_c32: 32 days (eligible: 30-40 days cohort)
    * rahul_c38: 38 days (eligible: 30-40 days cohort)
- Group D (> 40 days):
    * maya_d55: 55 days (eligible: multi-month tracking cohort)

All data uses authentic nutritional items, physiological Mifflin-St Jeor BMR,
and MET activity formulas consistent with seed_test_data.py.
"""

import sys
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from werkzeug.security import generate_password_hash

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import get_connection, init_database
from seed_test_data import generate_daily_meals, generate_daily_exercise


COHORT_SPECS = [
    {
        "username": "alex_a16",
        "email": "alex_a16@calorie-tracker.local",
        "days": 16,
        "age": 24,
        "gender": "Male",
        "height_cm": 175.0,
        "start_weight": 75.0,
        "goal": "Weight Loss",
        "activity_level": "Moderately Active",
        "seed_offset": 3100
    },
    {
        "username": "priya_c32",
        "email": "priya_c32@calorie-tracker.local",
        "days": 32,
        "age": 26,
        "gender": "Female",
        "height_cm": 162.0,
        "start_weight": 64.5,
        "goal": "Weight Loss",
        "activity_level": "Lightly Active",
        "seed_offset": 4100
    },
    {
        "username": "rahul_c38",
        "email": "rahul_c38@calorie-tracker.local",
        "days": 38,
        "age": 28,
        "gender": "Male",
        "height_cm": 178.0,
        "start_weight": 82.5,
        "goal": "Weight Loss",
        "activity_level": "Moderately Active",
        "seed_offset": 5100
    },
    {
        "username": "maya_d55",
        "email": "maya_d55@calorie-tracker.local",
        "days": 55,
        "age": 23,
        "gender": "Female",
        "height_cm": 168.0,
        "start_weight": 68.2,
        "goal": "Weight Loss",
        "activity_level": "Moderately Active",
        "seed_offset": 6100
    }
]


def seed_single_cohort_user(conn, spec, end_date_str="2026-10-05"):
    cursor = conn.cursor()
    username = spec["username"]
    email = spec["email"]
    days = spec["days"]
    password_hash = generate_password_hash("password123")

    # 1. Create or get user
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    row = cursor.fetchone()
    if row:
        user_id = row[0]
    else:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        """, (username, email, password_hash))
        user_id = cursor.lastrowid

    # 2. Ensure profile
    cursor.execute("SELECT id FROM user_profiles WHERE user_id = ?", (user_id,))
    p_row = cursor.fetchone()
    if p_row:
        cursor.execute("""
            UPDATE user_profiles
            SET age = ?, gender = ?, height_cm = ?, goal = ?, activity_level = ?
            WHERE user_id = ?
        """, (spec["age"], spec["gender"], spec["height_cm"], spec["goal"], spec["activity_level"], user_id))
    else:
        cursor.execute("""
            INSERT INTO user_profiles (user_id, age, gender, height_cm, goal, activity_level)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, spec["age"], spec["gender"], spec["height_cm"], spec["goal"], spec["activity_level"]))

    # 3. Clear only this user's logs
    cursor.execute("DELETE FROM calorie_in_records WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM calorie_out_records WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM weight_records WHERE user_id = ?", (user_id,))

    # 4. Generate data
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=days - 1)
    dates_list = [start_date + timedelta(days=i) for i in range(days)]

    true_weight = float(spec["start_weight"])
    water_offset = 0.0
    rng = random.Random(spec["seed_offset"])
    np_rng = np.random.RandomState(spec["seed_offset"])

    all_meals = []
    all_exercises = []
    all_weights = []

    for day_idx, dt in enumerate(dates_list, start=1):
        date_str = dt.strftime("%Y-%m-%d")
        dow = dt.weekday()

        meals = generate_daily_meals(date_str, seed_val=spec["seed_offset"] + day_idx)
        daily_c_in = sum(m["calories"] for m in meals)
        all_meals.extend(meals)

        exercise = generate_daily_exercise(date_str, dow, true_weight, seed_val=spec["seed_offset"] + 1000 + day_idx)
        daily_c_out = exercise["calories_burned"] if exercise else 0.0
        if exercise:
            all_exercises.append(exercise)

        # Mifflin-St Jeor BMR
        base_bmr = (10.0 * true_weight) + (6.25 * spec["height_cm"]) - (5.0 * spec["age"])
        if spec["gender"].lower() == "male":
            bmr = base_bmr + 5.0
        else:
            bmr = base_bmr - 161.0
        
        mult = 1.375 if "light" in spec["activity_level"].lower() else 1.55
        baseline_burn = bmr * mult
        total_burn = baseline_burn + (daily_c_out * 0.6)  # avoid full double-count
        balance = daily_c_in - total_burn
        tissue_delta = balance / 7700.0
        true_weight += tissue_delta

        noise = float(np_rng.normal(0, 0.06))
        carb_effect = (daily_c_in - 2000.0) * 0.0001
        water_offset = 0.35 * water_offset + noise + carb_effect
        water_offset = max(-0.20, min(0.20, water_offset))

        scale_weight = round(true_weight + water_offset, 2)
        all_weights.append({
            "weight_kg": scale_weight,
            "recorded_date": date_str
        })

    # Batch insert
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

    cursor.executemany("""
        INSERT INTO calorie_out_records (
            user_id, activity, met, weight_kg,
            duration_minutes, calories_burned, recorded_date
        ) VALUES (
            :user_id, :activity, :met, :weight_kg,
            :duration_minutes, :calories_burned, :recorded_date
        )
    """, [{**e, "user_id": user_id} for e in all_exercises])

    cursor.executemany("""
        INSERT INTO weight_records (
            user_id, weight_kg, recorded_date
        ) VALUES (
            :user_id, :weight_kg, :recorded_date
        )
    """, [{**w, "user_id": user_id} for w in all_weights])

    conn.commit()
    return user_id


def seed_all_cohorts(end_date_str="2026-10-05"):
    init_database()
    conn = get_connection()
    results = []
    print("=" * 60)
    print("SEEDING EVALUATION COHORTS (NON-DESTRUCTIVE)")
    print("=" * 60)
    for spec in COHORT_SPECS:
        uid = seed_single_cohort_user(conn, spec, end_date_str)
        results.append((spec["username"], uid, spec["days"]))
        print(f"  * Seeded '{spec['username']}' (User ID: {uid}) -> {spec['days']} days ({spec['days'] - 7} sequences)")
    conn.close()
    print("=" * 60)
    return results


if __name__ == "__main__":
    seed_all_cohorts()
