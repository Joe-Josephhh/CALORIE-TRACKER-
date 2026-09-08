import numpy as np
from datetime import datetime, timedelta
from database import (
    get_user_profile,
    get_latest_weight,
    get_starting_weight,
    get_weight_history,
    get_user_history_daily
)

# ---------------------------------------------------------
# METABOLIC BASICS (Mifflin-St Jeor & TDEE)
# ---------------------------------------------------------

def calculate_bmr(weight_kg, height_cm, age, gender="Other"):
    """
    Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation.
    Men: 10*weight + 6.25*height - 5*age + 5
    Women: 10*weight + 6.25*height - 5*age - 161
    Other/Neutral: average
    """
    base = (10.0 * float(weight_kg)) + (6.25 * float(height_cm)) - (5.0 * float(age))
    g = str(gender).strip().lower()
    if g == "male":
        return round(base + 5.0, 1)
    elif g == "female":
        return round(base - 161.0, 1)
    else:
        return round(base - 78.0, 1)


def calculate_tdee(bmr, activity_level="Moderately Active"):
    """
    Calculate Total Daily Energy Expenditure (TDEE) without exercise logs.
    """
    multipliers = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "super active": 1.9
    }
    key = str(activity_level).strip().lower()
    factor = multipliers.get(key, 1.55)
    return round(bmr * factor, 1)


# ---------------------------------------------------------
# TIME SERIES PREPARATION FOR LSTM
# ---------------------------------------------------------

def prepare_user_time_series(user_id):
    """
    Extract chronological daily series of [calories_in, calories_out, net_calories, weight_kg].
    Fills in missing weight data by forward-filling or baseline weight.
    """
    history = get_user_history_daily(user_id, order="ASC")
    if not history:
        return []

    latest_w_obj = get_latest_weight(user_id)
    fallback_weight = float(latest_w_obj["weight_kg"]) if latest_w_obj else 70.0

    series = []
    last_known_weight = fallback_weight

    for entry in history:
        if entry["weight_kg"] is not None:
            last_known_weight = float(entry["weight_kg"])

        series.append({
            "date": entry["date"],
            "calories_in": float(entry["calories_in"]),
            "calories_out": float(entry["calories_out"]),
            "net_calories": float(entry["net_calories"]),
            "weight_kg": last_known_weight
        })

    return series


def build_lstm_model(input_shape=(7, 4)):
    """
    Builds a lightweight TensorFlow Keras LSTM model for sequential weight prediction.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, activation="tanh", return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_and_predict(series, forecast_days=30, window_size=7):
    """
    Trains an LSTM on the user's sequential records if sufficient data exists (>= 14 days),
    and predicts future weight trajectory iteratively.
    """
    if len(series) < window_size * 2:
        return None

    # Prepare features: [calories_in, calories_out, net_calories, weight_kg]
    features = np.array([
        [s["calories_in"], s["calories_out"], s["net_calories"], s["weight_kg"]]
        for s in series
    ], dtype=np.float32)

    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        # Target: next day's weight
        y.append(features[i + window_size, 3])

    X = np.array(X)
    y = np.array(y)

    model = build_lstm_model(input_shape=(window_size, 4))
    # Quick fine-tuning epochs for responsive web inference
    model.fit(X, y, epochs=15, batch_size=4, verbose=0)

    # Iterative rolling prediction
    current_window = features[-window_size:].copy()
    avg_in = float(np.mean([s["calories_in"] for s in series[-window_size:]]))
    avg_out = float(np.mean([s["calories_out"] for s in series[-window_size:]]))
    avg_net = avg_in - avg_out

    predicted_weights = []
    for _ in range(forecast_days):
        input_data = current_window.reshape(1, window_size, 4)
        next_weight = float(model.predict(input_data, verbose=0)[0, 0])
        # Cap daily rate of change to realistic bounds (max 0.4 kg/day)
        prev_weight = current_window[-1, 3]
        clamped_weight = max(prev_weight - 0.4, min(prev_weight + 0.4, next_weight))
        predicted_weights.append(round(clamped_weight, 2))

        # Roll window forward
        new_row = np.array([avg_in, avg_out, avg_net, clamped_weight], dtype=np.float32)
        current_window = np.vstack([current_window[1:], new_row])

    return predicted_weights


# ---------------------------------------------------------
# MAIN PREDICTION PIPELINE
# ---------------------------------------------------------

def predict_weight_trajectory(user_id, forecast_days=30):
    """
    Master prediction pipeline.
    Combines LSTM time-series modeling with thermodynamic energy balance baseline.
    """
    profile = get_user_profile(user_id)
    latest_w_obj = get_latest_weight(user_id)
    starting_w_obj = get_starting_weight(user_id)

    if not latest_w_obj or not profile:
        return {
            "status": "insufficient_data",
            "message": "Please set up your profile and record your body weight to enable AI weight predictions."
        }

    current_weight = float(latest_w_obj["weight_kg"])
    starting_weight = float(starting_w_obj["weight_kg"]) if starting_w_obj else current_weight
    height_cm = float(profile["height_cm"])
    age = int(profile["age"])
    gender = profile["gender"]
    goal = profile["goal"]
    activity_level = profile["activity_level"]

    # 1. Baseline Metabolic Calculations
    bmr = calculate_bmr(current_weight, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    # 2. Time Series Historical Analysis
    series = prepare_user_time_series(user_id)

    if series:
        # Use recent 14 days or all available
        recent = series[-14:]
        avg_calories_in = float(np.mean([s["calories_in"] for s in recent]))
        avg_exercise_burned = float(np.mean([s["calories_out"] for s in recent]))
    else:
        avg_calories_in = tdee
        avg_exercise_burned = 0.0

    # Total daily burn = TDEE baseline + logged physical activity
    total_daily_burn = tdee + avg_exercise_burned
    # Net daily energy balance: positive = surplus, negative = deficit
    daily_energy_balance = round(avg_calories_in - total_daily_burn, 1)

    # 3. Model Selection: LSTM vs. Metabolic Energy Balance
    model_type = "metabolic_model"
    model_name = "Metabolic Energy Balance"
    predicted_weights = None

    if len(series) >= 14:
        try:
            lstm_preds = train_lstm_and_predict(series, forecast_days=forecast_days, window_size=7)
            if lstm_preds and len(lstm_preds) == forecast_days:
                predicted_weights = lstm_preds
                model_type = "lstm_model"
                model_name = "LSTM Deep Learning Neural Network"
        except Exception as e:
            print("LSTM prediction fallback to thermodynamic model:", e)

    # Fallback to thermodynamic energy balance model (7,700 kcal ≈ 1 kg)
    if not predicted_weights:
        predicted_weights = []
        sim_weight = current_weight
        # 1 kg body mass ≈ 7700 kcal
        daily_delta_kg = daily_energy_balance / 7700.0

        for _ in range(forecast_days):
            sim_weight += daily_delta_kg
            # Prevent absurd negative numbers or impossible extremes
            sim_weight = max(25.0, min(350.0, sim_weight))
            predicted_weights.append(round(sim_weight, 2))

    # 4. Generate Trajectory Dates & Milestone Predictions
    today_dt = datetime.now()
    trajectory = []
    for i, w in enumerate(predicted_weights, start=1):
        future_date = (today_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        trajectory.append({
            "day": i,
            "date": future_date,
            "weight": w,
            "change": round(w - current_weight, 2)
        })

    proj_7 = trajectory[6] if len(trajectory) >= 7 else trajectory[-1]
    proj_14 = trajectory[13] if len(trajectory) >= 14 else trajectory[-1]
    proj_30 = trajectory[29] if len(trajectory) >= 30 else trajectory[-1]

    # 5. Personalized Recommendations & Insights
    insights = []
    if goal.lower() == "lose weight":
        if daily_energy_balance < -200:
            insights.append({
                "type": "success",
                "title": "On Track for Weight Loss",
                "text": f"Your current deficit of {abs(daily_energy_balance):.0f} kcal/day is optimal for safe, sustainable fat loss."
            })
        elif daily_energy_balance >= 0:
            insights.append({
                "type": "warning",
                "title": "Calorie Surplus Detected",
                "text": f"You are averaging a surplus of {daily_energy_balance:.0f} kcal/day against your target to lose weight. Aim to reduce portion sizes or increase daily activity."
            })
        else:
            insights.append({
                "type": "info",
                "title": "Mild Deficit",
                "text": f"You have a small deficit of {abs(daily_energy_balance):.0f} kcal/day. Increasing daily steps or exercise can accelerate your results."
            })
    elif goal.lower() == "gain weight":
        if daily_energy_balance > 250:
            insights.append({
                "type": "success",
                "title": "Lean Bulking Pace",
                "text": f"Your daily surplus of {daily_energy_balance:.0f} kcal/day supports steady muscle building without excessive fat accumulation."
            })
        else:
            insights.append({
                "type": "warning",
                "title": "Higher Intake Needed",
                "text": f"To gain weight, aim to eat 300–500 kcal above your TDEE ({total_daily_burn:.0f} kcal/day)."
            })
    else: # Maintain
        if abs(daily_energy_balance) <= 200:
            insights.append({
                "type": "success",
                "title": "Weight Maintenance Achieved",
                "text": "Your daily calorie intake is nicely balanced with your total daily burn."
            })
        elif daily_energy_balance > 200:
            insights.append({
                "type": "info",
                "title": "Slight Surplus",
                "text": f"You are consuming about {daily_energy_balance:.0f} kcal above maintenance."
            })
        else:
            insights.append({
                "type": "info",
                "title": "Slight Deficit",
                "text": f"You are consuming about {abs(daily_energy_balance):.0f} kcal below maintenance."
            })

    days_logged = len(series)
    lstm_unlocked = days_logged >= 14

    return {
        "status": "success",
        "model_type": model_type,
        "model_name": model_name,
        "days_logged": days_logged,
        "lstm_unlocked": lstm_unlocked,
        "current_weight": current_weight,
        "starting_weight": starting_weight,
        "bmr": bmr,
        "tdee": tdee,
        "total_daily_burn": round(total_daily_burn, 1),
        "avg_calories_in": round(avg_calories_in, 1),
        "avg_exercise_burned": round(avg_exercise_burned, 1),
        "daily_energy_balance": daily_energy_balance,
        "goal": goal,
        "activity_level": activity_level,
        "proj_7": proj_7,
        "proj_14": proj_14,
        "proj_30": proj_30,
        "trajectory": trajectory,
        "insights": insights
    }
