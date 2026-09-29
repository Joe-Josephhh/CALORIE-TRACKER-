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


def simulate_metabolic_trajectory(
    current_weight, height_cm, age, gender, activity_level,
    avg_calories_in, avg_exercise_burned, forecast_days=30
):
    """
    Simulates weight trajectory using Dynamic Metabolic Adaptation.
    Unlike static linear formulas, BMR and TDEE are dynamically recalculated
    daily as simulated weight changes, producing a realistic asymptotic plateau curve.
    """
    weights = []
    sim_weight = float(current_weight)

    for _ in range(1, forecast_days + 1):
        day_bmr = calculate_bmr(sim_weight, height_cm, age, gender)
        day_tdee = calculate_tdee(day_bmr, activity_level)
        day_burn = day_tdee + avg_exercise_burned
        day_balance = avg_calories_in - day_burn
        # 1 kg body mass ≈ 7,700 kcal
        daily_delta = day_balance / 7700.0
        # Physiological limits on fat/lean tissue change
        daily_delta = max(-0.25, min(0.25, daily_delta))
        sim_weight += daily_delta
        sim_weight = max(25.0, min(350.0, sim_weight))
        weights.append(round(sim_weight, 2))

    return weights


def train_lstm_and_predict(series, forecast_days=30, window_size=7):
    """
    Trains an LSTM on the user's sequential records if sufficient data exists (>= 14 days),
    incorporating day-of-week habit rhythms and smooth physiological dampening.
    """
    if len(series) < window_size * 2:
        return None

    # Prepare features: [calories_in, calories_out, net_calories, weight_kg]
    features = np.array([
        [s["calories_in"], s["calories_out"], s["net_calories"], s["weight_kg"]]
        for s in series
    ], dtype=np.float32)

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-5
    norm_features = (features - mean) / std

    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(norm_features[i:i + window_size])
        # Target: daily delta in weight
        y.append(features[i + window_size, 3] - features[i + window_size - 1, 3])

    X = np.array(X)
    y = np.array(y)

    model = build_lstm_model(input_shape=(window_size, 4))
    # Fine-tuning epochs for responsive web inference
    model.fit(X, y, epochs=30, batch_size=4, verbose=0)

    # Day-of-week habit patterns (e.g. weekend vs weekday intake)
    dow_in = {d: [] for d in range(7)}
    dow_out = {d: [] for d in range(7)}
    for s in series:
        try:
            dt = datetime.strptime(s["date"], "%Y-%m-%d")
            dow_in[dt.weekday()].append(s["calories_in"])
            dow_out[dt.weekday()].append(s["calories_out"])
        except Exception:
            pass

    overall_in = float(np.mean([s["calories_in"] for s in series[-window_size:]]))
    overall_out = float(np.mean([s["calories_out"] for s in series[-window_size:]]))

    dow_in_avg = {d: float(np.mean(dow_in[d])) if dow_in[d] else overall_in for d in range(7)}
    dow_out_avg = {d: float(np.mean(dow_out[d])) if dow_out[d] else overall_out for d in range(7)}

    # Iterative rolling prediction
    current_norm_window = norm_features[-window_size:].copy()
    current_weight = float(series[-1]["weight_kg"])
    today_dt = datetime.now()

    predicted_weights = []
    running_weight = current_weight

    for step in range(1, forecast_days + 1):
        input_data = current_norm_window.reshape(1, window_size, 4)
        pred_delta = float(model.predict(input_data, verbose=0)[0, 0])

        # Soft physiological dampening (smooth tanh squashing instead of hard clip)
        # Prevents hitting a flat wall while preserving dynamic curvature
        scaled_delta = float(0.20 * np.tanh(pred_delta / 0.15))
        running_weight += scaled_delta
        # Keep within physiological human boundaries
        running_weight = max(25.0, min(350.0, running_weight))
        predicted_weights.append(round(running_weight, 2))

        # Roll window forward with cyclical day-of-week habit values
        future_dt = today_dt + timedelta(days=step)
        f_dow = future_dt.weekday()
        f_in = dow_in_avg[f_dow]
        f_out = dow_out_avg[f_dow]
        f_net = f_in - f_out

        new_row_unnorm = np.array([f_in, f_out, f_net, running_weight], dtype=np.float32)
        new_row_norm = (new_row_unnorm - mean) / std
        current_norm_window = np.vstack([current_norm_window[1:], new_row_norm])

    return predicted_weights


# ---------------------------------------------------------
# MAIN PREDICTION PIPELINE
# ---------------------------------------------------------

def predict_weight_trajectory(user_id, forecast_days=30):
    """
    Master prediction pipeline.
    Combines LSTM time-series modeling with dynamic metabolic adaptation.
    Automatically gates model selection based on data sufficiency and recency (staleness).
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

    # 2. Time Series Historical Analysis & Staleness Check
    series = prepare_user_time_series(user_id)
    today_dt = datetime.now()
    today_date = today_dt.date()

    days_logged = len(series)
    days_since_last_log = 0
    is_stale = False
    last_log_str = today_date.strftime("%Y-%m-%d")

    if series:
        last_log_str = series[-1]["date"]
        try:
            last_log_date = datetime.strptime(last_log_str, "%Y-%m-%d").date()
            days_since_last_log = max(0, (today_date - last_log_date).days)
        except Exception:
            days_since_last_log = 0

        # Non-input staleness rule:
        # If user has not logged anything in > 3 days (4+ days ago), the time-series
        # momentum is broken. Fall back to the Metabolic Model.
        if days_since_last_log > 3:
            is_stale = True

        recent = series[-14:]
        avg_calories_in = float(np.mean([s["calories_in"] for s in recent]))
        avg_exercise_burned = float(np.mean([s["calories_out"] for s in recent]))
    else:
        avg_calories_in = tdee
        avg_exercise_burned = 0.0
        days_since_last_log = 999
        is_stale = True

    # Total daily burn = TDEE baseline + logged physical activity
    total_daily_burn = tdee + avg_exercise_burned
    # Net daily energy balance: positive = surplus, negative = deficit
    daily_energy_balance = round(avg_calories_in - total_daily_burn, 1)

    # 3. Model Computations:
    # Always compute the Dynamic Metabolic Adaptation baseline (plateau curve)
    metabolic_preds = simulate_metabolic_trajectory(
        current_weight=current_weight,
        height_cm=height_cm,
        age=age,
        gender=gender,
        activity_level=activity_level,
        avg_calories_in=avg_calories_in,
        avg_exercise_burned=avg_exercise_burned,
        forecast_days=forecast_days
    )

    # Model Selection: Check criteria for LSTM (>=14 days AND active recency <=3 days)
    lstm_preds = None
    lstm_unlocked = False
    model_type = "metabolic_model"
    model_name = "Metabolic Adaptation Model"
    status_reason = ""

    if days_logged >= 14:
        if is_stale:
            status_reason = f"LSTM model paused: No inputs recorded for {days_since_last_log} days. Showing Metabolic Adaptation Model."
        else:
            try:
                lstm_preds = train_lstm_and_predict(series, forecast_days=forecast_days, window_size=7)
                if lstm_preds and len(lstm_preds) == forecast_days:
                    lstm_unlocked = True
                    model_type = "lstm_model"
                    model_name = "LSTM Deep Learning Neural Network"
                    status_reason = "Running on active sliding-window habit sequences."
            except Exception as e:
                print("LSTM prediction fallback to metabolic model:", e)
                status_reason = "LSTM encountered an issue; utilizing Metabolic Adaptation Model."
    else:
        status_reason = f"Log {days_logged}/14 days to unlock personalized LSTM Deep Learning."

    # Active predicted weights
    predicted_weights = lstm_preds if (lstm_unlocked and lstm_preds) else metabolic_preds

    # 4. Generate Trajectory Dates & Milestone Predictions
    trajectory = []
    metabolic_trajectory = []
    lstm_trajectory = []

    for i in range(forecast_days):
        future_date = (today_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        active_w = predicted_weights[i]
        trajectory.append({
            "day": i + 1,
            "date": future_date,
            "weight": active_w,
            "change": round(active_w - current_weight, 2)
        })

        m_w = metabolic_preds[i]
        metabolic_trajectory.append({
            "day": i + 1,
            "date": future_date,
            "weight": m_w,
            "change": round(m_w - current_weight, 2)
        })

        if lstm_preds:
            l_w = lstm_preds[i]
            lstm_trajectory.append({
                "day": i + 1,
                "date": future_date,
                "weight": l_w,
                "change": round(l_w - current_weight, 2)
            })

    proj_7 = trajectory[6] if len(trajectory) >= 7 else trajectory[-1]
    proj_14 = trajectory[13] if len(trajectory) >= 14 else trajectory[-1]
    proj_30 = trajectory[29] if len(trajectory) >= 30 else trajectory[-1]

    # 5. Personalized Recommendations & Insights
    insights = []
    goal_str = goal.lower()
    if "loss" in goal_str or "lose" in goal_str:
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
    elif "gain" in goal_str:
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

    if is_stale and days_logged >= 14:
        insights.insert(0, {
            "type": "warning",
            "title": "Tracking Paused — Model Switched",
            "text": f"You haven't logged any diary or weight entries in {days_since_last_log} days. Time-series neural models require continuous habit tracking, so the system has temporarily switched to your profile's Metabolic Adaptation Model. Log food today to reactivate LSTM!"
        })

    return {
        "status": "success",
        "model_type": model_type,
        "model_name": model_name,
        "status_reason": status_reason,
        "days_logged": days_logged,
        "lstm_unlocked": lstm_unlocked,
        "is_stale": is_stale,
        "days_since_last_log": days_since_last_log,
        "last_log_date": last_log_str,
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
        "metabolic_trajectory": metabolic_trajectory,
        "lstm_trajectory": lstm_trajectory,
        "insights": insights
    }

