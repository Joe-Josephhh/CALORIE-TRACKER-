import numpy as np
from datetime import datetime, timedelta
from database import (
    get_user_profile,
    get_latest_weight,
    get_starting_weight,
    get_weight_history,
    get_user_history_daily
)

# =====================================================================
# TIME-SERIES WEIGHT PREDICTION ENGINE (HYBRID LSTM + METABOLIC MODEL)
# =====================================================================
#
# Core Pipeline Overview:
# 1. Historical Data: Daily chronological logs of food intake, exercise burn,
#    and body weight records stored in SQLite.
# 2. Sliding Windows: A 7-day observation window [Day t-6 .. Day t] used to
#    predict next-day body weight change (Day t+1).
# 3. Training Sequences: For n historical days and window size W=7, exactly
#    (n - W) sequence pairs (X, y) are generated.
# 4. Chronological Validation: Time-ordered split without random shuffling.
#    Earlier sequences train the model; future sequences evaluate unseen error.
# 5. Normalization without Leakage: Feature scaling statistics (mean, std)
#    are fitted EXCLUSIVELY on the training slice.
# 6. Next-Day Delta Target: The LSTM predicts weight change (delta kg),
#    not absolute weight, avoiding drift and scale dependence.
# 7. Latest 7-Day Context: After training, recursive forecasting begins from
#    the latest 7 actual historical days [Day n-6 .. Day n].
# 8. Recursive Forecasting: Autoregressive multi-step rollout forward in time,
#    incorporating day-of-week habit patterns and dynamic rolling features.
# 9. Metabolic Baseline & Hybrid Ensemble: Combines empirical LSTM habit
#    dynamics in the near term with thermodynamic metabolic adaptation over
#    longer horizons.
# =====================================================================


# ---------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = 7       # 7-day historical observation window
DEFAULT_VAL_SPLIT = 0.20       # 20% chronological holdout for unseen future validation

# Configurable feature set: 4 base features + 4 rolling/trend features.
# The model input shape adapts dynamically to the number of configured features.
DEFAULT_FEATURE_NAMES = [
    "calories_in",      # 0: Daily food intake (kcal)
    "calories_out",     # 1: Daily exercise / activity burn (kcal)
    "net_calories",     # 2: Daily energy balance: in - out (kcal)
    "weight_kg",        # 3: Weight on day t (kg)
    "weight_7d_avg",    # 4: 7-day rolling mean weight (kg) - filters acute water swings
    "weight_7d_trend",  # 5: 7-day weight change: weight[t] - weight[t-6] (kg) - weekly momentum
    "cal_in_7d_avg",    # 6: 7-day rolling mean calorie intake (kcal)
    "cal_net_7d_avg"    # 7: 7-day rolling mean net calorie balance (kcal)
]

# Wishnofsky's rule (1958): 1 kg body mass ≈ 7,700 kcal.
# NOTE ON PHYSIOLOGICAL APPROXIMATION:
# 7,700 kcal/kg is a classical rule-of-thumb based on ~87% lipid content and ~13% water/protein
# in human adipose tissue. In living physiology, real-world weight changes diverge from this
# static constant due to glycogen depletion (which binds ~3g water per gram glycogen), fluid shifts,
# changes in lean body mass, and metabolic adaptation (adaptive thermogenesis). It is retained
# here as a robust baseline approximation, but should not be treated as an exact linear guarantee.
CALORIES_PER_KG_FAT = 7700.0


# ---------------------------------------------------------------------
# 1. METABOLIC BASICS (Mifflin-St Jeor & Expenditure Semantics)
# ---------------------------------------------------------------------

def calculate_bmr(weight_kg, height_cm, age, gender="Other"):
    """
    Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation.
    Men:   10 * weight + 6.25 * height - 5 * age + 5
    Women: 10 * weight + 6.25 * height - 5 * age - 161
    Other: Average of male and female constants (-78)
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
    Calculate Total Daily Energy Expenditure (TDEE) multiplier baseline.
    Multipliers:
      sedentary: 1.2        (desk job, minimal incidental movement)
      lightly active: 1.375 (light exercise 1-3 days/week)
      moderately active: 1.55 (moderate exercise 3-5 days/week)
      very active: 1.725    (hard exercise 6-7 days/week)
      super active: 1.9     (physical labor / intense training)
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


def calculate_daily_burn(bmr, activity_level, logged_exercise_burned=0.0):
    """
    Computes total daily energy expenditure while preventing double-counting of exercise.
    
    EXPENDITURE SEMANTICS & DOUBLE-COUNTING PREVENTION:
    - BMR: Basal energy expended in a post-absorptive, resting state.
    - Sedentary TDEE (BMR * 1.2): Non-exercise activity thermogenesis (NEAT) + thermic effect of food (TEF).
    - Multiplier TDEE: Activity multipliers above 1.2 (1.375 to 1.9) ALREADY incorporate an assumed
      allowance for planned exercise sessions (e.g. 1.55 assumes 3-5 moderate workouts per week).
    - Logged Exercise (calorie_out_records): Discrete workout sessions explicitly tracked by the user.
    
    Double-Counting Risk:
    If a user has an active profile multiplier (e.g. 1.55) AND separately logs 400 kcal of running,
    simply doing (tdee + exercise) adds the workout twice: once inside the 1.55 multiplier and once from
    the log.
    
    Safe Integration Rule:
    When explicit exercise is logged, the non-exercise baseline is Sedentary TDEE (BMR * 1.2).
    Total burn = Sedentary TDEE + logged exercise.
    If the profile TDEE with general activity exceeds that sum (e.g. active occupation), we take
    max(profile_tdee, sedentary_tdee + logged_exercise).
    If no exercise is logged, profile_tdee is used directly.
    """
    sedentary_tdee = calculate_tdee(bmr, "sedentary")
    profile_tdee = calculate_tdee(bmr, activity_level)

    if logged_exercise_burned > 0:
        return round(max(profile_tdee, sedentary_tdee + logged_exercise_burned), 1)
    else:
        return profile_tdee


def simulate_metabolic_trajectory(
    current_weight, height_cm, age, gender, activity_level,
    avg_calories_in, avg_exercise_burned, forecast_days=30
):
    """
    Simulates weight trajectory using Dynamic Metabolic Adaptation.
    Daily BMR and daily burn are dynamically recalculated as simulated weight changes,
    producing a realistic asymptotic plateau curve rather than unrealistic linear decline.
    """
    weights = []
    sim_weight = float(current_weight)

    for _ in range(1, forecast_days + 1):
        day_bmr = calculate_bmr(sim_weight, height_cm, age, gender)
        day_burn = calculate_daily_burn(day_bmr, activity_level, avg_exercise_burned)
        day_balance = avg_calories_in - day_burn

        # Wishnofsky approximation: 1 kg body mass ≈ 7,700 kcal
        daily_delta = day_balance / CALORIES_PER_KG_FAT

        # Physiological limit on daily tissue change rate under sustainable human conditions
        daily_delta = max(-0.25, min(0.25, daily_delta))
        sim_weight += daily_delta

        # Human boundary bounds (25.0 kg to 350.0 kg)
        sim_weight = max(25.0, min(350.0, sim_weight))
        weights.append(round(sim_weight, 2))

    return weights


# ---------------------------------------------------------------------
# 2. TIME SERIES EXTRACTION & MISSING WEIGHT HANDLING
# ---------------------------------------------------------------------

def prepare_user_time_series(user_id):
    """
    Extract chronological daily series of [calories_in, calories_out, net_calories, weight_kg].
    
    MISSING WEIGHT HANDLING (Eliminating Artificial Flatlines):
    - Forward-filling missing scale readings blindly treats unmeasured days as zero-change days.
      For example, if Day 20 = 70.0 kg, Days 21-23 are missing, and Day 24 = 69.5 kg:
      Forward-fill produces: 70.0 (delta 0), 70.0 (delta 0), 70.0 (delta 0), 69.5 (delta -0.5).
      This creates artificial zero-delta targets and sudden steep cliffs that corrupt LSTM gradients.
    - Safer Approach Implemented:
      1. Linear interpolation between observed anchor weigh-ins, distributing gradual daily delta.
      2. Tracking 'weight_measured' (True for genuine scale readings, False for imputed values).
      3. Backfilling leading unmeasured days from the first observed measurement.
      4. Forward-filling trailing unmeasured days from the latest observed measurement.
      5. Downstream sequence training applies soft sample weighting to prevent fitting to
         synthetic target artifacts.
    """
    history = get_user_history_daily(user_id, order="ASC")
    if not history:
        return []

    latest_w_obj = get_latest_weight(user_id)
    fallback_weight = float(latest_w_obj["weight_kg"]) if latest_w_obj else 70.0

    n = len(history)
    measured_indices = [i for i, entry in enumerate(history) if entry["weight_kg"] is not None]

    imputed_weights = [None] * n
    weight_measured_flags = [False] * n

    if not measured_indices:
        # No weights ever recorded; fallback throughout
        imputed_weights = [fallback_weight] * n
    elif len(measured_indices) == 1:
        # Single weight recorded; constant throughout
        single_weight = float(history[measured_indices[0]]["weight_kg"])
        imputed_weights = [single_weight] * n
        weight_measured_flags[measured_indices[0]] = True
    else:
        first_idx = measured_indices[0]
        first_val = float(history[first_idx]["weight_kg"])
        last_idx = measured_indices[-1]
        last_val = float(history[last_idx]["weight_kg"])

        # Backfill leading unmeasured days before the first measurement
        for i in range(0, first_idx):
            imputed_weights[i] = first_val
            weight_measured_flags[i] = False

        # Forward-fill trailing unmeasured days after the last measurement
        for i in range(last_idx + 1, n):
            imputed_weights[i] = last_val
            weight_measured_flags[i] = False

        # Mark anchor weigh-ins
        for idx in measured_indices:
            imputed_weights[idx] = float(history[idx]["weight_kg"])
            weight_measured_flags[idx] = True

        # Linearly interpolate gaps between measured anchors
        for k in range(len(measured_indices) - 1):
            start_i = measured_indices[k]
            end_i = measured_indices[k + 1]
            start_w = imputed_weights[start_i]
            end_w = imputed_weights[end_i]
            gap = end_i - start_i

            for j in range(start_i + 1, end_i):
                fraction = (j - start_i) / float(gap)
                interpolated_w = start_w + fraction * (end_w - start_w)
                imputed_weights[j] = round(interpolated_w, 2)
                weight_measured_flags[j] = False

    series = []
    for i, entry in enumerate(history):
        w = imputed_weights[i] if imputed_weights[i] is not None else fallback_weight
        series.append({
            "date": entry["date"],
            "calories_in": float(entry["calories_in"]),
            "calories_out": float(entry["calories_out"]),
            "net_calories": float(entry["net_calories"]),
            "weight_kg": float(w),
            "weight_measured": weight_measured_flags[i]
        })

    return series


# ---------------------------------------------------------------------
# 3. FEATURE EXTRACTION & ROLLING / TREND FEATURES
# ---------------------------------------------------------------------

def extract_feature_matrix(series, feature_names=None):
    """
    Extracts a 2D numpy array of shape (len(series), len(feature_names)).
    
    TEMPORAL & TREND FEATURES:
    In addition to daily intake, burn, and weight, rolling features are computed:
    - weight_7d_avg: 7-day rolling mean weight (dampens day-to-day fluid noise)
    - weight_7d_trend: 7-day change weight[t] - weight[t-6] (indicates weekly momentum)
    - cal_in_7d_avg: 7-day rolling mean calorie intake
    - cal_net_7d_avg: 7-day rolling mean net energy balance
    
    These temporal features allow the LSTM to distinguish transient single-day fluctuations
    from true underlying trajectory shifts.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    n = len(series)
    k = len(feature_names)
    matrix = np.zeros((n, k), dtype=np.float32)

    cal_in = np.array([float(s["calories_in"]) for s in series], dtype=np.float32)
    cal_out = np.array([float(s["calories_out"]) for s in series], dtype=np.float32)
    cal_net = np.array([float(s["net_calories"]) for s in series], dtype=np.float32)
    weights = np.array([float(s["weight_kg"]) for s in series], dtype=np.float32)

    for i in range(n):
        # 7-day historical window ending at day i: max(0, i - 6) to i inclusive
        start_w = max(0, i - 6)
        w_slice = weights[start_w:i + 1]
        cin_slice = cal_in[start_w:i + 1]
        cnet_slice = cal_net[start_w:i + 1]

        row_map = {
            "calories_in": cal_in[i],
            "calories_out": cal_out[i],
            "net_calories": cal_net[i],
            "weight_kg": weights[i],
            "weight_7d_avg": float(np.mean(w_slice)),
            "weight_7d_trend": float(weights[i] - weights[start_w]),
            "cal_in_7d_avg": float(np.mean(cin_slice)),
            "cal_net_7d_avg": float(np.mean(cnet_slice))
        }

        for col_idx, col_name in enumerate(feature_names):
            matrix[i, col_idx] = row_map.get(col_name, 0.0)

    return matrix


# ---------------------------------------------------------------------
# 4. LIGHTWEIGHT LSTM DEEP LEARNING ARCHITECTURE
# ---------------------------------------------------------------------

def build_lstm_model(input_shape=(DEFAULT_WINDOW_SIZE, len(DEFAULT_FEATURE_NAMES)), seed=42):
    """
    Builds a compact Keras LSTM model for sequential next-day delta prediction.
    
    Architecture:
      Input(shape=(7, num_features))
      LSTM(16, activation="tanh", return_sequences=False)
      Dense(8, activation="relu")
      Dense(1)
      
    Optimizer: Adam
    Loss: Mean Squared Error (MSE)
    
    Design Rationale:
    Per-user health tracking data is inherently low-sample (14 to ~365 daily points).
    A compact network (16 LSTM units -> 8 Dense units) prevents overfitting, minimizes
    parameter count, trains in <1 second on CPU, and maintains high generalizability.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if seed is not None:
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(16, activation="tanh", return_sequences=False),
        layers.Dense(8, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------
# 5. CHRONOLOGICAL TRAINING, VALIDATION & RECURSIVE FORECASTING
# ---------------------------------------------------------------------

def train_lstm_and_predict(
    series,
    forecast_days=30,
    window_size=DEFAULT_WINDOW_SIZE,
    val_split=DEFAULT_VAL_SPLIT,
    feature_names=None,
    return_details=False,
    seed=42
):
    """
    Trains an LSTM on user sequential records using strictly chronological validation.
    
    EXACT SEQUENCE COUNT (n Historical Days with Window W=7):
      Sequence 0:     Days 0..6 (1..7)     -> target is delta to Day 7 (Day 8)
      Sequence 1:     Days 1..7 (2..8)     -> target is delta to Day 8 (Day 9)
      ...
      Sequence M-1:   Days (n-8)..(n-2)   -> target is delta to Day n-1 (Day n)
      Exact sequence count: M = n - W (e.g. 150 - 7 = 143 sequences; 180 - 7 = 173 sequences).
    
    DATA LEAKAGE PREVENTION:
      Standardizing time-series data with mean/std computed over the entire dataset introduces
      lookahead bias (information from future days leaks into past days).
      Here, the scaler (mean, std) is fitted EXCLUSIVELY on the training slice [0 .. train_cutoff_day].
      Validation data and future forecast inputs are transformed using these exact training parameters.
    
    CHRONOLOGICAL VALIDATION:
      Data is NOT randomly shuffled. The earliest sequences form the training set, while
      the later sequences form genuinely unseen future validation data.
      Validation metrics reported:
      - Weight MAE (kg) and Weight RMSE (kg): predicted actual weight vs true historical weight
      - Delta MAE (kg) and Delta RMSE (kg): predicted daily change vs true daily change
    
    RECURSIVE FUTURE FORECASTING:
      Begins from the latest 7 actual historical days:
        Days (n-7)..(n-1) -> predict Day n+1
        Days (n-6)..(n-1) + predicted Day n+1 -> predict Day n+2, etc.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    import random
    import tensorflow as tf

    # DETERMINISTIC TRAINING & SEEDING:
    # Health habit sequences are compact (14-365 days). Without an explicit seed,
    # random weight initialization and mini-batch stochasticity cause the model to converge
    # to different weights on consecutive requests, producing jumping predictions even
    # when the user has not logged anything new. Setting a deterministic seed guarantees
    # stability while still allowing the model to adapt when new data is entered.
    if seed is not None:
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        random.seed(seed)
        np.random.seed(seed)

    n_days = len(series)
    if n_days < window_size * 2:
        return (None, None) if return_details else None

    # Extract full feature matrix of shape (n_days, num_features)
    raw_features = extract_feature_matrix(series, feature_names=feature_names)
    num_features = len(feature_names)

    # Total possible sequence pairs: M = n_days - window_size
    total_sequences = n_days - window_size

    # Chronological Split
    val_count = max(3, int(total_sequences * val_split))
    train_count = total_sequences - val_count

    # The latest historical day index observed during training:
    # Sequence (train_count - 1) uses input window ending at (train_count + window_size - 2)
    # and targets day (train_count + window_size - 1).
    train_day_cutoff = train_count + window_size

    # FIT SCALER ON TRAINING PORTION ONLY (Prevent Data Leakage)
    train_slice = raw_features[:train_day_cutoff]
    train_mean = train_slice.mean(axis=0)
    train_std = train_slice.std(axis=0)
    train_std = np.where(train_std < 1e-5, 1.0, train_std)  # Avoid division by zero

    # Transform all historical features using training statistics
    norm_features = (raw_features - train_mean) / train_std

    # Build sequence pairs (X, y) and sample weights
    X_all, y_all, sample_weights = [], [], []

    for i in range(total_sequences):
        X_all.append(norm_features[i:i + window_size])
        # Target: daily delta in weight (kg) from day t to day t+1
        target_delta = raw_features[i + window_size, 3] - raw_features[i + window_size - 1, 3]
        y_all.append(target_delta)

        # Missing weight target safety:
        # Actual scale readings receive full weight (1.0).
        # Linearly interpolated values receive soft weight (0.5) so artificial
        # interpolation artifacts do not dominate the training loss gradient.
        target_entry = series[i + window_size]
        sw = 1.0 if target_entry.get("weight_measured", True) else 0.5
        sample_weights.append(sw)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    sample_weights = np.array(sample_weights, dtype=np.float32)

    # Chronological partition
    X_train = X_all[:train_count]
    y_train = y_all[:train_count]
    sw_train = sample_weights[:train_count]

    X_val = X_all[train_count:]
    y_val = y_all[train_count:]
    sw_val = sample_weights[train_count:]

    # Train model on training data only
    model = build_lstm_model(input_shape=(window_size, num_features), seed=seed)
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        epochs=30,
        batch_size=4,
        verbose=0
    )

    # Chronological Validation Evaluation
    val_pred_deltas = model.predict(X_val, verbose=0).flatten()

    delta_mae = float(np.mean(np.abs(val_pred_deltas - y_val)))
    delta_rmse = float(np.sqrt(np.mean((val_pred_deltas - y_val) ** 2)))

    # Evaluate predicted actual weight against actual historical weight
    val_pred_weights = []
    val_actual_weights = []
    for j in range(len(X_val)):
        base_day = train_count + j + window_size - 1
        target_day = train_count + j + window_size
        base_w = raw_features[base_day, 3]
        act_w = raw_features[target_day, 3]

        # Apply smooth physiological dampening to delta
        pred_d = float(0.20 * np.tanh(val_pred_deltas[j] / 0.15))
        val_pred_weights.append(base_w + pred_d)
        val_actual_weights.append(act_w)

    val_pred_weights = np.array(val_pred_weights)
    val_actual_weights = np.array(val_actual_weights)

    weight_mae = float(np.mean(np.abs(val_pred_weights - val_actual_weights)))
    weight_rmse = float(np.sqrt(np.mean((val_pred_weights - val_actual_weights) ** 2)))

    val_metrics = {
        "train_sequences": int(train_count),
        "val_sequences": int(val_count),
        "total_sequences": int(total_sequences),
        "weight_mae": round(weight_mae, 3),
        "weight_rmse": round(weight_rmse, 3),
        "delta_mae": round(delta_mae, 4),
        "delta_rmse": round(delta_rmse, 4)
    }

    # Day-of-week habit rhythms (captures weekend vs weekday calorie intake and activity)
    dow_in = {d: [] for d in range(7)}
    dow_out = {d: [] for d in range(7)}
    for s in series:
        try:
            dt = datetime.strptime(s["date"], "%Y-%m-%d")
            dow_in[dt.weekday()].append(s["calories_in"])
            dow_out[dt.weekday()].append(s["calories_out"])
        except Exception:
            pass

    recent_slice = series[-window_size:]
    overall_in = float(np.mean([s["calories_in"] for s in recent_slice]))
    overall_out = float(np.mean([s["calories_out"] for s in recent_slice]))

    dow_in_avg = {d: float(np.mean(dow_in[d])) if dow_in[d] else overall_in for d in range(7)}
    dow_out_avg = {d: float(np.mean(dow_out[d])) if dow_out[d] else overall_out for d in range(7)}

    # RECURSIVE FUTURE PREDICTION
    # The real future forecast begins from the latest 7 actual historical days:
    # Days (n-7)..(n-1) -> predict Day n+1
    # Days (n-6)..(n-1) + predicted Day n+1 -> predict Day n+2, etc.
    forecast_buffer = [dict(s) for s in series]
    current_weight = float(series[-1]["weight_kg"])
    running_weight = current_weight
    today_dt = datetime.now()

    predicted_weights = []

    for step in range(1, forecast_days + 1):
        # Extract features for the latest window_size days in forecast_buffer
        last_window_slice = forecast_buffer[-window_size:]
        raw_window = extract_feature_matrix(last_window_slice, feature_names=feature_names)
        norm_window = (raw_window - train_mean) / train_std
        input_data = norm_window.reshape(1, window_size, num_features)

        pred_delta = float(model.predict(input_data, verbose=0)[0, 0])

        # Smooth physiological dampening:
        # Scaled delta = 0.20 * tanh(pred_delta / 0.15)
        # Prevents runaway recursive drift while allowing smooth non-linear curvature
        scaled_delta = float(0.20 * np.tanh(pred_delta / 0.15))
        # Physiological safety clamp: max ±0.30 kg/day sustained tissue shift
        scaled_delta = max(-0.30, min(0.30, scaled_delta))

        running_weight += scaled_delta
        # Human boundary bounds (25.0 kg to 350.0 kg)
        running_weight = max(25.0, min(350.0, running_weight))
        predicted_weights.append(round(running_weight, 2))

        # Advance calendar step with cyclical day-of-week habit averages
        future_dt = today_dt + timedelta(days=step)
        f_dow = future_dt.weekday()
        f_in = dow_in_avg[f_dow]
        f_out = dow_out_avg[f_dow]
        f_net = f_in - f_out

        # Append predicted day to forecast_buffer so subsequent steps have accurate rolling features
        forecast_buffer.append({
            "date": future_dt.strftime("%Y-%m-%d"),
            "calories_in": f_in,
            "calories_out": f_out,
            "net_calories": f_net,
            "weight_kg": running_weight,
            "weight_measured": False
        })

    if return_details:
        return predicted_weights, val_metrics
    return predicted_weights


# ---------------------------------------------------------------------
# 6. MASTER PREDICTION PIPELINE (HYBRID MODEL)
# ---------------------------------------------------------------------

def predict_weight_trajectory(user_id, forecast_days=30):
    """
    Master prediction pipeline.
    Combines LSTM time-series modeling with dynamic metabolic adaptation.
    Automatically gates model selection based on data sufficiency and recency (staleness).
    
    RETRAINING BEHAVIOR:
    The LSTM is retrained on-demand when predict_weight_trajectory() is called.
    Because the model is lightweight (16 LSTM units, ~1s training) and user historical series
    are compact (14-365 days), on-demand training ensures each user receives instantaneous
    incorporation of their latest daily nutrition and weigh-in data without requiring a complex
    server-side model persistence registry.
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

    # Total daily burn: accurately accounts for exercise without double-counting against activity multiplier
    total_daily_burn = calculate_daily_burn(bmr, activity_level, avg_exercise_burned)
    # Net daily energy balance: positive = surplus, negative = deficit
    daily_energy_balance = round(avg_calories_in - total_daily_burn, 1)

    # 3. Model Computations:
    # A. Dynamic Metabolic Adaptation baseline (plateau curve)
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

    # B. LSTM Deep Learning Model (trained and chronologically validated)
    lstm_preds = None
    lstm_unlocked = False
    validation_metrics = None
    model_type = "metabolic_model"
    model_name = "Metabolic Adaptation Model"
    status_reason = ""

    if days_logged >= 14:
        if is_stale:
            status_reason = f"LSTM model paused: No inputs recorded for {days_since_last_log} days. Showing Metabolic Adaptation Model."
        else:
            try:
                lstm_preds, validation_metrics = train_lstm_and_predict(
                    series,
                    forecast_days=forecast_days,
                    window_size=DEFAULT_WINDOW_SIZE,
                    val_split=DEFAULT_VAL_SPLIT,
                    feature_names=DEFAULT_FEATURE_NAMES,
                    return_details=True,
                    seed=42
                )
                if lstm_preds and len(lstm_preds) == forecast_days:
                    lstm_unlocked = True
                    model_type = "lstm_model"
                    model_name = "LSTM Deep Learning & Metabolic Hybrid"
                    status_reason = (
                        f"Running on personalized sliding-window habit sequences. "
                        f"Chronologically validated on {validation_metrics['val_sequences']} unseen days "
                        f"(MAE: {validation_metrics['weight_mae']} kg, RMSE: {validation_metrics['weight_rmse']} kg)."
                    )
            except Exception as e:
                print("LSTM prediction fallback to metabolic model:", e)
                status_reason = "LSTM encountered an issue; utilizing Metabolic Adaptation Model."
    else:
        status_reason = f"Log {days_logged}/14 days to unlock personalized LSTM Deep Learning."

    # C. Hybrid Trajectory Construction
    # Blends empirical LSTM habit sensitivity in the near-term (days 1-7)
    # with thermodynamic metabolic adaptation stability over longer forecast horizons (days 8-30).
    hybrid_preds = []
    if lstm_unlocked and lstm_preds:
        # Confidence weighting: if validation RMSE is high, down-weight LSTM
        val_rmse = validation_metrics.get("weight_rmse", 0.5) if validation_metrics else 0.5
        rmse_conf = max(0.4, min(1.0, 1.0 / max(1.0, val_rmse)))

        for step_idx in range(forecast_days):
            # Horizon decay factor: 0.80 weight at day 1 smoothly decays to 0.45 at day 30
            horizon_frac = step_idx / float(max(1, forecast_days - 1))
            alpha = max(0.35, 0.80 - 0.35 * horizon_frac) * rmse_conf
            h_weight = alpha * lstm_preds[step_idx] + (1.0 - alpha) * metabolic_preds[step_idx]
            hybrid_preds.append(round(h_weight, 2))
    else:
        hybrid_preds = list(metabolic_preds)

    # Active predicted weights for display (Hybrid if LSTM available, else Metabolic baseline)
    predicted_weights = hybrid_preds if lstm_unlocked else metabolic_preds

    # 4. Generate Trajectory Dates & Milestone Predictions
    trajectory = []
    metabolic_trajectory = []
    lstm_trajectory = []
    hybrid_trajectory = []

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

            h_w = hybrid_preds[i]
            hybrid_trajectory.append({
                "day": i + 1,
                "date": future_date,
                "weight": h_w,
                "change": round(h_w - current_weight, 2)
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
                "text": f"To gain weight, aim to eat 300–500 kcal above your total daily burn ({total_daily_burn:.0f} kcal/day)."
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
        "hybrid_trajectory": hybrid_trajectory,
        "validation_metrics": validation_metrics,
        "feature_names": DEFAULT_FEATURE_NAMES,
        "insights": insights
    }
