import sqlite3
from pathlib import Path
from datetime import datetime


# =========================================================
# DATABASE LOCATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "calorie_tracker.db"


# =========================================================
# GET DATABASE CONNECTION
# =========================================================

def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


# =========================================================
# INITIALIZE DATABASE & TABLES
# =========================================================

def init_database():
    connection = get_connection()
    cursor = connection.cursor()

    # 1. USERS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. USER PROFILES TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            height_cm REAL NOT NULL,
            goal TEXT NOT NULL,
            activity_level TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 3. WEIGHT RECORDS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            recorded_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 4. CALORIE IN RECORDS (Food Intake)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calorie_in_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dish_name TEXT NOT NULL,
            quantity REAL NOT NULL,
            meal_type TEXT,
            calories REAL DEFAULT 0.0,
            carbohydrates REAL DEFAULT 0.0,
            protein REAL DEFAULT 0.0,
            fats REAL DEFAULT 0.0,
            free_sugar REAL DEFAULT 0.0,
            fibre REAL DEFAULT 0.0,
            sodium REAL DEFAULT 0.0,
            calcium REAL DEFAULT 0.0,
            iron REAL DEFAULT 0.0,
            vitamin_c REAL DEFAULT 0.0,
            folate REAL DEFAULT 0.0,
            notes TEXT DEFAULT '',
            recorded_date TEXT NOT NULL,
            recorded_time TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 5. CALORIE OUT RECORDS (Exercise / Activity)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calorie_out_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity TEXT NOT NULL,
            met REAL NOT NULL,
            weight_kg REAL NOT NULL,
            duration_minutes REAL NOT NULL,
            calories_burned REAL NOT NULL,
            recorded_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 6. LEGACY EXERCISE RECORDS (Preserved for backward compatibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exercise_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity TEXT NOT NULL,
            met REAL NOT NULL,
            weight_kg REAL NOT NULL,
            duration_minutes REAL NOT NULL,
            calories_burned REAL NOT NULL,
            recorded_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Indexes for fast querying per user and date
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_weight_user_date ON weight_records(user_id, recorded_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cal_in_user_date ON calorie_in_records(user_id, recorded_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cal_out_user_date ON calorie_out_records(user_id, recorded_date)")

    connection.commit()
    connection.close()


# =========================================================
# USER QUERIES
# =========================================================

def create_user(username, email, password_hash):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
    """, (username.strip(), email.strip().lower(), password_hash))
    user_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return user_id


def get_user_by_id(user_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    connection.close()
    return user


def get_user_by_email(email):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    user = cursor.fetchone()
    connection.close()
    return user


def get_user_by_username(username):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
    user = cursor.fetchone()
    connection.close()
    return user


# =========================================================
# USER PROFILE QUERIES
# =========================================================

def create_user_profile(user_id, age, gender, height_cm, goal, activity_level):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO user_profiles (user_id, age, gender, height_cm, goal, activity_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, int(age), str(gender), float(height_cm), str(goal), str(activity_level)))
    connection.commit()
    connection.close()


def get_user_profile(user_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
    profile = cursor.fetchone()
    connection.close()
    return profile


def update_user_profile(user_id, age, gender, height_cm, goal, activity_level):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE user_profiles
        SET age = ?, gender = ?, height_cm = ?, goal = ?, activity_level = ?
        WHERE user_id = ?
    """, (int(age), str(gender), float(height_cm), str(goal), str(activity_level), user_id))
    connection.commit()
    connection.close()


# =========================================================
# WEIGHT TRACKING QUERIES
# =========================================================

def add_weight_record(user_id, weight_kg, recorded_date=None):
    if recorded_date is None:
        recorded_date = datetime.now().strftime("%Y-%m-%d")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO weight_records (user_id, weight_kg, recorded_date)
        VALUES (?, ?, ?)
    """, (user_id, float(weight_kg), recorded_date))
    record_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return record_id


def get_latest_weight(user_id):
    """
    Returns the most recently recorded weight record (Row or None).
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM weight_records
        WHERE user_id = ?
        ORDER BY recorded_date DESC, created_at DESC
        LIMIT 1
    """, (user_id,))
    record = cursor.fetchone()
    connection.close()
    return record


def get_starting_weight(user_id):
    """
    Returns the earliest recorded weight record (Row or None).
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM weight_records
        WHERE user_id = ?
        ORDER BY recorded_date ASC, created_at ASC
        LIMIT 1
    """, (user_id,))
    record = cursor.fetchone()
    connection.close()
    return record


def get_weight_history(user_id, limit=100):
    """
    Returns weight records for the user ordered chronologically.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM weight_records
        WHERE user_id = ?
        ORDER BY recorded_date ASC, created_at ASC
        LIMIT ?
    """, (user_id, limit))
    records = cursor.fetchall()
    connection.close()
    return records


def get_previous_weight(user_id):
    """
    Returns the weight record recorded immediately before the latest one (Row or None).
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM weight_records
        WHERE user_id = ?
        ORDER BY recorded_date DESC, created_at DESC
        LIMIT 1 OFFSET 1
    """, (user_id,))
    record = cursor.fetchone()
    connection.close()
    return record


def get_weight_stats(user_id):
    """
    Calculates starting weight, latest weight, previous weight, total change,
    recent change, and total number of weigh-ins.
    """
    starting = get_starting_weight(user_id)
    latest = get_latest_weight(user_id)
    previous = get_previous_weight(user_id)
    history = get_weight_history(user_id)

    starting_val = float(starting["weight_kg"]) if starting else None
    latest_val = float(latest["weight_kg"]) if latest else None
    previous_val = float(previous["weight_kg"]) if previous else None

    total_change = round(latest_val - starting_val, 2) if (latest_val is not None and starting_val is not None) else 0.0
    recent_change = round(latest_val - previous_val, 2) if (latest_val is not None and previous_val is not None) else 0.0

    return {
        "starting_weight": starting_val,
        "starting_date": starting["recorded_date"] if starting else None,
        "latest_weight": latest_val,
        "latest_date": latest["recorded_date"] if latest else None,
        "previous_weight": previous_val,
        "previous_date": previous["recorded_date"] if previous else None,
        "total_change": total_change,
        "recent_change": recent_change,
        "count": len(history),
        "history": history
    }


def delete_weight_record(user_id, record_id):
    """
    Safely deletes a weight record belonging strictly to the authenticated user.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM weight_records
        WHERE id = ? AND user_id = ?
    """, (record_id, user_id))
    affected = cursor.rowcount
    connection.commit()
    connection.close()
    return affected > 0



# =========================================================
# CALORIE IN (FOOD) QUERIES
# =========================================================

def format_calorie_in_row(row):
    """
    Helper to convert a calorie_in_records Row into a dict compatible
    with existing templates that expect food['nutrition']['Calories (kcal)']
    and food['weight'].
    """
    if row is None:
        return None
    d = dict(row)
    d["weight"] = d.get("quantity", 0.0)
    d["date"] = d.get("recorded_date", "")
    d["time"] = d.get("recorded_time", "")
    d["nutrition"] = {
        "Calories (kcal)": d.get("calories", 0.0),
        "Protein (g)": d.get("protein", 0.0),
        "Carbohydrates (g)": d.get("carbohydrates", 0.0),
        "Fats (g)": d.get("fats", 0.0),
        "Free Sugar (g)": d.get("free_sugar", 0.0),
        "Fibre (g)": d.get("fibre", 0.0),
        "Sodium (mg)": d.get("sodium", 0.0),
        "Calcium (mg)": d.get("calcium", 0.0),
        "Iron (mg)": d.get("iron", 0.0),
        "Vitamin C (mg)": d.get("vitamin_c", 0.0),
        "Folate (µg)": d.get("folate", 0.0)
    }
    return d


def add_calorie_in_record(
    user_id,
    dish_name,
    quantity,
    meal_type="other",
    nutrition=None,
    notes="",
    recorded_date=None,
    recorded_time=None
):
    now = datetime.now()
    if recorded_date is None:
        recorded_date = now.strftime("%Y-%m-%d")
    if recorded_time is None:
        recorded_time = now.strftime("%H:%M:%S")

    nutrition = nutrition or {}
    calories = float(nutrition.get("Calories (kcal)", nutrition.get("calories", 0.0)) or 0.0)
    carbs = float(nutrition.get("Carbohydrates (g)", nutrition.get("carbohydrates", 0.0)) or 0.0)
    protein = float(nutrition.get("Protein (g)", nutrition.get("protein", 0.0)) or 0.0)
    fats = float(nutrition.get("Fats (g)", nutrition.get("fats", 0.0)) or 0.0)
    free_sugar = float(nutrition.get("Free Sugar (g)", nutrition.get("free_sugar", 0.0)) or 0.0)
    fibre = float(nutrition.get("Fibre (g)", nutrition.get("fibre", 0.0)) or 0.0)
    sodium = float(nutrition.get("Sodium (mg)", nutrition.get("sodium", 0.0)) or 0.0)
    calcium = float(nutrition.get("Calcium (mg)", nutrition.get("calcium", 0.0)) or 0.0)
    iron = float(nutrition.get("Iron (mg)", nutrition.get("iron", 0.0)) or 0.0)
    vitamin_c = float(nutrition.get("Vitamin C (mg)", nutrition.get("vitamin_c", 0.0)) or 0.0)
    folate = float(nutrition.get("Folate (µg)", nutrition.get("folate", 0.0)) or 0.0)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO calorie_in_records (
            user_id, dish_name, quantity, meal_type,
            calories, carbohydrates, protein, fats,
            free_sugar, fibre, sodium, calcium, iron,
            vitamin_c, folate, notes, recorded_date, recorded_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, dish_name.strip(), float(quantity), meal_type.lower(),
        calories, carbs, protein, fats,
        free_sugar, fibre, sodium, calcium, iron,
        vitamin_c, folate, (notes or "").strip(), recorded_date, recorded_time
    ))
    record_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return record_id


def get_calorie_in_by_date(user_id, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM calorie_in_records
        WHERE user_id = ? AND recorded_date = ?
        ORDER BY created_at ASC
    """, (user_id, date_str))
    rows = cursor.fetchall()
    connection.close()
    return [format_calorie_in_row(r) for r in rows]


def get_calorie_in_total(user_id, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT
            COALESCE(SUM(calories), 0.0) as total_calories,
            COALESCE(SUM(protein), 0.0) as total_protein,
            COALESCE(SUM(carbohydrates), 0.0) as total_carbs,
            COALESCE(SUM(fats), 0.0) as total_fats
        FROM calorie_in_records
        WHERE user_id = ? AND recorded_date = ?
    """, (user_id, date_str))
    row = cursor.fetchone()
    connection.close()

    return {
        "Calories (kcal)": round(float(row["total_calories"]), 2),
        "Protein (g)": round(float(row["total_protein"]), 2),
        "Carbohydrates (g)": round(float(row["total_carbs"]), 2),
        "Fats (g)": round(float(row["total_fats"]), 2)
    }


def get_calorie_in_range(user_id, start_date, end_date):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM calorie_in_records
        WHERE user_id = ? AND recorded_date >= ? AND recorded_date <= ?
        ORDER BY recorded_date ASC, created_at ASC
    """, (user_id, start_date, end_date))
    rows = cursor.fetchall()
    connection.close()
    return [format_calorie_in_row(r) for r in rows]


def get_all_calorie_in_records(user_id):
    """
    Returns all food records for a user ordered chronologically.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM calorie_in_records
        WHERE user_id = ?
        ORDER BY recorded_date ASC, recorded_time ASC, created_at ASC
    """, (user_id,))
    rows = cursor.fetchall()
    connection.close()
    return [format_calorie_in_row(r) for r in rows]


def delete_calorie_in_record(user_id, record_id):
    """
    Safely deletes a food intake record belonging strictly to the authenticated user.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM calorie_in_records
        WHERE id = ? AND user_id = ?
    """, (record_id, user_id))
    affected = cursor.rowcount
    connection.commit()
    connection.close()
    return affected > 0



# =========================================================
# CALORIE OUT (EXERCISE) QUERIES
# =========================================================

def add_calorie_out_record(
    user_id,
    activity,
    met,
    weight_kg,
    duration_minutes,
    calories_burned,
    recorded_date=None
):
    if recorded_date is None:
        recorded_date = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO calorie_out_records (
            user_id, activity, met, weight_kg,
            duration_minutes, calories_burned, recorded_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        activity,
        float(met),
        float(weight_kg),
        float(duration_minutes),
        float(calories_burned),
        recorded_date
    ))
    record_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return record_id


def get_calorie_out_by_date(user_id, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM calorie_out_records
        WHERE user_id = ? AND recorded_date = ?
        ORDER BY created_at DESC
    """, (user_id, date_str))
    rows = cursor.fetchall()
    connection.close()
    return [dict(r) for r in rows]


def get_calorie_out_total(user_id, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT COALESCE(SUM(calories_burned), 0.0)
        FROM calorie_out_records
        WHERE user_id = ? AND recorded_date = ?
    """, (user_id, date_str))
    total = cursor.fetchone()[0]
    connection.close()
    return round(float(total), 2)


def get_calorie_out_range(user_id, start_date, end_date):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM calorie_out_records
        WHERE user_id = ? AND recorded_date >= ? AND recorded_date <= ?
        ORDER BY recorded_date ASC, created_at ASC
    """, (user_id, start_date, end_date))
    rows = cursor.fetchall()
    connection.close()
    return [dict(r) for r in rows]


def delete_calorie_out_record(user_id, record_id):
    """
    Safely deletes an exercise record belonging strictly to the authenticated user.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM calorie_out_records
        WHERE id = ? AND user_id = ?
    """, (record_id, user_id))
    affected = cursor.rowcount
    connection.commit()
    connection.close()
    return affected > 0



# =========================================================
# HISTORICAL DAILY SUMMARY QUERIES
# =========================================================

def get_user_daily_summary(user_id, date_str):
    """
    Returns total calories in, calories out, and net calories for a specific date.
    """
    food_totals = get_calorie_in_total(user_id, date_str)
    c_in = food_totals["Calories (kcal)"]
    c_out = get_calorie_out_total(user_id, date_str)
    net = round(c_in - c_out, 2)
    return {
        "date": date_str,
        "calories_in": c_in,
        "calories_out": c_out,
        "net_calories": net,
        "food_totals": food_totals
    }


def get_user_history_daily(user_id, start_date=None, end_date=None, order="DESC"):
    """
    Returns daily breakdown (date, in, out, net, weight) for a user within an optional date range.
    """
    connection = get_connection()
    cursor = connection.cursor()

    date_filter = ""
    params_union = [user_id, user_id, user_id]
    if start_date and end_date:
        date_filter = "AND recorded_date BETWEEN ? AND ?"
        params_union = [user_id, start_date, end_date, user_id, start_date, end_date, user_id, start_date, end_date]

    direction = "ASC" if str(order).upper() == "ASC" else "DESC"

    query = f"""
        SELECT
            dates.d as recorded_date,
            COALESCE(cin.total_in, 0.0) as calories_in,
            COALESCE(cout.total_out, 0.0) as calories_out,
            w.weight_kg as weight_kg
        FROM (
            SELECT DISTINCT recorded_date as d FROM calorie_in_records WHERE user_id = ? {date_filter}
            UNION
            SELECT DISTINCT recorded_date as d FROM calorie_out_records WHERE user_id = ? {date_filter}
            UNION
            SELECT DISTINCT recorded_date as d FROM weight_records WHERE user_id = ? {date_filter}
        ) dates
        LEFT JOIN (
            SELECT recorded_date, SUM(calories) as total_in
            FROM calorie_in_records
            WHERE user_id = ?
            GROUP BY recorded_date
        ) cin ON dates.d = cin.recorded_date
        LEFT JOIN (
            SELECT recorded_date, SUM(calories_burned) as total_out
            FROM calorie_out_records
            WHERE user_id = ?
            GROUP BY recorded_date
        ) cout ON dates.d = cout.recorded_date
        LEFT JOIN (
            SELECT recorded_date, weight_kg
            FROM (
                SELECT recorded_date, weight_kg,
                       ROW_NUMBER() OVER (PARTITION BY recorded_date ORDER BY created_at DESC, id DESC) as rn
                FROM weight_records
                WHERE user_id = ?
            )
            WHERE rn = 1
        ) w ON dates.d = w.recorded_date
        ORDER BY dates.d {direction}
    """

    all_params = params_union + [user_id, user_id, user_id]
    cursor.execute(query, tuple(all_params))
    rows = cursor.fetchall()
    connection.close()

    history = []
    for r in rows:
        c_in = round(float(r["calories_in"]), 2)
        c_out = round(float(r["calories_out"]), 2)
        w_kg = round(float(r["weight_kg"]), 2) if r["weight_kg"] is not None else None
        history.append({
            "date": r["recorded_date"],
            "calories_in": c_in,
            "calories_out": c_out,
            "net_calories": round(c_in - c_out, 2),
            "weight_kg": w_kg
        })
    return history


# =========================================================
# PRESERVED LEGACY EXERCISE FUNCTIONS
# (Kept for backward compatibility)
# =========================================================

def add_exercise_record(
    activity,
    met,
    weight_kg,
    duration_minutes,
    calories_burned
):
    recorded_date = datetime.now().strftime("%Y-%m-%d")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO exercise_records (
            activity, met, weight_kg, duration_minutes,
            calories_burned, recorded_date
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        activity,
        met,
        weight_kg,
        duration_minutes,
        calories_burned,
        recorded_date
    ))
    connection.commit()
    connection.close()


def get_exercise_by_date(date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM exercise_records
        WHERE recorded_date = ?
        ORDER BY created_at DESC
    """, (date_str,))
    records = cursor.fetchall()
    connection.close()
    return records


def get_exercise_total(date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT COALESCE(SUM(calories_burned), 0)
        FROM exercise_records
        WHERE recorded_date = ?
    """, (date_str,))
    total = cursor.fetchone()[0]
    connection.close()
    return round(float(total), 2)