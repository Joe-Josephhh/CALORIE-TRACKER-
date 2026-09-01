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

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exercise_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            activity TEXT NOT NULL,

            met REAL NOT NULL,

            weight_kg REAL NOT NULL,

            duration_minutes REAL NOT NULL,

            calories_burned REAL NOT NULL,

            recorded_date TEXT NOT NULL,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


# =========================================================
# ADD EXERCISE RECORD
# =========================================================

def add_exercise_record(
    activity,
    met,
    weight_kg,
    duration_minutes,
    calories_burned
):

    recorded_date = datetime.now().strftime(
        "%Y-%m-%d"
    )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO exercise_records
        (
            activity,
            met,
            weight_kg,
            duration_minutes,
            calories_burned,
            recorded_date
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


# =========================================================
# GET EXERCISES FOR A DATE
# =========================================================

def get_exercise_by_date(
    date_str=None
):

    if date_str is None:

        date_str = datetime.now().strftime(
            "%Y-%m-%d"
        )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM exercise_records

        WHERE recorded_date = ?

        ORDER BY created_at DESC
    """, (
        date_str,
    ))

    records = cursor.fetchall()

    connection.close()

    return records


# =========================================================
# GET TOTAL CALORIES BURNED
# =========================================================

def get_exercise_total(
    date_str=None
):

    if date_str is None:

        date_str = datetime.now().strftime(
            "%Y-%m-%d"
        )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COALESCE(
            SUM(calories_burned),
            0
        )

        FROM exercise_records

        WHERE recorded_date = ?
    """, (
        date_str,
    ))

    total = cursor.fetchone()[0]

    connection.close()

    return round(
        float(total),
        2
    )