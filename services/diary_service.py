import json
import csv
from datetime import datetime
from pathlib import Path

from config import (
    DIARY_FILE,
    DIARY_CSV,
    UPLOAD_FOLDER
)

from services.nutrition_service import (
    get_nutrition,
    scale_custom_nutrition
)

from database import (
    add_calorie_in_record,
    get_calorie_in_by_date,
    get_calorie_in_range,
    get_all_calorie_in_records,
    delete_calorie_in_record
)


# =========================================================
# ADD FOOD INTAKE (CALORIES IN)
# =========================================================

def add_food(
    user_id,
    dish_name,
    weight,
    meal_type="other",
    notes="",
    custom_nutrition_per_100g=None
):
    """
    Validates food intake, computes nutrition values, and saves
    the entry into the calorie_in_records database table attached to user_id.
    """
    if not user_id:
        return {
            "success": False,
            "error": "Authentication required. user_id missing."
        }

    try:
        weight = float(weight)
    except (ValueError, TypeError):
        return {
            "success": False,
            "error": "Invalid weight."
        }

    if weight <= 0:
        return {
            "success": False,
            "error": "Weight must be greater than 0."
        }

    # ---------------------------------------------
    # Custom food nutrition
    # ---------------------------------------------
    if custom_nutrition_per_100g is not None:
        nutrition = scale_custom_nutrition(
            custom_nutrition_per_100g,
            weight
        )
    # ---------------------------------------------
    # Dataset food nutrition
    # ---------------------------------------------
    else:
        nutrition = get_nutrition(
            dish_name,
            weight
        )
        if nutrition is None:
            return {
                "success": False,
                "error": f"Dish '{dish_name}' not found."
            }

    now = datetime.now()
    recorded_date = now.strftime("%Y-%m-%d")
    recorded_time = now.strftime("%H:%M:%S")

    # ---------------------------------------------
    # Persist in SQLite calorie_in_records
    # ---------------------------------------------
    record_id = add_calorie_in_record(
        user_id=user_id,
        dish_name=dish_name.strip(),
        quantity=weight,
        meal_type=meal_type.lower(),
        nutrition=nutrition,
        notes=notes.strip(),
        recorded_date=recorded_date,
        recorded_time=recorded_time
    )

    entry = {
        "id": record_id,
        "user_id": user_id,
        "dish_name": dish_name.strip(),
        "weight": weight,
        "quantity": weight,
        "meal_type": meal_type.lower(),
        "nutrition": nutrition,
        "notes": notes.strip(),
        "date": recorded_date,
        "time": recorded_time
    }

    return {
        "success": True,
        "record_id": record_id,
        "entry": entry,
        "nutrition": nutrition
    }


# =========================================================
# GET DIARY ENTRIES FOR A USER
# =========================================================

def get_diary_by_date_user(user_id, date_str=None):
    """
    Retrieve a user's food diary for a specific date (defaults to today).
    """
    if not user_id:
        return []
    return get_calorie_in_by_date(user_id, date_str)


def get_diary_range_user(user_id, start_date, end_date):
    """
    Retrieve a user's food diary entries between two dates.
    """
    if not user_id:
        return []
    return get_calorie_in_range(user_id, start_date, end_date)


def delete_food_entry(user_id, record_id):
    """
    Delete a user's food diary entry.
    """
    if not user_id:
        return False
    return delete_calorie_in_record(user_id, record_id)


# =========================================================
# CALCULATE TOTALS (IN-MEMORY AGGREGATION)
# =========================================================

def calculate_totals(entries):
    """
    Sums macro nutrients for a list of diary entries.
    """
    totals = {
        "Calories (kcal)": 0.0,
        "Protein (g)": 0.0,
        "Carbohydrates (g)": 0.0,
        "Fats (g)": 0.0
    }

    for entry in entries:
        nutrition = entry.get("nutrition", {})
        totals["Calories (kcal)"] += float(nutrition.get("Calories (kcal)", 0) or 0)
        totals["Protein (g)"] += float(nutrition.get("Protein (g)", 0) or 0)
        totals["Carbohydrates (g)"] += float(nutrition.get("Carbohydrates (g)", 0) or 0)
        totals["Fats (g)"] += float(nutrition.get("Fats (g)", 0) or 0)

    for key in totals:
        totals[key] = round(totals[key], 2)

    return totals


# =========================================================
# GROUP ENTRIES BY DATE
# =========================================================

def group_by_date(entries):
    """
    Groups diary entries by recorded date.
    """
    grouped = {}
    for entry in entries:
        date = entry.get("date") or entry.get("recorded_date") or ""
        if date not in grouped:
            grouped[date] = []
        grouped[date].append(entry)
    return dict(sorted(grouped.items()))


# =========================================================
# USER-SPECIFIC CSV EXPORT
# =========================================================

def export_user_diary_csv(user_id, output_path=None):
    """
    Exports all food records for a specific user to a CSV file.
    """
    records = get_all_calorie_in_records(user_id)
    if not records:
        return None

    if output_path is None:
        output_path = UPLOAD_FOLDER / f"food_diary_user_{user_id}.csv"

    csv_data = []
    for entry in records:
        nutrition = entry.get("nutrition", {})
        csv_data.append({
            "Date": entry.get("date", ""),
            "Time": entry.get("time", ""),
            "Dish": entry.get("dish_name", ""),
            "Weight (g)": entry.get("weight", entry.get("quantity", 0)),
            "Meal Type": entry.get("meal_type", ""),
            "Calories": nutrition.get("Calories (kcal)", 0),
            "Protein (g)": nutrition.get("Protein (g)", 0),
            "Carbs (g)": nutrition.get("Carbohydrates (g)", 0),
            "Fats (g)": nutrition.get("Fats (g)", 0),
            "Notes": entry.get("notes", "")
        })

    fieldnames = list(csv_data[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    return output_path


# =========================================================
# LEGACY FALLBACKS
# =========================================================

def load_diary():
    if not DIARY_FILE.exists():
        return []
    try:
        with open(DIARY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_diary(diary):
    with open(DIARY_FILE, "w", encoding="utf-8") as file:
        json.dump(diary, file, indent=2, ensure_ascii=False)


def diary_to_csv(diary=None):
    if diary is None:
        diary = load_diary()
    if not diary:
        return False
    csv_data = []
    for entry in diary:
        nutrition = entry.get("nutrition", {})
        csv_data.append({
            "Date": entry.get("date", ""),
            "Time": entry.get("time", ""),
            "Dish": entry.get("dish_name", ""),
            "Weight (g)": entry.get("weight", 0),
            "Meal Type": entry.get("meal_type", ""),
            "Calories": nutrition.get("Calories (kcal)", 0),
            "Protein (g)": nutrition.get("Protein (g)", 0),
            "Carbs (g)": nutrition.get("Carbohydrates (g)", 0),
            "Fats (g)": nutrition.get("Fats (g)", 0),
            "Notes": entry.get("notes", "")
        })
    fieldnames = list(csv_data[0].keys())
    with open(DIARY_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    return True