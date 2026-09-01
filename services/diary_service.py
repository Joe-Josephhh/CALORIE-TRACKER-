import json
import csv

from datetime import datetime

from config import (
    DIARY_FILE,
    DIARY_CSV
)

from services.nutrition_service import (
    get_nutrition,
    scale_custom_nutrition
)


# ---------------------------------------------------------
# Load diary
# ---------------------------------------------------------

def load_diary():

    if not DIARY_FILE.exists():

        return []

    try:

        with open(
            DIARY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


# ---------------------------------------------------------
# Save diary
# ---------------------------------------------------------

def save_diary(diary):

    with open(
        DIARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            diary,
            file,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# Export diary to CSV
# ---------------------------------------------------------

def diary_to_csv(
    diary=None
):

    if diary is None:

        diary = load_diary()

    if not diary:

        return False

    csv_data = []

    for entry in diary:

        nutrition = entry.get(
            "nutrition",
            {}
        )

        csv_data.append({
            "Date":
                entry.get("date", ""),

            "Time":
                entry.get("time", ""),

            "Dish":
                entry.get("dish_name", ""),

            "Weight (g)":
                entry.get("weight", 0),

            "Meal Type":
                entry.get("meal_type", ""),

            "Calories":
                nutrition.get(
                    "Calories (kcal)",
                    0
                ),

            "Protein (g)":
                nutrition.get(
                    "Protein (g)",
                    0
                ),

            "Carbs (g)":
                nutrition.get(
                    "Carbohydrates (g)",
                    0
                ),

            "Fats (g)":
                nutrition.get(
                    "Fats (g)",
                    0
                ),

            "Notes":
                entry.get("notes", "")
        })

    fieldnames = list(
        csv_data[0].keys()
    )

    with open(
        DIARY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_data
        )

    return True


# ---------------------------------------------------------
# Add food
# ---------------------------------------------------------

def add_food(
    dish_name,
    weight,
    meal_type="other",
    notes="",
    custom_nutrition_per_100g=None
):

    try:

        weight = float(weight)

    except (
        ValueError,
        TypeError
    ):

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
    # Custom food
    # ---------------------------------------------

    if custom_nutrition_per_100g is not None:

        nutrition = scale_custom_nutrition(
            custom_nutrition_per_100g,
            weight
        )

    # ---------------------------------------------
    # Dataset food
    # ---------------------------------------------

    else:

        nutrition = get_nutrition(
            dish_name,
            weight
        )

        if nutrition is None:

            return {
                "success": False,
                "error":
                    f"Dish '{dish_name}' not found."
            }

    # ---------------------------------------------
    # Create diary entry
    # ---------------------------------------------

    now = datetime.now()

    entry = {

        "date":
            now.strftime("%Y-%m-%d"),

        "time":
            now.strftime("%H:%M:%S"),

        "dish_name":
            dish_name.strip(),

        "weight":
            weight,

        "meal_type":
            meal_type.lower(),

        "nutrition":
            nutrition,

        "notes":
            notes.strip()
    }

    diary = load_diary()

    diary.append(entry)

    save_diary(diary)

    diary_to_csv(diary)

    return {
        "success": True,
        "entry": entry,
        "nutrition": nutrition
    }


# ---------------------------------------------------------
# Get diary by date
# ---------------------------------------------------------

def get_diary_by_date(
    date_str=None
):

    if date_str is None:

        date_str = datetime.now().strftime(
            "%Y-%m-%d"
        )

    diary = load_diary()

    entries = [
        entry
        for entry in diary
        if entry.get("date") == date_str
    ]

    return entries


# ---------------------------------------------------------
# Get diary by date range
# ---------------------------------------------------------

def get_diary_range(
    start_date,
    end_date
):

    diary = load_diary()

    entries = [
        entry
        for entry in diary
        if start_date
        <= entry.get("date", "")
        <= end_date
    ]

    return entries


# ---------------------------------------------------------
# Calculate totals
# ---------------------------------------------------------

def calculate_totals(
    entries
):

    totals = {

        "Calories (kcal)": 0,
        "Protein (g)": 0,
        "Carbohydrates (g)": 0,
        "Fats (g)": 0
    }

    for entry in entries:

        nutrition = entry.get(
            "nutrition",
            {}
        )

        totals[
            "Calories (kcal)"
        ] += float(
            nutrition.get(
                "Calories (kcal)",
                0
            )
        )

        totals[
            "Protein (g)"
        ] += float(
            nutrition.get(
                "Protein (g)",
                0
            )
        )

        totals[
            "Carbohydrates (g)"
        ] += float(
            nutrition.get(
                "Carbohydrates (g)",
                0
            )
        )

        totals[
            "Fats (g)"
        ] += float(
            nutrition.get(
                "Fats (g)",
                0
            )
        )

    for key in totals:

        totals[key] = round(
            totals[key],
            2
        )

    return totals


# ---------------------------------------------------------
# Group entries by date
# ---------------------------------------------------------

def group_by_date(
    entries
):

    grouped = {}

    for entry in entries:

        date = entry.get(
            "date",
            ""
        )

        if date not in grouped:

            grouped[date] = []

        grouped[date].append(
            entry
        )

    return dict(
        sorted(
            grouped.items()
        )
    )