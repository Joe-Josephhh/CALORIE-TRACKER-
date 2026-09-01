from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MET_FILE = BASE_DIR / "data" / "met.csv"


# ---------------------------------------------------------
# Load MET database
# ---------------------------------------------------------

def load_met_data():
    """
    Load the MET activity database from the CSV file.
    """

    if not MET_FILE.exists():
        raise FileNotFoundError(
            f"MET database not found at: {MET_FILE}"
        )

    df = pd.read_csv(MET_FILE)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Your uploaded dataset uses these columns
    required_columns = ["Actvitiy", "Code", "MET", "Description"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in MET CSV: {missing_columns}"
        )

    # Clean text columns
    df["Actvitiy"] = df["Actvitiy"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    # Convert MET values to numbers
    df["MET"] = pd.to_numeric(df["MET"], errors="coerce")

    # Remove invalid MET rows
    df = df.dropna(subset=["MET"])

    return df


# Load once when Flask starts
met_data = load_met_data()

print(f"Loaded {len(met_data)} activities from MET database.")


# ---------------------------------------------------------
# Get activity categories
# ---------------------------------------------------------

def get_activity_categories():
    """
    Return all unique activity categories.
    """

    categories = (
        met_data["Actvitiy"]
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(categories)


# ---------------------------------------------------------
# Get activities belonging to a category
# ---------------------------------------------------------

def get_activities_by_category(category):
    """
    Return activities belonging to a selected category.
    """

    if not category:
        return []

    filtered = met_data[
        met_data["Actvitiy"].str.lower() == category.strip().lower()
    ]

    activities = []

    for _, row in filtered.iterrows():
        activities.append({
            "code": str(row["Code"]),
            "description": row["Description"],
            "met": float(row["MET"])
        })

    return activities


# ---------------------------------------------------------
# Search activities
# ---------------------------------------------------------

def search_activities(search_text):
    """
    Search the MET database using activity category
    or activity description.
    """

    if not search_text:
        return []

    search_text = search_text.strip().lower()

    mask = (
        met_data["Actvitiy"].str.lower().str.contains(
            search_text,
            na=False
        )
        |
        met_data["Description"].str.lower().str.contains(
            search_text,
            na=False
        )
    )

    filtered = met_data[mask]

    activities = []

    for _, row in filtered.iterrows():
        activities.append({
            "category": row["Actvitiy"],
            "code": str(row["Code"]),
            "description": row["Description"],
            "met": float(row["MET"])
        })

    return activities


# ---------------------------------------------------------
# Find activity by code
# ---------------------------------------------------------

def get_activity_by_code(code):
    """
    Find one activity using its Code.
    """

    if code is None:
        return None

    filtered = met_data[
        met_data["Code"].astype(str).str.strip()
        == str(code).strip()
    ]

    if filtered.empty:
        return None

    row = filtered.iloc[0]

    return {
        "category": row["Actvitiy"],
        "code": str(row["Code"]),
        "description": row["Description"],
        "met": float(row["MET"])
    }


# ---------------------------------------------------------
# Calculate calories burned
# ---------------------------------------------------------

def calculate_calories_burned(weight_kg, duration_minutes, met):
    """
    Calculate calories burned using the MET formula.

    Formula:

    Calories burned =
        MET × 3.5 × weight(kg) / 200 × duration(minutes)
    """

    try:
        weight_kg = float(weight_kg)
        duration_minutes = float(duration_minutes)
        met = float(met)
    except (TypeError, ValueError):
        raise ValueError(
            "Weight, duration and MET must be numbers."
        )

    if weight_kg <= 0:
        raise ValueError(
            "Weight must be greater than 0."
        )

    if duration_minutes <= 0:
        raise ValueError(
            "Duration must be greater than 0."
        )

    if met <= 0:
        raise ValueError(
            "MET value must be greater than 0."
        )

    calories = (
        met
        * 3.5
        * weight_kg
        / 200
        * duration_minutes
    )

    return round(calories, 2)


# ---------------------------------------------------------
# Calculate using activity code
# ---------------------------------------------------------

def calculate_calories_for_activity(
    activity_code,
    weight_kg,
    duration_minutes
):
    """
    Find the MET value using the activity code
    and calculate calories burned.
    """

    activity = get_activity_by_code(activity_code)

    if activity is None:
        raise ValueError(
            f"Activity with code '{activity_code}' was not found."
        )

    calories = calculate_calories_burned(
        weight_kg=weight_kg,
        duration_minutes=duration_minutes,
        met=activity["met"]
    )

    return {
        "category": activity["category"],
        "code": activity["code"],
        "description": activity["description"],
        "met": activity["met"],
        "weight_kg": float(weight_kg),
        "duration_minutes": float(duration_minutes),
        "calories_burned": calories
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nMET database test")
    print("-----------------")

    categories = get_activity_categories()

    print(f"Number of categories: {len(categories)}")

    print("\nFirst 10 categories:")

    for category in categories[:10]:
        print("-", category)

    print("\nSearching for 'walking'...")

    results = search_activities("walking")

    for activity in results[:5]:
        print(
            activity["description"],
            "→ MET:",
            activity["met"]
        )

    if results:

        activity_code = results[0]["code"]

        result = calculate_calories_for_activity(
            activity_code=activity_code,
            weight_kg=65,
            duration_minutes=30
        )

        print("\nExample calculation")
        print("-------------------")
        print("Activity:", result["description"])
        print("MET:", result["met"])
        print("Weight:", result["weight_kg"], "kg")
        print("Duration:", result["duration_minutes"], "minutes")
        print("Calories burned:", result["calories_burned"], "kcal")