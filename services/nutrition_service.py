import pandas as pd
from config import NUTRITION_CSV, GENERAL_FOODS_CSV


# ---------------------------------------------------------
# Load nutrition database
# ---------------------------------------------------------

def load_nutrition_database():

    # Preserve compatibility with your existing CSV,
    # which may contain unnecessary quotation marks.
    with open(
        NUTRITION_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        raw_data = file.read().replace('"', '')

    from io import StringIO

    df = pd.read_csv(
        StringIO(raw_data)
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    # Find dish-name column
    dish_col = None

    for column in df.columns:

        if "dish" in column.lower():

            dish_col = column
            break

    if dish_col is None:

        dish_col = df.columns[0]

    df[dish_col] = (
        df[dish_col]
        .astype(str)
        .str.strip()
    )

    return df, dish_col


def load_general_foods_database():
    if not GENERAL_FOODS_CSV.exists():
        return pd.DataFrame(), "Dish Name"

    try:
        from io import StringIO
        with open(
            GENERAL_FOODS_CSV,
            "r",
            encoding="utf-8-sig"
        ) as file:
            raw_data = file.read()

        gdf = pd.read_csv(StringIO(raw_data))
        gdf.columns = gdf.columns.str.strip()

        dish_col = None
        for col in gdf.columns:
            if "dish" in col.lower() or "food" in col.lower():
                dish_col = col
                break
        if dish_col is None:
            dish_col = gdf.columns[0]

        gdf[dish_col] = gdf[dish_col].astype(str).str.strip()
        return gdf, dish_col
    except Exception as e:
        print(f"Error loading general foods database: {e}")
        return pd.DataFrame(), "Dish Name"


df, dish_col = load_nutrition_database()
general_df, general_dish_col = load_general_foods_database()

print(
    f"Loaded {len(df)} dishes from nutrition database."
)
print(
    f"Loaded {len(general_df)} dishes from general foods presets database."
)


# ---------------------------------------------------------
# Nutrition fields
# ---------------------------------------------------------

NUTRIENT_COLUMNS = [
    "Calories (kcal)",
    "Protein (g)",
    "Carbohydrates (g)",
    "Fats (g)",
    "Free Sugar (g)",
    "Fibre (g)",
    "Sodium (mg)",
    "Calcium (mg)",
    "Iron (mg)",
    "Vitamin C (mg)",
    "Folate (µg)"
]


# ---------------------------------------------------------
# Get nutrition for weight
# ---------------------------------------------------------

def get_nutrition(
    dish_name,
    weight
):

    if weight <= 0:
        return None

    # Check primary ML nutrition dataset
    matching = df[
        df[dish_col]
        .str.lower()
        == dish_name.strip().lower()
    ]

    # If not found, check general foods database
    if matching.empty and not general_df.empty:
        matching = general_df[
            general_df[general_dish_col]
            .str.lower()
            == dish_name.strip().lower()
        ]

    if matching.empty:
        return None

    row = matching.iloc[0]

    scale_factor = weight / 100

    nutrition = {}

    for column in NUTRIENT_COLUMNS:

        value = row.get(
            column,
            0
        )

        try:
            value = float(value)
        except (
            ValueError,
            TypeError
        ):
            value = 0.0

        nutrition[column] = round(
            value * scale_factor,
            2
        )

    return nutrition


# ---------------------------------------------------------
# Get nutrition per 100g
# ---------------------------------------------------------

def get_nutrition_per_100g(
    dish_name
):

    matching = df[
        df[dish_col]
        .str.lower()
        == dish_name.strip().lower()
    ]

    if matching.empty and not general_df.empty:
        matching = general_df[
            general_df[general_dish_col]
            .str.lower()
            == dish_name.strip().lower()
        ]

    if matching.empty:
        return None

    row = matching.iloc[0]

    nutrition = {}

    for column in NUTRIENT_COLUMNS:

        value = row.get(
            column,
            0
        )

        try:
            value = float(value)
        except (
            ValueError,
            TypeError
        ):
            value = 0.0

        nutrition[column] = value

    return nutrition


# ---------------------------------------------------------
# Search foods
# ---------------------------------------------------------

def search_foods(
    search_term=""
):

    search_term = (
        search_term
        .strip()
        .lower()
    )

    if not search_term:
        return get_all_foods()

    matches = df[
        df[dish_col]
        .str.lower()
        .str.contains(
            search_term,
            na=False
        )
    ][dish_col].tolist()

    if not general_df.empty:
        gen_matches = general_df[
            general_df[general_dish_col]
            .str.lower()
            .str.contains(
                search_term,
                na=False
            )
        ][general_dish_col].tolist()

        for item in gen_matches:
            if item not in matches:
                matches.append(item)

    return matches


# ---------------------------------------------------------
# All foods
# ---------------------------------------------------------

def get_all_foods():

    foods = df[dish_col].tolist()

    if not general_df.empty:
        for item in general_df[general_dish_col].tolist():
            if item not in foods:
                foods.append(item)

    return foods


# ---------------------------------------------------------
# Custom nutrition scaling
# ---------------------------------------------------------

def scale_custom_nutrition(
    nutrition_per_100g,
    weight
):

    scale_factor = weight / 100

    nutrition = {}

    for key, value in nutrition_per_100g.items():

        nutrition[key] = round(
            float(value) * scale_factor,
            2
        )

    return nutrition


# ---------------------------------------------------------
# General foods presets for Custom Food area
# ---------------------------------------------------------

def get_general_foods_presets():
    """
    Returns list of predefined common foods with macros per 100g,
    ideal for instant auto-filling and custom food suggestions.
    """
    if general_df.empty:
        return []

    presets = []
    for _, row in general_df.iterrows():
        name = str(row[general_dish_col]).strip()
        try:
            calories = float(row.get("Calories (kcal)", 0))
        except (ValueError, TypeError):
            calories = 0.0

        try:
            protein = float(row.get("Protein (g)", 0))
        except (ValueError, TypeError):
            protein = 0.0

        try:
            carbohydrates = float(row.get("Carbohydrates (g)", 0))
        except (ValueError, TypeError):
            carbohydrates = 0.0

        try:
            fats = float(row.get("Fats (g)", 0))
        except (ValueError, TypeError):
            fats = 0.0

        try:
            fibre = float(row.get("Fibre (g)", 0))
        except (ValueError, TypeError):
            fibre = 0.0

        item = {
            "name": name,
            "calories": calories,
            "protein": protein,
            "carbohydrates": carbohydrates,
            "fats": fats,
            "fibre": fibre,
            "free_sugar": 0.0,
            "sodium": 0.0,
            "calcium": 0.0,
            "iron": 0.0,
            "vitamin_c": 0.0,
            "folate": 0.0
        }
        presets.append(item)

    return presets