import pandas as pd

from config import NUTRITION_CSV


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


df, dish_col = load_nutrition_database()

print(
    f"Loaded {len(df)} dishes from nutrition database."
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

    matching = df[
        df[dish_col]
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

        return (
            df[dish_col]
            .tolist()
        )

    matches = df[
        df[dish_col]
        .str.lower()
        .str.contains(
            search_term,
            na=False
        )
    ]

    return (
        matches[dish_col]
        .tolist()
    )


# ---------------------------------------------------------
# All foods
# ---------------------------------------------------------

def get_all_foods():

    return (
        df[dish_col]
        .tolist()
    )


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