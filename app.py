from pathlib import Path
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file
)

from flask import Flask, render_template, request, jsonify

from werkzeug.utils import secure_filename


from config import (
    SECRET_KEY,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    PRESET_WEIGHTS,
    MEAL_TYPES,
    DIARY_CSV
)

from services.calorie_burn_service import (
    get_activity_categories,
    get_activities_by_category,
    search_activities,
    calculate_calories_for_activity
)

from services.prediction_service import (
    predict_food
)

from services.nutrition_service import (
    search_foods,
    get_all_foods,
    get_nutrition
)

from services.diary_service import (
    add_food,
    get_diary_by_date,
    get_diary_range,
    calculate_totals,
    group_by_date,
    load_diary,
    diary_to_csv
)

from database import (
    init_database,
    add_exercise_record,
    get_exercise_by_date,
    get_exercise_total
)

# =========================================================
# Flask application
# =========================================================

app = Flask(__name__)

app.secret_key = SECRET_KEY

init_database()

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_CONTENT_LENGTH

app.config[
    "UPLOAD_FOLDER"
] = str(UPLOAD_FOLDER)


# =========================================================
# Helper
# =========================================================

def allowed_file(filename):

    if "." not in filename:

        return False

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    return extension in ALLOWED_EXTENSIONS


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# =========================================================
# FOOD SCAN PAGE
# =========================================================

@app.route("/food-scan")
def food_scan():

    return render_template(
        "food_scan.html"
    )


# =========================================================
# PREDICT IMAGE
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        flash(
            "Please select an image.",
            "error"
        )

        return redirect(
            url_for("food_scan")
        )

    image = request.files["image"]

    if image.filename == "":

        flash(
            "No image selected.",
            "error"
        )

        return redirect(
            url_for("food_scan")
        )

    if not allowed_file(
        image.filename
    ):

        flash(
            "Unsupported image format.",
            "error"
        )

        return redirect(
            url_for("food_scan")
        )

    filename = secure_filename(
        image.filename
    )

    # Prevent filename collisions
    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    filename = (
        timestamp
        + "_"
        + filename
    )

    image_path = (
        UPLOAD_FOLDER
        / filename
    )

    try:

        image.save(
            image_path
        )

        predictions = predict_food(
            image_path
        )

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        if image_path.exists():

            image_path.unlink()

        flash(
            "Unable to process the image. "
            "Please try another image.",
            "error"
        )

        return redirect(
            url_for("food_scan")
        )

    return render_template(
        "prediction_result.html",
        predictions=predictions,
        image_filename=filename
    )


# =========================================================
# LOG FOOD PAGE
# =========================================================

@app.route(
    "/log-food",
    methods=["GET", "POST"]
)
def log_food():

    if request.method == "GET":

        selected_food = request.args.get(
            "food",
            ""
        )

        return render_template(
            "log_food.html",
            selected_food=selected_food,
            preset_weights=PRESET_WEIGHTS,
            meal_types=MEAL_TYPES
        )

    # ---------------------------------------------
    # POST
    # ---------------------------------------------

    dish_name = request.form.get(
        "dish_name",
        ""
    ).strip()

    weight = request.form.get(
        "weight",
        ""
    ).strip()

    meal_type = request.form.get(
        "meal_type",
        "other"
    ).strip().lower()

    notes = request.form.get(
        "notes",
        ""
    ).strip()

    is_custom = (
        request.form.get(
            "is_custom"
        )
        == "yes"
    )

    # ---------------------------------------------
    # Validate
    # ---------------------------------------------

    if not dish_name:

        flash(
            "Food name is required.",
            "error"
        )

        return redirect(
            url_for("log_food")
        )

    try:

        weight_value = float(
            weight
        )

    except ValueError:

        flash(
            "Invalid weight.",
            "error"
        )

        return redirect(
            url_for(
                "log_food",
                food=dish_name
            )
        )

    if weight_value <= 0:

        flash(
            "Weight must be greater than 0.",
            "error"
        )

        return redirect(
            url_for(
                "log_food",
                food=dish_name
            )
        )

    if meal_type not in MEAL_TYPES:

        meal_type = "other"

    # ---------------------------------------------
    # Custom nutrition
    # ---------------------------------------------

    custom_nutrition = None

    if is_custom:

        nutrition_fields = {

            "Calories (kcal)":
                "calories",

            "Protein (g)":
                "protein",

            "Carbohydrates (g)":
                "carbohydrates",

            "Fats (g)":
                "fats",

            "Free Sugar (g)":
                "free_sugar",

            "Fibre (g)":
                "fibre",

            "Sodium (mg)":
                "sodium",

            "Calcium (mg)":
                "calcium",

            "Iron (mg)":
                "iron",

            "Vitamin C (mg)":
                "vitamin_c",

            "Folate (µg)":
                "folate"
        }

        custom_nutrition = {}

        try:

            for nutrient, form_name in (
                nutrition_fields.items()
            ):

                raw_value = request.form.get(
                    form_name,
                    "0"
                ).strip()

                if raw_value == "":

                    raw_value = "0"

                custom_nutrition[
                    nutrient
                ] = float(
                    raw_value
                )

        except ValueError:

            flash(
                "Custom nutrition values must be numbers.",
                "error"
            )

            return redirect(
                url_for(
                    "log_food",
                    food=dish_name
                )
            )

    # ---------------------------------------------
    # Save food
    # ---------------------------------------------

    result = add_food(
        dish_name=dish_name,
        weight=weight_value,
        meal_type=meal_type,
        notes=notes,
        custom_nutrition_per_100g=custom_nutrition
    )

    if not result["success"]:

        flash(
            result["error"],
            "error"
        )

        return redirect(
            url_for(
                "log_food",
                food=dish_name
            )
        )

    return render_template(
        "diary.html",
        entries=get_diary_by_date(),
        totals=calculate_totals(
            get_diary_by_date()
        ),
        date=datetime.now().strftime(
            "%Y-%m-%d"
        ),
        success_message=(
            f"{dish_name} added successfully."
        )
    )


# =========================================================
# DIARY - TODAY
# =========================================================

@app.route("/diary")
def diary():

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    entries = get_diary_by_date(
        today
    )

    totals = calculate_totals(
        entries
    )

    return render_template(
        "diary.html",
        entries=entries,
        totals=totals,
        date=today
    )


# =========================================================
# DIARY - SPECIFIC DATE
# =========================================================

@app.route(
    "/diary/<date>"
)
def diary_date(date):

    try:

        datetime.strptime(
            date,
            "%Y-%m-%d"
        )

    except ValueError:

        flash(
            "Invalid date. Use YYYY-MM-DD.",
            "error"
        )

        return redirect(
            url_for("diary")
        )

    entries = get_diary_by_date(
        date
    )

    totals = calculate_totals(
        entries
    )

    return render_template(
        "diary.html",
        entries=entries,
        totals=totals,
        date=date
    )


# =========================================================
# HISTORY / DATE RANGE
# =========================================================

@app.route(
    "/history",
    methods=["GET", "POST"]
)
def history():

    if request.method == "GET":

        return render_template(
            "history.html",
            entries=None,
            grouped=None,
            totals=None
        )

    start_date = request.form.get(
        "start_date",
        ""
    )

    end_date = request.form.get(
        "end_date",
        ""
    )

    try:

        start = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )

        end = datetime.strptime(
            end_date,
            "%Y-%m-%d"
        )

    except ValueError:

        flash(
            "Please enter valid dates.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    if start > end:

        flash(
            "Start date must be before end date.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    entries = get_diary_range(
        start_date,
        end_date
    )

    grouped = group_by_date(
        entries
    )

    totals = calculate_totals(
        entries
    )

    return render_template(
        "history.html",
        entries=entries,
        grouped=grouped,
        totals=totals,
        start_date=start_date,
        end_date=end_date
    )


# =========================================================
# FOOD SEARCH
# =========================================================

@app.route("/foods")
def foods():

    query = request.args.get(
        "q",
        ""
    ).strip()

    matching_foods = search_foods(
        query
    )

    return render_template(
        "foods.html",
        foods=matching_foods,
        query=query
    )


# =========================================================
# API FOOD SEARCH
# =========================================================

@app.route("/api/foods")
def api_foods():

    query = request.args.get(
        "q",
        ""
    ).strip()

    foods = search_foods(
        query
    )

    return {
        "foods": foods
    }


# =========================================================
# NUTRITION API
# =========================================================

@app.route(
    "/api/nutrition",
    methods=["POST"]
)
def api_nutrition():

    data = request.get_json(
        silent=True
    )

    if not data:

        return {
            "success": False,
            "error": "Invalid JSON."
        }, 400

    dish_name = data.get(
        "dish_name",
        ""
    )

    weight = data.get(
        "weight",
        0
    )

    try:

        weight = float(
            weight
        )

    except (
        ValueError,
        TypeError
    ):

        return {
            "success": False,
            "error": "Invalid weight."
        }, 400

    nutrition = get_nutrition(
        dish_name,
        weight
    )

    if nutrition is None:

        return {
            "success": False,
            "error": "Food not found."
        }, 404

    return {
        "success": True,
        "food": dish_name,
        "weight": weight,
        "nutrition": nutrition
    }


# =========================================================
# EXPORT DIARY
# =========================================================

@app.route(
    "/export"
)
def export_diary():

    diary = load_diary()

    if not diary:

        flash(
            "There is no diary data to export.",
            "error"
        )

        return redirect(
            url_for("diary")
        )

    diary_to_csv(
        diary
    )

    return send_file(
        DIARY_CSV,
        as_attachment=True,
        download_name="food_diary.csv"
    )


# =========================================================
# ERROR - FILE TOO LARGE
# =========================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    flash(
        "Image is too large. Maximum size is 10 MB.",
        "error"
    )

    return redirect(
        url_for("food_scan")
    )

# =========================================================
# CALORIE OUT
# =========================================================

@app.route("/calorie-out", methods=["GET"])
def calorie_out():
    """
    Display the calorie-out page.
    """

    categories = get_activity_categories()

    return render_template(
        "calorie_out.html",
        categories=categories
    )


@app.route("/api/activities/<path:category>", methods=["GET"])
def activities_by_category(category):
    """
    Return activities belonging to a selected category.
    """

    activities = get_activities_by_category(category)

    return jsonify({
        "success": True,
        "activities": activities
    })


@app.route("/api/search-activities", methods=["GET"])
def search_activity_api():
    """
    Search the MET database.
    """

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "success": True,
            "activities": []
        })

    activities = search_activities(query)

    return jsonify({
        "success": True,
        "activities": activities[:50]
    })


@app.route("/api/calculate-calories", methods=["POST"])
def calculate_calories_api():
    """
    Calculate calories burned for an activity.
    """

    try:

        data = request.get_json()

        activity_code = data.get("activity_code")
        weight_kg = data.get("weight_kg")
        duration_minutes = data.get("duration_minutes")

        if not activity_code:
            return jsonify({
                "success": False,
                "error": "Please select an activity."
            }), 400

        if weight_kg is None or duration_minutes is None:
            return jsonify({
                "success": False,
                "error": "Weight and duration are required."
            }), 400

        result = calculate_calories_for_activity(
            activity_code=activity_code,
            weight_kg=weight_kg,
            duration_minutes=duration_minutes
        )
        add_exercise_record(
            activity=result["description"],
            met=result["met"],
            weight_kg=result["weight_kg"],
            duration_minutes=result["duration_minutes"],
            calories_burned=result["calories_burned"]
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except ValueError as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:

        print("Calorie calculation error:", e)

        return jsonify({
            "success": False,
            "error": "An unexpected error occurred."
        }), 500

@app.route("/dashboard")
def dashboard():

    today = datetime.now().strftime("%Y-%m-%d")

    # -----------------------------------------
    # CALORIES IN
    # -----------------------------------------

    food_entries = get_diary_by_date(today)

    food_totals = calculate_totals(food_entries)

    calories_in = food_totals.get(
        "Calories (kcal)",
        0
    )


    # -----------------------------------------
    # CALORIES OUT
    # -----------------------------------------

    exercise_entries = get_exercise_by_date(today)

    calories_out = get_exercise_total(today)


    # -----------------------------------------
    # NET CALORIES
    # -----------------------------------------

    net_calories = calories_in - calories_out


    return render_template(
        "dashboard.html",

        date=today,

        calories_in=round(
            calories_in,
            2
        ),

        calories_out=round(
            calories_out,
            2
        ),

        net_calories=round(
            net_calories,
            2
        ),

        food_entries=food_entries,

        exercise_entries=exercise_entries,

        food_totals=food_totals
    )

# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )