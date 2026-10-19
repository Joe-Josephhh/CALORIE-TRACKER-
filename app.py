from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
    session
)

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from config import (
    SECRET_KEY,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    PRESET_WEIGHTS,
    MEAL_TYPES
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
    get_nutrition,
    get_nutrition_per_100g,
    get_general_foods_presets
)

from services.diary_service import (
    add_food,
    get_diary_by_date_user,
    get_diary_range_user,
    delete_food_entry,
    calculate_totals,
    group_by_date,
    export_user_diary_csv
)

from services.weight_prediction_service import (
    predict_weight_trajectory
)

from services.coach_service import (
    get_daily_calorie_coach
)

from database import (
    init_database,
    create_user,
    get_user_by_id,
    get_user_by_email,
    get_user_by_username,
    create_user_profile,
    get_user_profile,
    update_user_profile,
    add_weight_record,
    get_latest_weight,
    get_starting_weight,
    get_previous_weight,
    get_weight_history,
    get_weight_stats,
    delete_weight_record,
    add_calorie_in_record,
    get_calorie_in_by_date,
    get_calorie_in_total,
    get_calorie_in_range,
    add_calorie_out_record,
    get_calorie_out_by_date,
    get_calorie_out_total,
    get_calorie_out_range,
    delete_calorie_out_record,
    get_user_daily_summary,
    get_user_history_daily
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
# Auth Decorator & Template Context
# =========================================================

def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped_view


@app.context_processor
def inject_user():
    user_id = session.get("user_id")
    if user_id:
        user = get_user_by_id(user_id)
        if user:
            return {"current_user": user, "logged_in": True}
    return {"current_user": None, "logged_in": False}


# =========================================================
# REGISTRATION
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template("register.html", form_data=None)

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "Other").strip()
    height_cm = request.form.get("height_cm", "").strip()
    weight_kg = request.form.get("weight_kg", "").strip()
    goal = request.form.get("goal", "Maintain").strip()
    activity_level = request.form.get("activity_level", "Moderately Active").strip()

    if not username or not email or not password:
        flash("Username, email, and password are required.", "error")
        return render_template("register.html", form_data=request.form)

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return render_template("register.html", form_data=request.form)

    if len(password) < 6:
        flash("Password must be at least 6 characters long.", "error")
        return render_template("register.html", form_data=request.form)

    if get_user_by_email(email):
        flash("An account with that email already exists.", "error")
        return render_template("register.html", form_data=request.form)

    if get_user_by_username(username):
        flash("That username is already taken. Please choose another.", "error")
        return render_template("register.html", form_data=request.form)

    try:
        age_val = int(age)
        if age_val < 5 or age_val > 120:
            raise ValueError()
    except ValueError:
        flash("Please enter a valid age between 5 and 120.", "error")
        return render_template("register.html", form_data=request.form)

    try:
        height_val = float(height_cm)
        if height_val < 50 or height_val > 270:
            raise ValueError()
    except ValueError:
        flash("Please enter a valid height in cm (e.g. 170).", "error")
        return render_template("register.html", form_data=request.form)

    try:
        weight_val = float(weight_kg)
        if weight_val < 20 or weight_val > 400:
            raise ValueError()
    except ValueError:
        flash("Please enter a valid weight in kg (e.g. 68.5).", "error")
        return render_template("register.html", form_data=request.form)

    password_hash = generate_password_hash(password)

    user_id = create_user(
        username=username,
        email=email,
        password_hash=password_hash
    )

    create_user_profile(
        user_id=user_id,
        age=age_val,
        gender=gender,
        height_cm=height_val,
        goal=goal,
        activity_level=activity_level
    )

    add_weight_record(
        user_id=user_id,
        weight_kg=weight_val
    )

    session["user_id"] = user_id
    session["username"] = username

    flash(f"Welcome, {username}! Your account has been created.", "success")
    return redirect(url_for("dashboard"))


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template("login.html")

    identifier = request.form.get("identifier", "").strip()
    password = request.form.get("password", "")
    next_page = request.args.get("next") or request.form.get("next")

    if not identifier or not password:
        flash("Please provide both email/username and password.", "error")
        return render_template("login.html", identifier=identifier)

    user = get_user_by_email(identifier)
    if not user:
        user = get_user_by_username(identifier)

    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid email/username or password.", "error")
        return render_template("login.html", identifier=identifier)

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    flash(f"Welcome back, {user['username']}!", "success")
    if next_page and next_page.startswith("/") and not next_page.startswith("//"):
        return redirect(next_page)
    return redirect(url_for("dashboard"))


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("home"))


# =========================================================
# USER PROFILE & WEIGHT MANAGEMENT
# =========================================================

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("user_id")

    if request.method == "POST":
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "Other").strip()
        height_cm = request.form.get("height_cm", "").strip()
        goal = request.form.get("goal", "Maintain").strip()
        activity_level = request.form.get("activity_level", "Moderately Active").strip()

        try:
            age_val = int(age)
            if age_val < 5 or age_val > 120:
                raise ValueError("Age must be between 5 and 120.")
            height_val = float(height_cm)
            if height_val < 50 or height_val > 270:
                raise ValueError("Height must be between 50 and 270 cm.")
            update_user_profile(
                user_id=user_id,
                age=age_val,
                gender=gender,
                height_cm=height_val,
                goal=goal,
                activity_level=activity_level
            )
            flash("Profile updated successfully.", "success")
        except ValueError as err:
            msg = str(err)
            if not msg:
                msg = "Please enter valid numbers for age and height."
            flash(msg, "error")

        return redirect(url_for("profile"))

    user_prof = get_user_profile(user_id)
    stats = get_weight_stats(user_id)
    today = datetime.now().strftime("%Y-%m-%d")

    return render_template(
        "profile.html",
        profile=user_prof,
        stats=stats,
        latest_weight=get_latest_weight(user_id),
        starting_weight=get_starting_weight(user_id),
        previous_weight=get_previous_weight(user_id),
        weight_history=get_weight_history(user_id),
        today=today
    )


@app.route("/weight")
@login_required
def weight_tracking():
    return redirect(url_for("profile"))


@app.route("/weight/add", methods=["POST"])
@login_required
def add_weight():
    user_id = session.get("user_id")
    weight_kg = request.form.get("weight_kg", "").strip()
    recorded_date = request.form.get("recorded_date", "").strip()

    try:
        weight_val = float(weight_kg)
        if weight_val < 20 or weight_val > 400:
            raise ValueError("Weight must be between 20 and 400 kg.")
        if not recorded_date:
            recorded_date = datetime.now().strftime("%Y-%m-%d")
        else:
            datetime.strptime(recorded_date, "%Y-%m-%d")

        add_weight_record(user_id, weight_val, recorded_date)
        flash(f"Logged weight: {weight_val} kg for {recorded_date}.", "success")
    except ValueError as err:
        msg = str(err)
        if not msg:
            msg = "Please enter a valid weight (20-400 kg) and date (YYYY-MM-DD)."
        flash(msg, "error")

    return redirect(url_for("profile"))


@app.route("/weight/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_weight(record_id):
    user_id = session.get("user_id")
    deleted = delete_weight_record(user_id, record_id)
    if deleted:
        flash("Weight record removed.", "info")
    else:
        flash("Could not delete weight record.", "error")
    return redirect(url_for("profile"))


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
@login_required
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
@login_required
def predict():

    is_async = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
        or request.is_json
    )

    def send_error(message, status_code=400):
        if is_async:
            return jsonify({
                "success": False,
                "error": message
            }), status_code
        flash(message, "error")
        return redirect(url_for("food_scan"))

    file_key = "image" if "image" in request.files else ("food_image" if "food_image" in request.files else None)
    if not file_key:
        return send_error("Please select an image.")

    image = request.files[file_key]

    if image.filename == "":
        return send_error("No image selected.")

    if not allowed_file(image.filename):
        return send_error("Unsupported image format. Allowed formats: JPG, PNG, WEBP, BMP, GIF.")

    filename = secure_filename(image.filename)

    # Prevent filename collisions
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{timestamp}_{filename}"

    image_path = UPLOAD_FOLDER / filename

    try:
        image.save(image_path)
        predictions = predict_food(image_path)
    except Exception as error:
        print(f"Prediction error: {error}")
        if image_path.exists():
            image_path.unlink()
        return send_error("Unable to process the image. Please try another image.", 500)

    for p in predictions:
        p["class_name"] = p.get("food")
        p["confidence_score"] = p.get("confidence")

    if is_async:
        best_prediction = predictions[0] if predictions else None
        target_log_url = url_for("log_food", food=best_prediction["food"]) if best_prediction else url_for("log_food")
        return jsonify({
            "success": True,
            "predictions": predictions,
            "best_prediction": best_prediction,
            "image_filename": filename,
            "accept_url": target_log_url,
            "log_url": target_log_url
        })

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
@login_required
def log_food():

    user_id = session.get("user_id")
    today = datetime.now().strftime("%Y-%m-%d")

    if request.method == "GET":

        selected_food = request.args.get(
            "food",
            ""
        )
        today_entries = get_diary_by_date_user(user_id, today) if user_id else []
        today_totals = calculate_totals(today_entries)

        return render_template(
            "log_food.html",
            selected_food=selected_food,
            preset_weights=PRESET_WEIGHTS,
            meal_types=MEAL_TYPES,
            general_presets=get_general_foods_presets(),
            today_entries=today_entries,
            today_totals=today_totals,
            today=today
        )

    # ---------------------------------------------
    # POST
    # ---------------------------------------------

    is_async = (
        request.is_json
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )

    def send_error(message, status_code=400):
        if is_async:
            return jsonify({
                "success": False,
                "error": message
            }), status_code
        flash(message, "error")
        return redirect(
            url_for("log_food", food=dish_name) if dish_name else url_for("log_food")
        )

    json_data = request.get_json(silent=True) if request.is_json else None

    if json_data:
        dish_name = str(json_data.get("dish_name") or json_data.get("food_name") or "").strip()
        weight = str(json_data.get("weight") or json_data.get("serving_weight_grams") or "").strip()
        meal_type = str(json_data.get("meal_type", "other")).strip().lower()
        notes = str(json_data.get("notes", "")).strip()
        is_custom = json_data.get("is_custom") in [True, "yes", "true", 1]
    else:
        dish_name = (request.form.get("dish_name") or request.form.get("food_name") or "").strip()
        weight = (request.form.get("weight") or request.form.get("serving_weight_grams") or "").strip()
        meal_type = request.form.get("meal_type", "other").strip().lower()
        notes = request.form.get("notes", "").strip()
        is_custom = (request.form.get("is_custom") == "yes")

    # ---------------------------------------------
    # Validate
    # ---------------------------------------------

    if not dish_name:
        return send_error("Food name is required.")

    try:
        weight_value = float(weight)
    except (ValueError, TypeError):
        return send_error("Invalid weight.")

    if weight_value <= 0:
        return send_error("Weight must be greater than 0.")

    if meal_type not in MEAL_TYPES:
        meal_type = "other"

    # ---------------------------------------------
    # Custom nutrition
    # ---------------------------------------------

    custom_nutrition = None

    if is_custom:

        nutrition_fields = {
            "Calories (kcal)": "calories",
            "Protein (g)": "protein",
            "Carbohydrates (g)": "carbohydrates",
            "Fats (g)": "fats",
            "Free Sugar (g)": "free_sugar",
            "Fibre (g)": "fibre",
            "Sodium (mg)": "sodium",
            "Calcium (mg)": "calcium",
            "Iron (mg)": "iron",
            "Vitamin C (mg)": "vitamin_c",
            "Folate (µg)": "folate"
        }

        custom_nutrition = {}

        try:
            for nutrient, form_name in nutrition_fields.items():
                if json_data and isinstance(json_data.get("custom_nutrition"), dict):
                    raw_value = json_data["custom_nutrition"].get(
                        form_name,
                        json_data["custom_nutrition"].get(nutrient, "0")
                    )
                elif json_data:
                    raw_value = json_data.get(form_name, "0")
                else:
                    raw_value = request.form.get(form_name, "0").strip()

                if raw_value == "" or raw_value is None:
                    raw_value = "0"

                custom_nutrition[nutrient] = float(raw_value)

        except (ValueError, TypeError):
            return send_error("Custom nutrition values must be numbers.")

    # ---------------------------------------------
    # Save food
    # ---------------------------------------------

    result = add_food(
        user_id=user_id,
        dish_name=dish_name,
        weight=weight_value,
        meal_type=meal_type,
        notes=notes,
        custom_nutrition_per_100g=custom_nutrition
    )

    if not result["success"]:
        return send_error(result["error"])

    if is_async:
        today_entries = get_diary_by_date_user(user_id, today)
        today_totals = calculate_totals(today_entries)
        # Macro aliases for JSON clients
        today_totals["calories"] = today_totals.get("Calories (kcal)", 0.0)
        today_totals["protein"] = today_totals.get("Protein (g)", 0.0)
        today_totals["carbohydrates"] = today_totals.get("Carbohydrates (g)", 0.0)
        today_totals["fats"] = today_totals.get("Fats (g)", 0.0)

        entry_data = dict(result.get("entry") or {})
        if entry_data:
            entry_data["food_name"] = entry_data.get("dish_name", dish_name)
            entry_data["serving_weight_grams"] = entry_data.get("weight", weight_value)

        coach = get_daily_calorie_coach(user_id, today)
        return jsonify({
            "success": True,
            "message": f"{dish_name} ({weight_value}g) added to your diary.",
            "entry": entry_data,
            "totals": today_totals,
            "today_entries": today_entries,
            "coach": coach
        })

    flash(
        f"{dish_name} ({weight_value}g) added to your diary.",
        "success"
    )

    return redirect(
        url_for("diary")
    )


# =========================================================
# DIARY - TODAY
# =========================================================

@app.route("/diary")
@login_required
def diary():

    user_id = session.get("user_id")

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    entries = get_diary_by_date_user(
        user_id,
        today
    )

    totals = calculate_totals(
        entries
    )

    coach = get_daily_calorie_coach(
        user_id,
        today
    )

    return render_template(
        "diary.html",
        entries=entries,
        totals=totals,
        date=today,
        coach=coach
    )


# =========================================================
# DIARY - SPECIFIC DATE
# =========================================================

@app.route(
    "/diary/<date>"
)
@login_required
def diary_date(date):

    user_id = session.get("user_id")

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

    entries = get_diary_by_date_user(
        user_id,
        date
    )

    totals = calculate_totals(
        entries
    )

    coach = get_daily_calorie_coach(
        user_id,
        date
    )

    return render_template(
        "diary.html",
        entries=entries,
        totals=totals,
        date=date,
        coach=coach
    )


# =========================================================
# HISTORY / DATE RANGE
# =========================================================

@app.route(
    "/history",
    methods=["GET", "POST"]
)
@login_required
def history():
    user_id = session.get("user_id")

    # Read dates from POST form or GET query params
    if request.method == "POST":
        start_date = request.form.get("start_date", "").strip()
        end_date = request.form.get("end_date", "").strip()
    else:
        start_date = request.args.get("start_date", "").strip()
        end_date = request.args.get("end_date", "").strip()

    # Default to past 14 days if dates not provided
    today = datetime.now()
    if not start_date and not end_date:
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    elif not start_date:
        start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    elif not end_date:
        end_date = today.strftime("%Y-%m-%d")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        flash("Please enter valid dates (YYYY-MM-DD).", "error")
        return redirect(url_for("history"))

    if start > end:
        flash("Start date must be before or equal to end date.", "error")
        return redirect(url_for("history"))

    # Daily aggregation (Date, Calories IN, Calories OUT, Net, Weight)
    daily_history = get_user_history_daily(user_id, start_date, end_date, order="DESC")

    # Food detailed records
    entries = get_diary_range_user(user_id, start_date, end_date)
    grouped = group_by_date(entries)
    totals = calculate_totals(entries)

    total_in = float(totals.get("Calories (kcal)", 0.0))
    total_burned = float(sum(d["calories_out"] for d in daily_history))
    total_net = round(total_in - total_burned, 1)

    return render_template(
        "history.html",
        entries=entries,
        grouped=grouped,
        totals=totals,
        daily_history=daily_history,
        total_in=round(total_in, 1),
        total_burned=round(total_burned, 1),
        total_net=total_net,
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
        query=query,
        general_presets=get_general_foods_presets()
    )


# =========================================================
# API FOOD SEARCH
# =========================================================

@app.route("/api/foods")
def api_foods():

    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=40, type=int)

    matching_names = search_foods(query)

    limited_names = matching_names[:limit] if limit > 0 else matching_names

    foods_detailed = []
    for name in limited_names:
        nutr = get_nutrition_per_100g(name) or {}
        foods_detailed.append({
            "name": name,
            "calories": round(float(nutr.get("Calories (kcal)", 0) or 0), 1),
            "protein": round(float(nutr.get("Protein (g)", 0) or 0), 1),
            "carbohydrates": round(float(nutr.get("Carbohydrates (g)", 0) or 0), 1),
            "fats": round(float(nutr.get("Fats (g)", 0) or 0), 1),
            "fibre": round(float(nutr.get("Fibre (g)", 0) or 0), 1)
        })

    return jsonify({
        "success": True,
        "query": query,
        "count": len(matching_names),
        "foods": foods_detailed,
        "food_names": matching_names
    })


# =========================================================
# API DYNAMIC HISTORY & TRENDS
# =========================================================

@app.route("/api/history")
@login_required
def api_history():
    """
    Returns daily history and trends (Calories In, Out, Net, Weight) as JSON
    to dynamically update charts and summary metrics without full page reload.
    """
    user_id = session.get("user_id")
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()

    today = datetime.now()
    if not start_date and not end_date:
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    elif not start_date:
        start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    elif not end_date:
        end_date = today.strftime("%Y-%m-%d")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please enter valid dates (YYYY-MM-DD)."
        }), 400

    if start > end:
        return jsonify({
            "success": False,
            "error": "Start date must be before or equal to end date."
        }), 400

    daily_history = get_user_history_daily(user_id, start_date, end_date, order="ASC")
    entries = get_diary_range_user(user_id, start_date, end_date)
    totals = calculate_totals(entries)

    total_in = round(float(totals.get("Calories (kcal)", 0.0)), 1)
    total_burned = round(float(sum(d["calories_out"] for d in daily_history)), 1)
    total_net = round(total_in - total_burned, 1)

    return jsonify({
        "success": True,
        "start_date": start_date,
        "end_date": end_date,
        "total_in": total_in,
        "total_burned": total_burned,
        "total_net": total_net,
        "active_days": len(daily_history),
        "history": daily_history,
        "data": daily_history,
        "summary": {
            "total_calories_in": total_in,
            "total_calories_out": total_burned,
            "net_calories": total_net,
            "days_logged": len(daily_history),
            "start_date": start_date,
            "end_date": end_date
        }
    })


# =========================================================
# API DYNAMIC WEIGHT PREDICTION FORECAST
# =========================================================

@app.route("/api/weight-prediction")
@login_required
def api_weight_prediction():
    """
    Returns personalized LSTM and metabolic weight trajectory forecast data
    as JSON for interactive dynamic Chart.js horizon updating.
    """
    user_id = session.get("user_id")
    days = request.args.get("days", default=30, type=int)
    if days not in [7, 14, 30, 60, 90]:
        days = 30

    try:
        forecast_data = predict_weight_trajectory(user_id, forecast_days=days)
        return jsonify({
            "success": True,
            "days": forecast_data.get("days", days),
            "current_weight": forecast_data.get("current_weight"),
            "trajectory": forecast_data.get("trajectory", []),
            "milestones": forecast_data.get("milestones", {}),
            "model_used": forecast_data.get("model_used", "lstm_and_metabolic"),
            "projected_final_weight": forecast_data.get("projected_final_weight"),
            "total_projected_change": forecast_data.get("total_projected_change"),
            "data": forecast_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# API GENERAL FOODS PRESETS
# =========================================================

@app.route("/api/general-foods")
def api_general_foods():

    query = request.args.get(
        "q",
        ""
    ).strip().lower()

    presets = get_general_foods_presets()

    if query:
        presets = [
            p for p in presets
            if query in p["name"].lower()
        ]

    return jsonify(presets)


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

#cgabf


# =========================================================
# EXPORT DIARY
# =========================================================

@app.route(
    "/export"
)
@login_required
def export_diary():

    user_id = session.get("user_id")

    csv_path = export_user_diary_csv(user_id)

    if not csv_path or not Path(csv_path).exists():

        flash(
            "There is no diary data to export.",
            "error"
        )

        return redirect(
            url_for("diary")
        )

    username = session.get("username", "user")

    return send_file(
        csv_path,
        as_attachment=True,
        download_name=f"food_diary_{username}.csv"
    )


# =========================================================
# DELETE FOOD INTAKE
# =========================================================

@app.route(
    "/food/delete/<int:record_id>",
    methods=["POST"]
)
@login_required
def delete_food(record_id):

    user_id = session.get("user_id")

    deleted = delete_food_entry(user_id, record_id)

    if deleted:
        flash("Food entry removed from diary.", "info")
    else:
        flash("Could not remove food entry.", "error")

    return redirect(request.referrer or url_for("diary"))


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
@login_required
def calorie_out():
    """
    Display the calorie-out page with user-specific weight pre-filled and today's activity log.
    """
    user_id = session.get("user_id")
    categories = get_activity_categories()
    latest_w = get_latest_weight(user_id)
    user_weight = latest_w["weight_kg"] if latest_w else None
    today = datetime.now().strftime("%Y-%m-%d")
    today_exercises = get_calorie_out_by_date(user_id, today)
    today_burned = get_calorie_out_total(user_id, today)

    return render_template(
        "calorie_out.html",
        categories=categories,
        user_weight=user_weight,
        today_exercises=today_exercises,
        today_burned=today_burned,
        today=today
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
@login_required
def calculate_calories_api():
    """
    Calculate calories burned for an activity and save to calorie_out_records with user_id.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({
            "success": False,
            "error": "Authentication required."
        }), 401

    try:
        data = request.get_json(silent=True) or {}

        activity_code = data.get("activity_code")
        weight_kg = data.get("weight_kg")
        duration_minutes = data.get("duration_minutes")

        if not activity_code:
            return jsonify({
                "success": False,
                "error": "Please select an activity."
            }), 400

        # Auto-fill body weight from user's profile/weight records if not provided
        if weight_kg is None or str(weight_kg).strip() == "":
            latest_w = get_latest_weight(user_id)
            if latest_w:
                weight_kg = float(latest_w["weight_kg"])
            else:
                return jsonify({
                    "success": False,
                    "error": "Weight is required. Please set your weight in profile."
                }), 400

        if duration_minutes is None:
            return jsonify({
                "success": False,
                "error": "Duration is required."
            }), 400

        result = calculate_calories_for_activity(
            activity_code=activity_code,
            weight_kg=weight_kg,
            duration_minutes=duration_minutes
        )

        record_id = add_calorie_out_record(
            user_id=user_id,
            activity=result["description"],
            met=result["met"],
            weight_kg=result["weight_kg"],
            duration_minutes=result["duration_minutes"],
            calories_burned=result["calories_burned"]
        )

        return jsonify({
            "success": True,
            "record_id": record_id,
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


@app.route("/calorie-out/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_exercise(record_id):
    """
    Delete an exercise record belonging to the authenticated user.
    """
    user_id = session.get("user_id")
    deleted = delete_calorie_out_record(user_id, record_id)
    if deleted:
        flash("Exercise record removed.", "info")
    else:
        flash("Could not remove exercise record.", "error")
    return redirect(request.referrer or url_for("calorie_out"))


@app.route("/dashboard")
@login_required
def dashboard():

    user_id = session.get("user_id")

    today = datetime.now().strftime("%Y-%m-%d")

    # -----------------------------------------
    # CALORIES IN
    # -----------------------------------------

    food_entries = get_diary_by_date_user(user_id, today)

    food_totals = calculate_totals(food_entries)

    calories_in = food_totals.get(
        "Calories (kcal)",
        0
    )


    # -----------------------------------------
    # CALORIES OUT
    # -----------------------------------------

    exercise_entries = get_calorie_out_by_date(user_id, today)

    calories_out = get_calorie_out_total(user_id, today)


    # -----------------------------------------
    # NET CALORIES
    # -----------------------------------------

    net_calories = calories_in - calories_out

    # -----------------------------------------
    # USER WEIGHT & PROFILE STATS
    # -----------------------------------------

    latest_w = get_latest_weight(user_id)
    current_weight = latest_w["weight_kg"] if latest_w else None
    weight_stats = get_weight_stats(user_id)
    profile = get_user_profile(user_id)
    coach = get_daily_calorie_coach(user_id, today)

    return render_template(
        "dashboard.html",
        date=today,
        calories_in=round(calories_in, 2),
        calories_out=round(calories_out, 2),
        net_calories=round(net_calories, 2),
        food_entries=food_entries,
        exercise_entries=exercise_entries,
        food_totals=food_totals,
        current_weight=current_weight,
        weight_stats=weight_stats,
        profile=profile,
        coach=coach
    )


# =========================================================
# WEIGHT PREDICTION (LSTM & METABOLIC FORECASTING)
# =========================================================

@app.route("/predict-weight")
@login_required
def predict_weight():
    """
    AI and metabolic weight prediction for the authenticated user.
    """
    user_id = session.get("user_id")
    days = request.args.get("days", default=30, type=int)
    if days not in [7, 14, 30, 60, 90]:
        days = 30
    try:
        forecast_data = predict_weight_trajectory(user_id, forecast_days=days)
    except Exception as e:
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({"success": False, "error": str(e)}), 500
        flash(f"Error calculating weight prediction: {e}", "danger")
        return redirect(url_for("dashboard"))

    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({"success": True, "data": forecast_data})

    return render_template("predict_weight.html", data=forecast_data)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )