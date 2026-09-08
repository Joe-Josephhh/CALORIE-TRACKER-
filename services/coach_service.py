from datetime import datetime
from database import (
    get_user_profile,
    get_latest_weight,
    get_calorie_out_total
)
from services.diary_service import (
    get_diary_by_date_user,
    calculate_totals
)
from services.weight_prediction_service import (
    calculate_bmr,
    calculate_tdee
)


# Standard everyday activities with their MET values for burn-off recommendations
BURN_OFF_ACTIVITIES = [
    {
        "name": "Brisk Walking (3.5 mph)",
        "category": "walking",
        "met": 4.3,
        "icon": "🚶",
        "description": "Brisk walking at a comfortable 3.5 mph pace"
    },
    {
        "name": "Cycling (Moderate Effort)",
        "category": "bicycling",
        "met": 7.0,
        "icon": "🚴",
        "description": "Outdoor or stationary cycling with moderate effort"
    },
    {
        "name": "Jogging / Running (5 mph)",
        "category": "running",
        "met": 8.3,
        "icon": "🏃",
        "description": "Light jogging or running at 12 min/mile pace"
    },
    {
        "name": "Football / Team Sports",
        "category": "sports",
        "met": 8.0,
        "icon": "⚽",
        "description": "Playing soccer, football, or fast-paced sports"
    },
    {
        "name": "Swimming Laps",
        "category": "water activities",
        "met": 8.0,
        "icon": "🏊",
        "description": "Freestyle laps with continuous moderate effort"
    },
    {
        "name": "Jump Rope (Moderate)",
        "category": "conditioning exercise",
        "met": 10.0,
        "icon": "⚡",
        "description": "Skipping rope at moderate continuous pace"
    }
]


def calculate_burn_off_duration(excess_calories, met, weight_kg):
    """
    Inverts the MET calorie formula:
    Calories = (MET * 3.5 * weight_kg / 200) * duration_minutes
    => duration_minutes = (Calories * 200) / (MET * 3.5 * weight_kg)
    """
    if weight_kg <= 0 or met <= 0 or excess_calories <= 0:
        return 0
    duration = (float(excess_calories) * 200.0) / (float(met) * 3.5 * float(weight_kg))
    return max(1, round(duration))


def get_daily_calorie_coach(user_id, date_str=None):
    """
    Dynamic calorie target and burn-off activity coach for the user.
    - Determines daily budget from biological profile (BMR, TDEE) & Goal.
    - Tracks live calories consumed vs burned today.
    - Tells user how many calories they can still eat to stay on goal.
    - If over budget, calculates exact exercise durations to burn off the excess.
    """
    if not user_id:
        return {
            "has_profile": False,
            "message": "Authentication required."
        }

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    profile = get_user_profile(user_id)
    latest_weight_obj = get_latest_weight(user_id)

    if not profile or not latest_weight_obj:
        return {
            "has_profile": False,
            "message": "Please set up your profile and record your body weight to enable the Smart Calorie Coach."
        }

    weight_kg = float(latest_weight_obj["weight_kg"])
    height_cm = float(profile["height_cm"])
    age = int(profile["age"])
    gender = profile["gender"]
    raw_goal = str(profile["goal"]).strip()
    activity_level = profile["activity_level"]

    # 1. Base metabolism (BMR & TDEE)
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    # 2. Goal-specific daily calorie target
    goal_lower = raw_goal.lower()
    if "lose" in goal_lower or "loss" in goal_lower:
        goal_type = "loss"
        goal_display = "Weight Loss"
        # Standard sustainable deficit of 500 kcal/day (floor at 1200 for health safety)
        daily_target = max(1200.0, round(tdee - 500.0, 1))
        target_description = "Deficit target (-500 kcal/day for ~0.5 kg/week fat loss)"
    elif "gain" in goal_lower:
        goal_type = "gain"
        goal_display = "Weight Gain"
        # Moderate lean surplus of 400 kcal/day
        daily_target = round(tdee + 400.0, 1)
        target_description = "Surplus target (+400 kcal/day for lean muscle growth)"
    else:
        goal_type = "maintain"
        goal_display = "Maintain Weight"
        daily_target = round(tdee, 1)
        target_description = "Maintenance target (equal to your daily energy expenditure)"

    # 3. Live daily intake & exercise burn
    food_entries = get_diary_by_date_user(user_id, date_str)
    food_totals = calculate_totals(food_entries)
    calories_in = round(float(food_totals.get("Calories (kcal)", 0.0) or 0.0), 1)
    calories_out = round(float(get_calorie_out_total(user_id, date_str) or 0.0), 1)

    # Net intake: Food consumed minus physical activity
    net_intake = round(calories_in - calories_out, 1)

    # Remaining calorie budget
    remaining = round(daily_target - net_intake, 1)
    is_over_budget = remaining < 0
    excess_calories = round(abs(remaining), 1) if is_over_budget else 0.0

    # Percentage of daily budget used (capped between 0% and 150%)
    if daily_target > 0:
        pct_used = round((net_intake / daily_target) * 100, 1)
        progress_bar_pct = min(100.0, max(0.0, pct_used))
    else:
        pct_used = 0.0
        progress_bar_pct = 0.0

    # 4. Burn-off exercise recommendations if over budget
    burn_off_suggestions = []
    if is_over_budget and excess_calories > 0:
        for act in BURN_OFF_ACTIVITIES:
            mins = calculate_burn_off_duration(excess_calories, act["met"], weight_kg)
            burn_off_suggestions.append({
                "name": act["name"],
                "category": act["category"],
                "met": act["met"],
                "icon": act["icon"],
                "description": act["description"],
                "duration_minutes": mins,
                "calories_burned": excess_calories
            })

    # 5. Tailored coach messages & insights
    if not is_over_budget:
        if goal_type == "loss":
            status_title = "On Track for Fat Loss"
            status_message = f"You can eat {remaining:.0f} kcal more today to stay within your optimal deficit target."
            tip = "Choose lean proteins, vegetables, or light snacks for your remaining calories."
        elif goal_type == "gain":
            status_title = "Surplus Target in Progress"
            status_message = f"You still need {remaining:.0f} kcal more today to hit your growth surplus."
            tip = "Try calorie-dense whole foods like nuts, peanut butter, eggs, or oats."
        else:
            status_title = "Balanced Energy Intake"
            status_message = f"You have {remaining:.0f} kcal remaining today to maintain your energy equilibrium."
            tip = "Your intake and expenditure are nicely aligned."
    else:
        if goal_type == "loss":
            status_title = "Calorie Target Exceeded"
            status_message = f"You are {excess_calories:.0f} kcal over your weight loss budget for today."
            tip = "No problem! You can easily offset this surplus by doing one of the suggested activities below."
        elif goal_type == "gain":
            status_title = "Above Growth Target"
            status_message = f"You are {excess_calories:.0f} kcal above your planned surplus for today."
            tip = "A brisk walk or short cardio session can help keep the gain lean."
        else:
            status_title = "Surplus Detected"
            status_message = f"You are {excess_calories:.0f} kcal above your maintenance target."
            tip = "Pick any activity below to burn off the extra calories and stay balanced."

    return {
        "has_profile": True,
        "goal": goal_display,
        "goal_type": goal_type,
        "target_description": target_description,
        "current_weight": weight_kg,
        "bmr": bmr,
        "tdee": tdee,
        "daily_target": daily_target,
        "calories_in": calories_in,
        "calories_out": calories_out,
        "net_intake": net_intake,
        "remaining": remaining,
        "is_over_budget": is_over_budget,
        "excess_calories": excess_calories,
        "pct_used": pct_used,
        "progress_bar_pct": progress_bar_pct,
        "status_title": status_title,
        "status_message": status_message,
        "tip": tip,
        "burn_off_suggestions": burn_off_suggestions
    }
