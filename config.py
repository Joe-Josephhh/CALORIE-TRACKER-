from pathlib import Path
import os


# ---------------------------------------------------------
# Base project directory
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Model and data paths
# ---------------------------------------------------------

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "outputs" / "checkpoints" / "best_model_finetuned.keras"
    )
)

CLASS_NAMES_PATH = Path(
    os.getenv(
        "CLASS_NAMES_PATH",
        BASE_DIR / "configs" / "class_names.csv"
    )
)

NUTRITION_CSV = Path(
    os.getenv(
        "NUTRITION_CSV",
        BASE_DIR / "data" / "Indian_Food_macros.csv"
    )
)


UPLOAD_FOLDER = BASE_DIR / "uploads"


# ---------------------------------------------------------
# ML configuration
# ---------------------------------------------------------

IMG_SIZE = (224, 224)
TOP_K = 5


# ---------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key"
)

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
}


# ---------------------------------------------------------
# Application choices
# ---------------------------------------------------------

PRESET_WEIGHTS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500
]

MEAL_TYPES = [
    "breakfast",
    "lunch",
    "snack",
    "dinner",
    "other"
]


# ---------------------------------------------------------
# Create required directories
# ---------------------------------------------------------

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

(BASE_DIR / "data").mkdir(
    parents=True,
    exist_ok=True
)