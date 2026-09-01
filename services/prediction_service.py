import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

from config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    TOP_K
)


# ---------------------------------------------------------
# Load model ONCE when Flask starts
# ---------------------------------------------------------

print("Loading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Load class names
# ---------------------------------------------------------

class_df = pd.read_csv(CLASS_NAMES_PATH)

class_df.columns = (
    class_df.columns
    .str.strip()
    .str.replace('"', '', regex=False)
)

class_names = (
    class_df[class_df.columns[0]]
    .astype(str)
    .str.strip()
    .str.replace('"', '', regex=False)
    .tolist()
)

print(f"Loaded {len(class_names)} food classes.")


# ---------------------------------------------------------
# Preprocess image
# ---------------------------------------------------------

def load_image(image_path):

    image = tf.io.read_file(str(image_path))

    # decode_image supports JPG, PNG, BMP, GIF etc.
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    return image


# ---------------------------------------------------------
# Predict food
# ---------------------------------------------------------

def predict_food(image_path, top_k=TOP_K):

    image_tensor = load_image(image_path)

    predictions = model.predict(
        image_tensor,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        results.append({
            "rank": rank,
            "food": class_names[index],
            "confidence": float(
                predictions[index] * 100
            )
        })

    return results


# ---------------------------------------------------------
# Best prediction
# ---------------------------------------------------------

def get_best_prediction(predictions):

    if not predictions:
        return None

    return predictions[0]