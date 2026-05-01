import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import L2
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ─── Config ────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 110
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "1.keras")
CLASS_NAMES = ["Not Orange", "Orange"]   # index 0 = Not Orange, 1 = Orange

model = None

# ─── Rebuild the exact architecture from your notebook ─────────────────────
def build_model():
    base_model = EfficientNetB0(
        weights=None,           # weights are loaded from file, not imagenet
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    for layer in base_model.layers:
        layer.trainable = False

    m = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=L2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation="softmax"),
    ])
    return m


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file not found at {MODEL_PATH}")
        print(f"[!] Copy your .keras file from Google Drive into the backend/ folder.")
        print(f"[!] Running in MOCK mode until model is added.")
        return

    try:
        # ── Attempt 1: direct load (works if Keras versions match) ──
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"[✓] Model loaded directly from {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Direct load failed ({e}), rebuilding architecture and loading weights…")
        try:
            # ── Attempt 2: rebuild architecture, load weights only ──
            model = build_model()
            # Build the model by running a dummy forward pass
            dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            model(dummy, training=False)
            model.load_weights(MODEL_PATH)
            print(f"[✓] Weights loaded into rebuilt architecture from {MODEL_PATH}")
        except Exception as e2:
            print(f"[✗] Weight loading also failed: {e2}")
            print(f"[!] Running in MOCK mode.")
            model = None
            return

    print(f"[✓] Input shape:  {model.input_shape}")
    print(f"[✓] Output shape: {model.output_shape}")

load_model()


# ─── Image pre-processing (matches your Colab notebook exactly) ────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)               # EfficientNet-specific normalisation
    return np.expand_dims(arr, axis=0)        # shape: (1, 110, 110, 3)


# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image' in form-data."}), 400

    image_bytes = request.files["image"].read()

    if model is not None:
        input_tensor = preprocess_image(image_bytes)
        probs        = model.predict(input_tensor, verbose=0)[0]  # shape: (2,)
        class_idx    = int(np.argmax(probs))
        confidence   = float(probs[class_idx])
        is_orange    = class_idx == 1
    else:
        # Mock mode — remove once model file is in place
        import random
        is_orange  = random.random() > 0.4
        confidence = round(random.uniform(0.75, 0.99), 4)

    return jsonify({
        "label":      CLASS_NAMES[1 if is_orange else 0],
        "is_orange":  is_orange,
        "confidence": round(confidence * 100, 2),
        "raw_score":  round(confidence, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)