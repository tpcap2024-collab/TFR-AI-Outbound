from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import os
import joblib
from sklearn.linear_model import SGDRegressor

app = Flask(__name__)

# =========================
# MODEL FILE
# =========================
MODEL_PATH = "tfr_regression.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = SGDRegressor(
        learning_rate="constant",
        eta0=0.01,
        max_iter=1,
        warm_start=True
    )

# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=(5, 10))
        if r.status_code != 200:
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )
        return img
    except:
        return None


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(img):

    img = cv2.resize(img, (640, 480))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, (10, 30, 60), (40, 255, 255)) |
        cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 50, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    )

    h, w = mask.shape
    roi = mask[int(h*0.10):int(h*0.95), int(w*0.02):int(w*0.98)]

    if roi.size == 0:
        return np.zeros(4)

    area = np.mean(roi > 0)

    top = np.mean(roi[:int(roi.shape[0]*0.3), :] > 0)
    mid = np.mean(roi[int(roi.shape[0]*0.3):int(roi.shape[0]*0.7), :] > 0)
    bottom = np.mean(roi[int(roi.shape[0]*0.7):, :] > 0)

    return np.array([area, top, mid, bottom])


# =========================
# PREDICT
# =========================
def predict(features):
    return model.predict([features])[0]


# =========================
# TRAIN
# =========================
def train(features, actual):

    model.partial_fit([features], [actual])

    joblib.dump(model, MODEL_PATH)


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        actual = data.get("actual")  # จาก AppSheet (%Fill Rate)

        if not image_url:
            return jsonify({"error": "missing link"}), 400

        # =========================
        # LOAD IMAGE
        # =========================
        img = download_image(image_url)
        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # FEATURES
        # =========================
        features = extract_features(img)

        # =========================
        # PREDICT
        # =========================
        pred = predict(features)
        pred = max(0, min(100, pred))
        pred_out = int(round(pred / 5) * 5)

        # =========================
        # ONLINE TRAINING
        # =========================
        if actual is not None:
            try:
                train(features, float(actual))
            except:
                pass

        return jsonify({
            "status": "success",
            "pred": pred_out
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


# =========================
# HEALTH CHECK
# =========================
@app.route("/")
def home():
    return "TFR Regression AI OK", 200


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "sgd-regression"
    }), 200


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )
