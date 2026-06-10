from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import threading
import pickle
import os

app = Flask(__name__)

APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

lock = threading.Lock()
processed_ids = set()

MODEL_PATH = "tfr_model.pkl"

# =========================
# LOAD MODEL
# =========================
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    model = {
        "w": np.array([1.0, 1.0, 1.0, 1.0]),
        "b": 0.0
    }


# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        return cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )
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
    roi = mask[int(h*0.08):int(h*0.95), int(w*0.02):int(w*0.98)]

    if roi.size == 0:
        return np.zeros(4)

    area = np.mean(roi > 0)

    top = roi[:int(roi.shape[0]*0.3), :]
    mid = roi[int(roi.shape[0]*0.3):int(roi.shape[0]*0.7), :]
    bottom = roi[int(roi.shape[0]*0.7):, :]

    return np.array([
        area,
        np.mean(top > 0),
        np.mean(mid > 0),
        np.mean(bottom > 0)
    ])


# =========================
# PREDICT
# =========================
def predict(features):
    return float(np.dot(model["w"], features) + model["b"])


# =========================
# TRAIN
# =========================
def train(features, y_true, y_pred, lr=0.03):

    error = y_true - y_pred

    model["w"] += lr * error * features
    model["b"] += lr * error


# =========================
# GET ACTUAL FROM APPSHEET
# =========================
def get_actual(row_id):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Find",
        "Properties": {
            "Selector": f"FILTER(\"{TABLE_NAME}\", [id] = \"{row_id}\")"
        }
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)

        data = r.json()

        rows = data.get("Rows", [])
        if not rows:
            return None

        return float(rows[0].get("Fill Rate", 0))  # 🔥 คอลัมน์จริงของคุณ

    except:
        return None


# =========================
# SAVE MODEL
# =========================
def save_model():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():

    try:
        data = request.get_json(silent=True)

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        img = download_image(image_url)
        if img is None:
            return jsonify({"error": "image fail"}), 400

        features = extract_features(img)

        pred = predict(features)
        pred = max(0, min(100, pred))
        pred_out = int(round(pred / 5) * 5)

        # =========================
        # AUTO LEARNING FROM APPSHEET
        # =========================
        actual = get_actual(row_id)

        if actual is not None and actual > 0:
            train(features, actual, pred)
            save_model()

            print(f"TRAINED | pred={pred_out} actual={actual}")

        return jsonify({
            "status": "success",
            "pred": pred_out,
            "actual": actual
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, threaded=True)
