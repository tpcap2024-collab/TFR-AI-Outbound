from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import threading

app = Flask(__name__)

# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# MEMORY (SELF-LEARNING BIAS)
# =========================
error_history = []
lock = threading.Lock()
processed_ids = set()


# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(url, timeout=15)
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
# RAW VOLUME MODEL (NO BIAS FIX)
# =========================
def gen_volume_raw(img):

    img = cv2.resize(img, (640, 480))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask = (
        cv2.inRange(hsv, (10, 35, 60), (40, 255, 255)) |
        cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 60, 50), (180, 255, 255)) |
        cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape

    # ROI (truck interior only)
    roi = mask[int(h*0.12):int(h*0.92), int(w*0.03):int(w*0.97)]

    if roi.size == 0:
        return 0

    area_density = np.count_nonzero(roi) / roi.size
    area_density = np.clip(area_density, 0, 1)

    volume = area_density * 100

    return volume


# =========================
# SELF CALIBRATION (BIAS CORRECTION)
# =========================
def calibrate(volume):

    global error_history

    if len(error_history) > 0:
        bias = np.mean(error_history[-30:])  # rolling window
    else:
        bias = 0

    corrected = volume + bias

    return max(0, min(100, corrected))


# =========================
# LEARNING SYSTEM
# =========================
def update_bias(pred, actual):

    global error_history

    error = actual - pred
    error_history.append(error)

    # limit memory
    if len(error_history) > 300:
        error_history = error_history[-300:]


# =========================
# APPSHEET UPDATE
# =========================
def update_appsheet(row_id, volume_text):

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "id": row_id,
                "TFR AI": volume_text,
                "status": "Done"
            }
        ]
    }

    try:
        requests.post(url, json=payload, headers=headers, timeout=20)
    except:
        pass


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        # =========================
        # DUPLICATE LOCK
        # =========================
        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped"}), 200
            processed_ids.add(row_id)

        # =========================
        # IMAGE LOAD
        # =========================
        img = download_image(image_url)
        if img is None:
            return jsonify({"error": "image fail"}), 400

        # =========================
        # MODEL
        # =========================
        raw = gen_volume_raw(img)
        corrected = calibrate(raw)

        volume = int(round(corrected / 5) * 5)
        volume = max(0, min(100, volume))

        # =========================
        # OPTIONAL LEARNING (if ground truth exists)
        # =========================
        actual = data.get("actual")
        if actual is not None:
            update_bias(volume, float(actual))

        volume_text = f"{volume}%"

        print(f"RAW: {raw:.2f} → FINAL: {volume_text}")

        # =========================
        # UPDATE SHEET
        # =========================
        update_appsheet(row_id, volume_text)

        return jsonify({
            "status": "success",
            "raw": raw,
            "volume": volume_text
        })

    except:
        print(traceback.format_exc())
        return jsonify({"error": "server error"}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )
