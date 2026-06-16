from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback
import time
import threading

app = Flask(__name__)

# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"

# =========================
# LOCK
# =========================
processed_ids = set()
lock = threading.Lock()


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
# 🔥 BALANCED VOLUME MODEL
# =========================
def gen_volume(img):

    # =========================
    # DETECT VIEW TYPE
    # =========================
    orig_h, orig_w = img.shape[:2]

    if orig_h > orig_w:
        view_type = "rear"
    else:
        view_type = "side"

    # =========================
    # RESIZE
    # =========================
    img = cv2.resize(img, (640, 480))

    h, w = img.shape[:2]

    # =========================
    # REMOVE WATERMARK
    # =========================
    img = img[:int(h * 0.92), :]

    h, w = img.shape[:2]

    # =========================
    # ROI
    # =========================
    roi = img[
        int(h * 0.10):int(h * 0.90),
        int(w * 0.05):int(w * 0.95)
    ]

    if roi.size == 0:
        return 0

    # =========================
    # GRAY
    # =========================
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # =========================
    # BLUR
    # =========================
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # =========================
    # THRESHOLD
    # =========================
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    # =========================
    # MORPHOLOGY
    # =========================
    kernel = np.ones((5, 5), np.uint8)

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # =========================
    # REMOVE SMALL NOISE
    # =========================
    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(th)

    for c in contours:

        area = cv2.contourArea(c)

        if area > 500:
            cv2.drawContours(
                mask,
                [c],
                -1,
                255,
                -1
            )

    # =========================
    # OCCUPANCY
    # =========================
    occupied_pixels = np.count_nonzero(mask)

    total_pixels = mask.size

    occupancy = occupied_pixels / total_pixels

    # =========================
    # SCALE
    # =========================
    volume = int(occupancy * 140)

    volume = max(0, min(100, volume))

    volume = int(round(volume / 5) * 5)

    print(
        f"VIEW={view_type} "
        f"OCC={occupancy:.3f} "
        f"VOL={volume}%"
    )

    return volume


# =========================
# UPDATE APPSHEET
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
        # AI PROCESS
        # =========================
        volume = gen_volume(img)
        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        # =========================
        # UPDATE SHEET
        # =========================
        update_appsheet(row_id, volume_text)

        return jsonify({
            "status": "success",
            "id": row_id,
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
