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

    img = cv2.resize(img, (640, 480))

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =========================
    # DETECT VIEW TYPE
    # =========================
    if h >= w:
        view_type = "rear"
    else:
        view_type = "side"

    # =========================
    # ROI
    # =========================
    if view_type == "rear":

        roi = hsv[
            int(h * 0.18):int(h * 0.82),
            int(w * 0.12):int(w * 0.88)
        ]

    else:

        roi = hsv[
            int(h * 0.10):int(h * 0.85),
            int(w * 0.08):int(w * 0.92)
        ]

    if roi.size == 0:
        return 0

    # =========================
    # EMPTY SPACE
    # =========================
    empty_mask = cv2.inRange(
        roi,
        (0, 0, 120),
        (180, 60, 255)
    )

    # =========================
    # CLEAN
    # =========================
    kernel = np.ones((5, 5), np.uint8)

    empty_mask = cv2.morphologyEx(
        empty_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    empty_mask = cv2.morphologyEx(
        empty_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =========================
    # EMPTY RATIO
    # =========================
    empty_ratio = (
        np.count_nonzero(empty_mask)
        / empty_mask.size
    )

    # =========================
    # LOAD
    # =========================
    load_ratio = 1 - empty_ratio

    volume = int(load_ratio * 100)

    # =========================
    # CALIBRATION
    # =========================
    if volume < 20:
        volume *= 0.8

    elif volume > 80:
        volume *= 1.1

    volume = int(round(volume / 5) * 5)

    volume = max(0, min(100, volume))

    print(
        f"VIEW={view_type} "
        f"EMPTY={empty_ratio:.2f} "
        f"LOAD={volume}%"
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
