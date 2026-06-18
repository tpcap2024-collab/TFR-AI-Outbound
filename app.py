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
   

    # ==============================
    # RESIZE & CROP
    # ==============================
    img = cv2.resize(img, (640, 480))

    h, w = img.shape[:2]

    # remove watermark
    img = img[:int(h * 0.92), :]
    h, w = img.shape[:2]

    roi = img[
        int(h * 0.08):int(h * 0.90),
        int(w * 0.05):int(w * 0.95)
    ]

    rh, rw = roi.shape[:2]

    # ==============================
    # LIGHT NORMALIZATION (better than CLAHE-only)
    # ==============================
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ==============================
    # COLOR SPACE
    # ==============================
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ==============================
    # WALL MASK (adaptive)
    # ==============================
    v_mean = np.mean(hsv[:, :, 2])

    # dynamic threshold (smooth transition)
    wall_lower = np.array([0, 0, max(60, int(v_mean * 0.5))])
    wall_upper = np.array([180, 60, min(255, int(v_mean * 1.2))])

    wall_mask = cv2.inRange(hsv, wall_lower, wall_upper)

    # ==============================
    # CARGO SEGMENTATION (Hybrid)
    # ==============================

    # color-based
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    brown = cv2.inRange(hsv, (5, 40, 40), (30, 255, 255))

    # adaptive dark detection
    _, dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # edge-based (better than Laplacian)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    cargo_mask = cv2.bitwise_or(green, brown)
    cargo_mask = cv2.bitwise_or(cargo_mask, dark)
    cargo_mask = cv2.bitwise_or(cargo_mask, edges)

    # ==============================
    # MORPHOLOGY
    # ==============================
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cargo_mask = cv2.morphologyEx(cargo_mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    cargo_mask = cv2.morphologyEx(cargo_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # ==============================
    # REMOVE NOISE
    # ==============================
    def clean(mask, ratio=0.002):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        result = np.zeros_like(mask)

        min_area = int(mask.size * ratio)

        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                result[labels == i] = 255

        return result

    cargo_mask = clean(cargo_mask, 0.002)
    wall_mask = clean(wall_mask, 0.002)

    # ==============================
    # EMPTY AREA
    # ==============================
    empty_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(cargo_mask))

    # ==============================
    # SMART PERSPECTIVE WEIGHT
    # ==============================

    # emphasize bottom more strongly (non-linear)
    y = np.linspace(0, 1, rh)
    weights = 0.5 + (y ** 1.8) * 1.5
    weights = weights.reshape(rh, 1)

    cargo_score = np.sum((cargo_mask > 0) * weights)
    empty_score = np.sum((empty_mask > 0) * weights)

    occupancy = cargo_score / (cargo_score + empty_score + 1e-6)
    occupancy = np.clip(occupancy, 0, 1)

    # ==============================
    # IMPROVED CALIBRATION
    # ==============================
    # smooth + reduce over-estimation
    volume = (occupancy ** 0.9) * 100

    # optional bias correction
    volume = volume * 0.95 + 2

    volume = np.clip(volume, 0, 100)

    # round to 5%
    volume = int(round(volume / 5) * 5)

    # ==============================
    # DEBUG OUTPUT
    # ==============================
    debug = roi.copy()

    debug[cargo_mask > 0] = (0, 255, 0)
    debug[empty_mask > 0] = (0, 0, 255)

    overlay = cv2.addWeighted(roi, 0.65, debug, 0.35, 0)

    cv2.imwrite("debug_overlay.jpg", overlay)
    cv2.imwrite("debug_cargo.jpg", cargo_mask)
    cv2.imwrite("debug_empty.jpg", empty_mask)

    print(f"VMEAN={v_mean:.1f} OCC={occupancy:.3f} VOL={volume}%")

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
