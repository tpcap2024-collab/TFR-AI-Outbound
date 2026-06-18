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

    # =====================================================
    # RESIZE
    # =====================================================
    img = cv2.resize(img, (640, 480))

    h, w = img.shape[:2]

    # ตัด watermark
    img = img[:int(h * 0.92), :]

    h, w = img.shape[:2]

    # =====================================================
    # CLAHE
    # =====================================================
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # =====================================================
    # ROI
    # =====================================================
    roi = img[
        int(h * 0.08):int(h * 0.90),
        int(w * 0.05):int(w * 0.95)
    ]

    rh, rw = roi.shape[:2]

    # =====================================================
    # HSV
    # =====================================================
    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # =====================================================
    # WALL MASK
    # =====================================================
    v_mean = np.mean(hsv[:, :, 2])

    if v_mean < 100:

        wall_lower = np.array(
            [0, 0, 70]
        )

        wall_upper = np.array(
            [180, 45, 220]
        )

    elif v_mean < 160:

        wall_lower = np.array(
            [0, 0, 90]
        )

        wall_upper = np.array(
            [180, 55, 245]
        )

    else:

        wall_lower = np.array(
            [0, 0, 110]
        )

        wall_upper = np.array(
            [180, 60, 255]
        )

    wall_mask = cv2.inRange(
        hsv,
        wall_lower,
        wall_upper
    )

    # =====================================================
    # CARGO MASK
    # =====================================================

    # GREEN PALLET

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 25, 25]),
        np.array([95, 255, 255])
    )

    # BROWN CARTON

    brown_mask = cv2.inRange(
        hsv,
        np.array([5, 30, 30]),
        np.array([35, 255, 255])
    )

    # DARK OBJECT

    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 75])
    )

    cargo_mask = cv2.bitwise_or(
        green_mask,
        brown_mask
    )

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        dark_mask
    )

    # =====================================================
    # TEXTURE MASK
    # =====================================================
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    lap = np.abs(
        lap
    ).astype(
        np.uint8
    )

    texture_mask = cv2.threshold(
        lap,
        20,
        255,
        cv2.THRESH_BINARY
    )[1]

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        texture_mask
    )

    # =====================================================
    # MORPHOLOGY
    # =====================================================

    cargo_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        cargo_kernel,
        iterations=2
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        cargo_kernel,
        iterations=1
    )

    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        wall_kernel,
        iterations=2
    )

    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_OPEN,
        wall_kernel,
        iterations=1
    )

    # =====================================================
    # REMOVE SMALL BLOBS
    # =====================================================

    def clean_mask(mask, min_area_ratio=0.002):

        num_labels, labels, stats, _ = \
            cv2.connectedComponentsWithStats(mask)

        result = np.zeros_like(mask)

        min_area = int(
            mask.size * min_area_ratio
        )

        for i in range(1, num_labels):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area > min_area:

                result[
                    labels == i
                ] = 255

        return result

    cargo_mask = clean_mask(
        cargo_mask,
        0.002
    )

    wall_mask = clean_mask(
        wall_mask,
        0.002
    )

    # =====================================================
    # EMPTY MASK
    # =====================================================

    empty_mask = cv2.bitwise_and(
        wall_mask,
        cv2.bitwise_not(
            cargo_mask
        )
    )

    # =====================================================
    # PERSPECTIVE WEIGHT
    # =====================================================

    weights = np.linspace(
        1.5,
        0.6,
        rh
    ).reshape(
        rh,
        1
    )

    cargo_score = np.sum(
        (cargo_mask > 0).astype(np.float32)
        * weights
    )

    empty_score = np.sum(
        (empty_mask > 0).astype(np.float32)
        * weights
    )

    occupancy = (
        cargo_score /
        (
            cargo_score +
            empty_score +
            1e-6
        )
    )

    occupancy = np.clip(
        occupancy,
        0,
        1
    )

    # =====================================================
    # GAMMA CALIBRATION
    # =====================================================

    volume = (
        occupancy ** 0.85
    ) * 100

    volume = np.clip(
        volume,
        0,
        100
    )

    volume = int(
        round(volume / 5) * 5
    )

    # =====================================================
    # DEBUG
    # =====================================================

    debug = roi.copy()

    # GREEN = cargo
    debug[
        cargo_mask > 0
    ] = (
        0,
        255,
        0
    )

    # RED = empty
    debug[
        empty_mask > 0
    ] = (
        0,
        0,
        255
    )

    debug_overlay = cv2.addWeighted(
        roi,
        0.65,
        debug,
        0.35,
        0
    )

    cv2.imwrite(
        "debug_wall.jpg",
        wall_mask
    )

    cv2.imwrite(
        "debug_cargo.jpg",
        cargo_mask
    )

    cv2.imwrite(
        "debug_empty.jpg",
        empty_mask
    )

    cv2.imwrite(
        "debug_overlay.jpg",
        debug_overlay
    )

    print(
        f"VMEAN={v_mean:.1f} "
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
