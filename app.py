import os
import cv2
import time
import numpy as np
import requests
import traceback
import threading

from flask import Flask, request, jsonify


app = Flask(__name__)


# =====================================================
# APPSHEET CONFIG
# แนะนำให้ตั้งค่าใน Environment Variables
# =====================================================
APP_ID = os.getenv("APPSHEET_APP_ID", "5ebec09a-62dd-4fa9-8f14-830fb104518f")
ACCESS_KEY = os.getenv("APPSHEET_ACCESS_KEY", "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp")
TABLE_NAME = os.getenv("APPSHEET_TABLE_NAME", "Data TFR")


# =====================================================
# REQUEST / PROCESS CONFIG
# =====================================================
REQUEST_TIMEOUT = 20
IMAGE_MAX_BYTES = 8 * 1024 * 1024       # จำกัดรูปไม่เกิน 8 MB
DUPLICATE_TTL_SEC = 10 * 60             # กัน duplicate 10 นาที


# =====================================================
# DUPLICATE LOCK WITH TTL
# =====================================================
processed_ids = {}
lock = threading.Lock()


def cleanup_processed_ids():
    """
    ลบ row_id เก่าที่หมดอายุ เพื่อไม่ให้ memory โตเรื่อย ๆ
    """
    now = time.time()

    expired_keys = [
        row_id
        for row_id, timestamp in processed_ids.items()
        if now - timestamp > DUPLICATE_TTL_SEC
    ]

    for row_id in expired_keys:
        processed_ids.pop(row_id, None)


def is_duplicate(row_id):
    """
    ตรวจ duplicate request แบบ thread-safe
    """
    with lock:
        cleanup_processed_ids()

        if row_id in processed_ids:
            return True

        processed_ids[row_id] = time.time()
        return False


# =====================================================
# DOWNLOAD IMAGE
# =====================================================
def download_image(url):
    """
    Download image from URL safely.
    Return OpenCV image or None.
    """
    try:
        if not isinstance(url, str):
            print("Invalid URL type")
            return None

        if not url.startswith(("http://", "https://")):
            print("Invalid URL scheme")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )

        if response.status_code != 200:
            print(f"Image download failed: HTTP {response.status_code}")
            return None

        content_type = response.headers.get("Content-Type", "").lower()

        if "image" not in content_type and len(response.content) < 200:
            print(f"Invalid content type: {content_type}")
            return None

        if len(response.content) > IMAGE_MAX_BYTES:
            print("Image too large")
            return None

        img_array = np.frombuffer(response.content, np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("OpenCV decode failed")
            return None

        if img.size == 0:
            print("Empty image")
            return None

        return img

    except requests.exceptions.Timeout:
        print("Image download timeout")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Image download request error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected image download error: {e}")
        return None


# =====================================================
# VOLUME MODEL
# =====================================================
def gen_volume(img, debug=False):
    """
    Estimate container occupancy volume percentage from image.

    Output:
        int volume percentage rounded to nearest 5%
    """

    if img is None:
        raise ValueError("Input image is None")

    if img.size == 0:
        raise ValueError("Input image is empty")

    # =====================================================
    # RESIZE & CROP
    # =====================================================
    img = cv2.resize(img, (640, 480))

    h, w = img.shape[:2]

    # remove watermark area
    img = img[:int(h * 0.92), :]

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size after crop")

    roi = img[
        int(h * 0.08):int(h * 0.90),
        int(w * 0.05):int(w * 0.95)
    ]

    if roi.size == 0:
        raise ValueError("ROI is empty")

    rh, rw = roi.shape[:2]

    # =====================================================
    # LIGHT NORMALIZATION
    # =====================================================
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # =====================================================
    # COLOR SPACE
    # =====================================================
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # WALL MASK
    # =====================================================
    v_mean = float(np.mean(hsv[:, :, 2]))

    wall_lower = np.array(
        [0, 0, max(60, int(v_mean * 0.5))],
        dtype=np.uint8
    )

    wall_upper = np.array(
        [180, 60, min(255, int(v_mean * 1.2))],
        dtype=np.uint8
    )

    wall_mask = cv2.inRange(
        hsv,
        wall_lower,
        wall_upper
    )

    # =====================================================
    # CARGO MASK
    # =====================================================
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 40, 40], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8)
    )

    brown_mask = cv2.inRange(
        hsv,
        np.array([5, 40, 40], dtype=np.uint8),
        np.array([30, 255, 255], dtype=np.uint8)
    )

    _, dark_mask = cv2.threshold(
        gray,
        70,
        255,
        cv2.THRESH_BINARY_INV
    )

    edges = cv2.Canny(
        gray,
        80,
        150
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    cargo_mask = cv2.bitwise_or(green_mask, brown_mask)
    cargo_mask = cv2.bitwise_or(cargo_mask, dark_mask)
    cargo_mask = cv2.bitwise_or(cargo_mask, edges)

    # =====================================================
    # MORPHOLOGY
    # =====================================================
    kernel_big = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        kernel_big,
        iterations=2
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        kernel_small,
        iterations=2
    )

    # =====================================================
    # CLEAN MASK
    # =====================================================
    def clean_mask(mask, ratio=0.002):
        """
        Remove small noise blobs.
        """
        if mask is None or mask.size == 0:
            return np.zeros_like(mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        result = np.zeros_like(mask)

        min_area = int(mask.size * ratio)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area > min_area:
                result[labels == i] = 255

        return result

    cargo_mask = clean_mask(cargo_mask, 0.002)
    wall_mask = clean_mask(wall_mask, 0.002)

    # =====================================================
    # EMPTY MASK
    # =====================================================
    empty_mask = cv2.bitwise_and(
        wall_mask,
        cv2.bitwise_not(cargo_mask)
    )

    # =====================================================
    # PERSPECTIVE WEIGHT
    # =====================================================
    y = np.linspace(0, 1, rh)

    weights = 0.5 + (y ** 1.8) * 1.5
    weights = weights.reshape(rh, 1).astype(np.float32)

    cargo_score = np.sum(
        (cargo_mask > 0).astype(np.float32) * weights
    )

    empty_score = np.sum(
        (empty_mask > 0).astype(np.float32) * weights
    )

    total_score = cargo_score + empty_score

    if total_score <= 1e-6:
        print("Warning: total_score is zero. Return 0%.")
        return 0

    occupancy = cargo_score / total_score
    occupancy = float(np.clip(occupancy, 0, 1))

    # =====================================================
    # CALIBRATION
    # =====================================================
    volume = (occupancy ** 0.9) * 100
    volume = volume * 0.95 + 2
    volume = float(np.clip(volume, 0, 100))

    volume = int(round(volume / 5) * 5)

    # =====================================================
    # DEBUG OUTPUT
    # =====================================================
    if debug:
        debug_img = roi.copy()

        debug_img[cargo_mask > 0] = (0, 255, 0)
        debug_img[empty_mask > 0] = (0, 0, 255)

        overlay = cv2.addWeighted(
            roi,
            0.65,
            debug_img,
            0.35,
            0
        )

        cv2.imwrite("debug_overlay.jpg", overlay)
        cv2.imwrite("debug_cargo.jpg", cargo_mask)
        cv2.imwrite("debug_empty.jpg", empty_mask)
        cv2.imwrite("debug_wall.jpg", wall_mask)

    print(
        f"VMEAN={v_mean:.1f} "
        f"OCC={occupancy:.3f} "
        f"VOL={volume}%"
    )

    return volume


# =====================================================
# UPDATE APPSHEET
# =====================================================
def update_appsheet(row_id, volume_text):
    """
    Update AppSheet row.
    Return True if success, False otherwise.
    """

    if not APP_ID or not ACCESS_KEY or ACCESS_KEY == "PUT_YOUR_ACCESS_KEY_HERE":
        print("Missing AppSheet config")
        return False

    url = (
        f"https://api.appsheet.com/api/v2/apps/"
        f"{APP_ID}/tables/{TABLE_NAME}/Action"
    )

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
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code not in (200, 201):
            print(
                "AppSheet update failed:",
                response.status_code,
                response.text[:500]
            )
            return False

        print("AppSheet update success")
        return True

    except requests.exceptions.Timeout:
        print("AppSheet update timeout")
        return False

    except requests.exceptions.RequestException as e:
        print(f"AppSheet request error: {e}")
        return False

    except Exception as e:
        print(f"Unexpected AppSheet error: {e}")
        return False


# =====================================================
# HEALTH CHECK
# =====================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "volume-ai"
        }
    ), 200


# =====================================================
# API ENDPOINT
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
        "id": "row_id",
        "link": "image_url"
    }
    """

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify(
                {
                    "status": "error",
                    "message": "invalid or missing json"
                }
            ), 400

        image_url = data.get("link")
        row_id = data.get("id")

        if not image_url or not row_id:
            return jsonify(
                {
                    "status": "error",
                    "message": "missing id or link"
                }
            ), 400

        row_id = str(row_id).strip()
        image_url = str(image_url).strip()

        if is_duplicate(row_id):
            return jsonify(
                {
                    "status": "skipped",
                    "message": "duplicate request",
                    "id": row_id
                }
            ), 200

        img = download_image(image_url)

        if img is None:
            return jsonify(
                {
                    "status": "error",
                    "message": "image download or decode failed",
                    "id": row_id
                }
            ), 400

        volume = gen_volume(
            img,
            debug=False
        )

        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        appsheet_ok = update_appsheet(
            row_id,
            volume_text
        )

        if not appsheet_ok:
            return jsonify(
                {
                    "status": "partial_success",
                    "message": "volume calculated but AppSheet update failed",
                    "id": row_id,
                    "volume": volume_text
                }
            ), 207

        return jsonify(
            {
                "status": "success",
                "id": row_id,
                "volume": volume_text
            }
        ), 200

    except Exception:
        print(traceback.format_exc())

        return jsonify(
            {
                "status": "error",
                "message": "server error"
            }
        ), 500


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True
    )
