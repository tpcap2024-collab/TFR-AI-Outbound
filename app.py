import os
import cv2
import time
import numpy as np
import requests
import traceback
import threading

from flask import Flask, request, jsonify, send_file


app = Flask(__name__)


# =====================================================
# APPSHEET CONFIG
# แนะนำให้ตั้งค่าเป็น Environment Variables
# =====================================================
APP_ID = os.getenv("APPSHEET_APP_ID", "5ebec09a-62dd-4fa9-8f14-830fb104518f")
ACCESS_KEY = os.getenv("APPSHEET_ACCESS_KEY", "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp")
TABLE_NAME = os.getenv("APPSHEET_TABLE_NAME", "Data TFR")


# =====================================================
# CONFIG
# =====================================================
REQUEST_TIMEOUT = 20
IMAGE_MAX_BYTES = 10 * 1024 * 1024
DUPLICATE_TTL_SEC = 10 * 60

DEBUG_DIR = os.getenv("DEBUG_DIR", "/tmp")
os.makedirs(DEBUG_DIR, exist_ok=True)


# =====================================================
# DUPLICATE LOCK WITH TTL
# =====================================================
processed_ids = {}
lock = threading.Lock()


def cleanup_processed_ids():
    now = time.time()

    expired = [
        row_id
        for row_id, ts in processed_ids.items()
        if now - ts > DUPLICATE_TTL_SEC
    ]

    for row_id in expired:
        processed_ids.pop(row_id, None)


def is_duplicate(row_id):
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
    Download image safely and decode to OpenCV BGR image.
    """

    try:
        if not isinstance(url, str):
            print("Invalid URL type")
            return None

        url = url.strip()

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

        if len(response.content) == 0:
            print("Empty response content")
            return None

        if len(response.content) > IMAGE_MAX_BYTES:
            print("Image too large")
            return None

        img_array = np.frombuffer(response.content, np.uint8)

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        if img is None:
            print("OpenCV decode failed")
            return None

        if img.size == 0:
            print("Decoded image is empty")
            return None

        return img

    except requests.exceptions.Timeout:
        print("Image download timeout")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Image request error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected image download error: {e}")
        return None


# =====================================================
# UTILS
# =====================================================
def clean_mask(mask, min_area_ratio=0.002):
    """
    Remove small connected components.
    """

    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)

    min_area = int(mask.size * min_area_ratio)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            result[labels == i] = 255

    return result


def save_debug_image(filename, image):
    """
    Save debug image to DEBUG_DIR.
    """

    path = os.path.join(DEBUG_DIR, filename)

    ok = cv2.imwrite(path, image)

    print(f"Save debug {filename}: {ok}")

    return ok


# =====================================================
# VOLUME MODEL
# =====================================================
def gen_volume(img, debug=False, return_empty=False):
    """
    Estimate volume from container image.

    Default:
        return filled volume %

    If return_empty=True:
        return empty space %

    Main logic:
        1. Create container_mask = full inner container ROI
        2. Create conservative cargo_mask
        3. empty_mask = container_mask - cargo_mask
        4. Calculate from full container_mask
    """

    if img is None:
        raise ValueError("Input image is None")

    if img.size == 0:
        raise ValueError("Input image is empty")

    # =====================================================
    # RESIZE
    # =====================================================
    img = cv2.resize(
        img,
        (640, 480)
    )

    h, w = img.shape[:2]

    # =====================================================
    # REMOVE WATERMARK AREA
    # =====================================================
    img = img[
        :int(h * 0.92),
        :
    ]

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size after watermark crop")

    # =====================================================
    # ROI: พื้นที่ภายในตู้โดยประมาณ
    # =====================================================
    roi = img[
        int(h * 0.08):int(h * 0.90),
        int(w * 0.05):int(w * 0.95)
    ]

    if roi.size == 0:
        raise ValueError("ROI is empty")

    rh, rw = roi.shape[:2]

    # =====================================================
    # CONTAINER MASK
    # ใช้ ROI ทั้งหมดเป็นพื้นที่ภายในตู้
    # =====================================================
    container_mask = np.ones(
        (rh, rw),
        dtype=np.uint8
    ) * 255

    # =====================================================
    # LIGHT NORMALIZATION
    # =====================================================
    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        [l, a, b]
    )

    roi_norm = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # =====================================================
    # COLOR SPACE
    # =====================================================
    hsv = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2GRAY
    )

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    v_mean = float(np.mean(v_channel))
    s_mean = float(np.mean(s_channel))

    # =====================================================
    # CONSERVATIVE CARGO MASK
    # =====================================================

    # GREEN PALLET / GREEN OBJECT
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 45, 45], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8)
    )

    # BROWN CARTON / WOODEN PALLET
    brown_mask = cv2.inRange(
        hsv,
        np.array([5, 45, 45], dtype=np.uint8),
        np.array([35, 255, 230], dtype=np.uint8)
    )

    # DARK CARGO
    # ต้องมืดและมี saturation พอสมควร เพื่อลดการจับเงาเป็นสินค้า
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 35, 0], dtype=np.uint8),
        np.array([180, 255, 70], dtype=np.uint8)
    )

    # =====================================================
    # TEXTURE MASK แบบ conservative
    # =====================================================
    blur_gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    adaptive_texture = cv2.adaptiveThreshold(
        blur_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    saturation_mask = cv2.inRange(
        s_channel,
        45,
        255
    )

    low_value_mask = cv2.inRange(
        v_channel,
        0,
        90
    )

    texture_candidate = cv2.bitwise_or(
        saturation_mask,
        low_value_mask
    )

    texture_mask = cv2.bitwise_and(
        adaptive_texture,
        texture_candidate
    )

    # =====================================================
    # COMBINE CARGO MASK
    # =====================================================
    cargo_mask = cv2.bitwise_or(
        green_mask,
        brown_mask
    )

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        dark_mask
    )

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        texture_mask
    )

    cargo_mask = cv2.bitwise_and(
        cargo_mask,
        container_mask
    )

    # =====================================================
    # MORPHOLOGY
    # =====================================================
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    cargo_mask = clean_mask(
        cargo_mask,
        min_area_ratio=0.003
    )

    # =====================================================
    # LIMIT OVER-DETECTION
    # =====================================================
    raw_cargo_ratio = np.sum(cargo_mask > 0) / float(container_mask.size)

    if raw_cargo_ratio > 0.98:
        print("Warning: cargo over-detected. Fallback to color-only mask.")

        cargo_mask = cv2.bitwise_or(
            green_mask,
            brown_mask
        )

        cargo_mask = cv2.bitwise_or(
            cargo_mask,
            dark_mask
        )

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_CLOSE,
            kernel_medium,
            iterations=1
        )

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_OPEN,
            kernel_small,
            iterations=1
        )

        cargo_mask = clean_mask(
            cargo_mask,
            min_area_ratio=0.003
        )

    # =====================================================
    # EMPTY MASK = CONTAINER - CARGO
    # =====================================================
    empty_mask = cv2.bitwise_and(
        container_mask,
        cv2.bitwise_not(cargo_mask)
    )

    # =====================================================
    # PERSPECTIVE WEIGHT
    # =====================================================
    y = np.linspace(
        0,
        1,
        rh
    )

    weights = 0.65 + (y ** 1.6) * 1.35
    weights = weights.reshape(
        rh,
        1
    ).astype(np.float32)

    container_score = np.sum(
        (container_mask > 0).astype(np.float32) * weights
    )

    cargo_score = np.sum(
        (cargo_mask > 0).astype(np.float32) * weights
    )

    empty_score = np.sum(
        (empty_mask > 0).astype(np.float32) * weights
    )

    if container_score <= 1e-6:
        print("Warning: container_score is zero")
        return 0

    filled_ratio = cargo_score / container_score
    empty_ratio = empty_score / container_score

    filled_ratio = float(
        np.clip(
            filled_ratio,
            0,
            1
        )
    )

    empty_ratio = float(
        np.clip(
            empty_ratio,
            0,
            1
        )
    )

    # =====================================================
    # CALIBRATION
    # =====================================================
    filled_volume = (filled_ratio ** 0.95) * 100

    # ลด bias เล็กน้อย ไม่ให้ overestimate
    filled_volume = filled_volume * 0.95

    filled_volume = float(
        np.clip(
            filled_volume,
            0,
            100
        )
    )

    empty_volume = 100 - filled_volume

    if return_empty:
        output_volume = empty_volume
    else:
        output_volume = filled_volume

    output_volume = int(
        round(output_volume / 5) * 5
    )

    output_volume = int(
        np.clip(
            output_volume,
            0,
            100
        )
    )

    # =====================================================
    # DEBUG OUTPUT
    # =====================================================
    if debug:
        debug_img = roi_norm.copy()

        # GREEN = cargo
        debug_img[cargo_mask > 0] = (
            0,
            255,
            0
        )

        # RED = empty
        debug_img[empty_mask > 0] = (
            0,
            0,
            255
        )

        overlay = cv2.addWeighted(
            roi_norm,
            0.65,
            debug_img,
            0.35,
            0
        )

        save_debug_image(
            "debug_original.jpg",
            roi
        )

        save_debug_image(
            "debug_normalized.jpg",
            roi_norm
        )

        save_debug_image(
            "debug_container.jpg",
            container_mask
        )

        save_debug_image(
            "debug_cargo.jpg",
            cargo_mask
        )

        save_debug_image(
            "debug_empty.jpg",
            empty_mask
        )

        save_debug_image(
            "debug_overlay.jpg",
            overlay
        )

        save_debug_image(
            "debug_green.jpg",
            green_mask
        )

        save_debug_image(
            "debug_brown.jpg",
            brown_mask
        )

        save_debug_image(
            "debug_dark.jpg",
            dark_mask
        )

        save_debug_image(
            "debug_texture.jpg",
            texture_mask
        )

    print(
        f"VMEAN={v_mean:.1f} "
        f"SMEAN={s_mean:.1f} "
        f"FILLED_RATIO={filled_ratio:.3f} "
        f"EMPTY_RATIO={empty_ratio:.3f} "
        f"FILLED_VOL={filled_volume:.1f}% "
        f"RETURN={output_volume}% "
        f"MODE={'EMPTY' if return_empty else 'FILLED'}"
    )

    return output_volume


# =====================================================
# UPDATE APPSHEET
# =====================================================
def update_appsheet(row_id, volume_text):
    """
    Update AppSheet row.
    """

    if not APP_ID:
        print("Missing APPSHEET_APP_ID")
        return False

    if not ACCESS_KEY:
        print("Missing APPSHEET_ACCESS_KEY")
        return False

    if not TABLE_NAME:
        print("Missing APPSHEET_TABLE_NAME")
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
            "service": "container-volume-ai"
        }
    ), 200


# =====================================================
# DEBUG BROWSER ENDPOINT
# =====================================================
@app.route("/debug/<filename>", methods=["GET"])
def get_debug_file(filename):
    """
    Open debug image in browser.

    Example:
        /debug/debug_overlay.jpg
        /debug/debug_cargo.jpg
        /debug/debug_empty.jpg
    """

    allowed_files = {
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    }

    if filename not in allowed_files:
        return jsonify(
            {
                "status": "error",
                "message": "file not allowed"
            }
        ), 403

    path = os.path.join(
        DEBUG_DIR,
        filename
    )

    if not os.path.exists(path):
        return jsonify(
            {
                "status": "error",
                "message": "debug file not found",
                "filename": filename
            }
        ), 404

    return send_file(
        path,
        mimetype="image/jpeg"
    )


@app.route("/debug-list", methods=["GET"])
def debug_list():
    """
    List debug files and browser URLs.
    """

    files = [
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    ]

    base_url = request.host_url.rstrip("/")

    return jsonify(
        {
            "status": "ok",
            "debug_files": [
                {
                    "file": f,
                    "url": f"{base_url}/debug/{f}",
                    "exists": os.path.exists(
                        os.path.join(DEBUG_DIR, f)
                    )
                }
                for f in files
            ]
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
        "link": "image_url",

        optional:
        "debug": true,
        "return_empty": false
    }

    return_empty:
        false = return filled volume %
        true  = return empty space %
    """

    try:
        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):
            return jsonify(
                {
                    "status": "error",
                    "message": "invalid or missing json"
                }
            ), 400

        image_url = data.get("link")
        row_id = data.get("id")

        debug = bool(
            data.get(
                "debug",
                True
            )
        )

        return_empty = bool(
            data.get(
                "return_empty",
                False
            )
        )

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

        img = download_image(
            image_url
        )

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
            debug=debug,
            return_empty=return_empty
        )

        volume_text = f"{volume}%"

        print("VOLUME:", volume_text)

        appsheet_ok = update_appsheet(
            row_id,
            volume_text
        )

