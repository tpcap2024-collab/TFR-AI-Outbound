from flask import Flask, request, jsonify, send_file
import requests
import cv2
import numpy as np
import traceback
import threading
import os

app = Flask(__name__)


# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"


# =========================
# DEBUG CONFIG
# =========================
DEBUG_DIR = "/tmp"
os.makedirs(DEBUG_DIR, exist_ok=True)


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
        r = requests.get(
            url,
            timeout=15,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            print("IMAGE HTTP ERROR:", r.status_code)
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        return img

    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(filename, img):
    try:
        path = os.path.join(DEBUG_DIR, filename)
        ok = cv2.imwrite(path, img)
        print(f"SAVE DEBUG {filename}: {ok}")
        return ok

    except Exception as e:
        print("SAVE DEBUG ERROR:", filename, e)
        return False


# =========================
# CLEAN MASK
# =========================
def clean_mask(mask, min_area_ratio=0.002):
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


# =========================
# ADDED: BUILD CARGO ZONE
# จำกัดพื้นที่ให้ texture/dark ทำงานเฉพาะโซนวางของ
# =========================
def build_cargo_zone(shape, view_type):
    """
    สร้าง zone ที่น่าจะเป็นพื้นที่วางของจริง

    จุดประสงค์:
    - ลดการจับผนัง / หลังคา / ผนังข้างตู้เป็น cargo
    - ใช้จำกัดเฉพาะ dark_mask และ texture_mask
    - ไม่จำกัด blue/brown/green mask เพราะ cargo สีชัดอาจอยู่สูงได้
    """

    rh, rw = shape[:2]

    zone = np.zeros(
        (rh, rw),
        dtype=np.uint8
    )

    if view_type == "rear":
        # rear view: ของมักอยู่กลาง-ล่าง
        y1 = int(rh * 0.36)
        y2 = int(rh * 0.98)
        x1 = int(rw * 0.03)
        x2 = int(rw * 0.97)
    else:
        # side view: ของมักอยู่กลาง-ล่าง
        y1 = int(rh * 0.32)
        y2 = int(rh * 0.98)
        x1 = int(rw * 0.03)
        x2 = int(rw * 0.97)

    zone[y1:y2, x1:x2] = 255

    return zone


# =========================
# REMOVE WALL-LIKE COMPONENTS
# ลบ component ใหญ่ที่เหมือนผนังตู้
# =========================
def remove_wall_like_components(mask, roi_bgr, hsv):
    """
    Remove wall/background-like components from cargo mask safely.

    เงื่อนไขลบ:
    - component ใหญ่
    - กว้างมาก
    - แตะขอบบน
    - saturation เฉลี่ยต่ำ คล้ายผนัง
    - ไม่ได้ลงมาถึงขอบล่างมากเกินไป
    """

    if mask is None or mask.size == 0:
        return mask

    rh, rw = mask.shape[:2]

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        area_ratio = area / float(mask.size)
        width_ratio = ww / float(rw)
        height_ratio = hh / float(rh)

        component_mask = labels == i

        mean_s = float(np.mean(s_channel[component_mask]))
        mean_v = float(np.mean(v_channel[component_mask]))

        touches_top = y <= int(rh * 0.08)
        very_wide = width_ratio >= 0.65
        very_large = area_ratio >= 0.15
        tall_enough = height_ratio >= 0.30

        touches_bottom = (y + hh) >= int(rh * 0.88)

        low_saturation_wall = mean_s < 80
        not_dark_object = mean_v > 55

        is_wall_like = (
            touches_top and
            very_wide and
            very_large and
            tall_enough and
            low_saturation_wall and
            not_dark_object and
            not touches_bottom
        )

        if is_wall_like:
            print(
                "REMOVE WALL-LIKE:",
                f"x={x}, y={y}, w={ww}, h={hh}, "
                f"area={area_ratio:.3f}, "
                f"mean_s={mean_s:.1f}, mean_v={mean_v:.1f}"
            )
        else:
            result[component_mask] = 255

    return result


# =========================
# BALANCED VOLUME MODEL
# =========================
def gen_volume(img, debug=True, return_empty=False):

    if img is None or img.size == 0:
        return 0

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

    # =========================
    # SIDE VIEW
    # 4:3 -> 16:9
    # =========================
    if view_type == "side":

        h, w = img.shape[:2]

        target_h = int(w * 9 / 16)
        top = max(0, (h - target_h) // 2)

        img = img[top:top + target_h, :]

    h, w = img.shape[:2]

    print(
        f"VIEW={view_type} "
        f"SIZE={w}x{h}"
    )

    # =========================
    # ROI
    # =========================
    if view_type == "rear":

        roi = img[
            int(h * 0.18):int(h * 0.82),
            int(w * 0.15):int(w * 0.85)
        ]

    else:

        roi = img[
            int(h * 0.25):int(h * 0.75),
            int(w * 0.15):int(w * 0.85)
        ]

    if roi.size == 0:
        return 0

    rh, rw = roi.shape[:2]

    # =========================
    # CONTAINER MASK
    # =========================
    container_mask = np.ones(
        (rh, rw),
        dtype=np.uint8
    ) * 255

    # =========================
    # ADDED: CARGO ZONE
    # ใช้จำกัดเฉพาะ dark_mask และ texture_mask
    # =========================
    cargo_zone = build_cargo_zone(
        (rh, rw),
        view_type
    )

    # =========================
    # LIGHT NORMALIZATION
    # =========================
    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

