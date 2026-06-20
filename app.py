from flask import Flask, request, jsonify, send_file
import requests
import cv2
import numpy as np
import traceback
import time
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
                "User-At": "Mozilla/5.0"
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
# BALANCED VOLUME MODEL
# =========================
def _volume(img, debug=True, return_empty=False):

    if img is None or img.size == 0:
        return 0

    # =========================
    # DETECT VIEW TYPE
    # =========================
    orig_h, orig_w = img.shape[:2]
    view_type = "rear" if orig_h > orig_w else "side"

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
    # ใช้ ROI ทั้งหมดเป็นพื้นที่ภายในตู้
    # =========================
    container_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    # =========================
    # LIGHT NORMALIZATION
    # =========================
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

    roi_norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    # =========================
    # COLOR SPACE
    # =========================
    hsv = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2GRAY
    )

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    h_channel, s_channel, v_channel = cv2.split(hsv)

    v_mean = float(v_channel.mean())
    s_mean = float(s_channel.mean())

    # =========================
    # CARGO COLOR MASKS
    # =========================

    # GREEN PALLET / GREEN OBJECT
    green_mask = cv2.inRange(
        hsv,
        (35, 45, 45),
        (95, 255, 255)
    )

    # BROWN CARTON / WOOD / PALLET
    brown_mask = cv2.inRange(
        hsv,
        (5, 45, 45),
        (35, 255, 230)
    )

    # BLUE / CYAN CRATE
    blue_mask = cv2.inRange(
        hsv,
        (85, 35, 35),
        (125, 255, 255)
    )

    # RED CRATE / RED CARGO
    # สีแดงใน HSV ต้องแยก 2 ช่วง เพราะ Hue อยู่ทั้งต้นและท้ายวงสี
    red_mask_1 = cv2.inRange(
        hsv,
        (0, 60, 50),
        (12, 255, 255)
    )

    red_mask_2 = cv2.inRange(
        hsv,
        (165, 60, 50),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    # DARK CARGO
    # ปรับเข้มขึ้นเพื่อลดการจับเงา / ผ้า / พื้นมืด
    dark_mask = cv2.inRange(
        hsv,
        (0, 55, 0),
        (180, 255, 65)
    )

    # =========================
    # TEXTURE MASK - CONSERVATIVE
    # =========================
    adaptive_texture = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # ใช้ AND แทน OR เพื่อลดการจับผนัง / ผ้า / เพดาน
    strong_saturation_mask = cv2.inRange(
        s_channel,
        70,
        255
    )

    strong_low_value_mask = cv2.inRange(
        v_channel,
        0,
        75
    )

    texture_candidate = cv2.bitwise_and(
        strong_saturation_mask,
        strong_low_value_mask
    )

    texture_mask = cv2.bitwise_and(
        adaptive_texture,
        texture_candidate
    )

    # =========================
    # EDGE DENSITY FILTER
    # ลด false positive จากผ้า / ผิวเรียบ / เพดาน
    # =========================
    edges = cv2.Canny(
        gray_blur,
        40,
        120
    )

    edge_density = cv2.blur(
        edges.astype(np.float32),
        (15, 15)
    )

    edge_mask = cv2.inRange(
        edge_density,
        10,
        255
    )

    texture_mask = cv2.bitwise_and(
        texture_mask,
        edge_mask
    )

    # =========================
    # TOP FALSE POSITIVE SUPPRESSION
    # ไม่ตัด color cargo ด้านบนทั้งหมด เพราะบางภาพมีกล่องจริงอยู่ด้านบน
    # ตัดเฉพาะ texture/dark ที่มักติดเพดานหรือผ้า
    # =========================
    top_suppress_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    top_cut_ratio = 0.12 if view_type == "rear" else 0.16
    top_cut = int(rh * top_cut_ratio)

    top_suppress_mask[:top_cut, :] = 0

    texture_mask = cv2.bitwise_and(
        texture_mask,
        top_suppress_mask
    )

    dark_mask = cv2.bitwise_and(
        dark_mask,
        top_suppress_mask
    )

    # =========================
    # COMBINE COLOR MASKS
    # =========================
    color_cargo_mask = cv2.bitwise_or(
        green_mask,
        brown_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        blue_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        red_mask
    )

    # =========================
    # COMBINE CARGO
    # =========================
    cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
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

    # =========================
    # MORPHOLOGY
    # =========================
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

    # ลด noise จุดเล็ก
    cargo_mask = clean_mask(
        cargo_mask,
        min_area_ratio=0.005
    )

    # =========================
    # CONTOUR FILTER
    # กรอง noise เล็ก ๆ และ shape ที่ไม่น่าเป็น cargo
    # =========================
    contours, _ = cv2.findContours(
        cargo_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_mask = np.zeros_like(cargo_mask)

    min_contour_area = rh * rw * 0.0035

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_contour_area:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)

        if h_box <= 0 or w_box <= 0:
            continue

        aspect_ratio = w_box / float(h_box)

        # ช่วงกว้างขึ้นเพื่อไม่ตัดกล่องที่เรียงยาวหลายใบ
        if 0.20 <= aspect_ratio <= 6.50:
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                thickness=-1
            )

    cargo_mask = filtered_mask

    # =========================
    # FALLBACK ถ้า cargo กินภาพเยอะผิดปกติ
    # =========================
    raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    if raw_cargo_ratio > 0.95:
        print("WARNING: cargo over-detected, fallback to color only")

        cargo_mask = color_cargo_mask.copy()

        cargo_mask = cv2.bitwise_or(
            cargo_mask,
            dark_mask
        )

        cargo_mask = cv2.bitwise_and(
            cargo_mask,
            container_mask
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
            min_area_ratio=0.005
        )

        raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    # =========================
    # EMPTY MASK = CONTAINER - CARGO
    # =========================
    empty_mask = cv2.bitwise_and(
        container_mask,
        cv2.bitwise_not(cargo_mask)
    )

    # =========================
    # PERSPECTIVE WEIGHT
    # =========================
    y = np.linspace(
        0,
        1,
        rh,
        dtype=np.float32
    ).reshape(rh, 1)

    if view_type == "rear":
        weights = 0.70 + (y ** 1.5) * 1.30
    else:
        weights = 0.80 + (y ** 1.3) * 1.10

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

    # =========================
    # CALIBRATION
    # =========================
    filled_volume = (filled_ratio ** 0.95) * 100
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

    output_volume = int(round(output_volume / 5) * 5)
    output_volume = max(0, min(100, output_volume))

    print(
        f"VIEW={view_type} "
        f"VMEAN={v_mean:.1f} "
        f"SMEAN={s_mean:.1f} "
        f"RAW_CARGO={raw_cargo_ratio:.3f} "
        f"FILLED_RATIO={filled_ratio:.3f} "
        f"EMPTY_RATIO={empty_ratio:.3f} "
        f"RETURN={output_volume}% "
        f"MODE={'EMPTY' if return_empty else 'FILLED'}"
    )

    # =========================
    # DEBUG OUTPUT
    # =========================
    if debug:

        color_layer = roi_norm.copy()

        # GREEN = cargo
        color_layer[cargo_mask > 0] = (
            0,
            255,
            0
        )

        # BLUE = empty
        # เปลี่ยนจากสีแดงเป็นสีน้ำเงิน เพื่อไม่ให้กลืนกับลังแดงจริง
        color_layer[empty_mask > 0] = (
            255,
            0,
            0
        )

        overlay = cv2.addWeighted(
            roi_norm,
            0.85,
            color_layer,
            0.15,
            0
        )

        cargo_contours, _ = cv2.findContours(
            cargo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # YELLOW contour = cargo
        cv2.drawContours(
            overlay,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

        empty_contours, _ = cv2.findContours(
            empty_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # BLUE contour = empty
        cv2.drawContours(
            overlay,
            empty_contours,
            -1,
            (255, 0, 0),
            1
        )

        overlay_light = cv2.addWeighted(
            roi_norm,
            0.92,
            color_layer,
            0.08,
            0
        )

        overlay_contour = roi_norm.copy()

        cv2.drawContours(
            overlay_contour,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

        cv2.drawContours(
            overlay_contour,
            empty_contours,
            -1,
            (255, 0, 0),
            1
        )

        save_debug("debug_original.jpg", roi)
        save_debug("debug_normalized.jpg", roi_norm)
        save_debug("debug_container.jpg", container_mask)
        save_debug("debug_cargo.jpg", cargo_mask)
        save_debug("debug_empty.jpg", empty_mask)
        save_debug("debug_overlay.jpg", overlay)
        save_debug("debug_overlay_light.jpg", overlay_light)
        save_debug("debug_overlay_contour.jpg", overlay_contour)

        save_debug("debug_green.jpg", green_mask)
        save_debug("debug_brown.jpg", brown_mask)
        save_debug("debug_blue.jpg", blue_mask)
        save_debug("debug_dark.jpg", dark_mask)
        save_debug("debug_texture.jpg", texture_mask)

    return output_volume

# =========================
# GEN PALLET (INBOUND)
# DYNAMIC PALLET DETECTION
# - Detect cream cage/pallet dynamically inside main cream block
# - Detect green rack/pallet outside cream block
# - Prevent overlap NMS
# - Debug image only, uses existing debug filenames
# =========================
def gen_pallet(img, debug=True):

    if img is None or img.size == 0:
        return 0

    # =========================
    # LOCAL HELPERS
    # =========================
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def smooth_1d(arr, k=31):
        arr = arr.astype(np.float32)

        if arr.size == 0:
            return arr

        if k % 2 == 0:
            k += 1

        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr, kernel, mode="same")

    def box_iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih
        union = aw * ah + bw * bh - inter

        if union <= 0:
            return 0.0

        return inter / float(union)

    def nms(candidates, iou_threshold=0.18):
        if not candidates:
            return []

        candidates = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True
        )

        selected = []

        for cand in candidates:
            keep = True

            for old in selected:
                if box_iou(cand["box"], old["box"]) > iou_threshold:
                    keep = False
                    break

            if keep:
                selected.append(cand)

        return selected

    def find_dense_region(proj, min_len, threshold_ratio=0.18, prefer_right=False):
        if proj is None or len(proj) == 0:
            return None

        proj_smooth = smooth_1d(proj, k=31)

        max_val = float(np.max(proj_smooth))

        if max_val <= 1e-6:
            return None

        threshold = max_val * threshold_ratio
        high = proj_smooth >= threshold

        runs = []
        start = None

        for i, flag in enumerate(high):
            if flag and start is None:
                start = i

            elif not flag and start is not None:
                runs.append((start, i - 1))
                start = None

        if start is not None:
            runs.append((start, len(high) - 1))

        candidates = []

        for s, e in runs:
            length = e - s + 1

            if length < min_len:
                continue

            strength = float(np.sum(proj_smooth[s:e + 1]))
            center = (s + e) / 2.0
            score = strength

            if prefer_right:
                score = score * (1.0 + center / float(len(proj_smooth)))

            candidates.append(
                {
                    "s": s,
                    "e": e,
                    "length": length,
                    "strength": strength,
                    "score": score
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["score"]
        )

        return best["s"], best["e"]

    def find_line_positions(mask, axis, min_gap, threshold_ratio=0.24):
        if axis == 0:
            proj = np.sum(mask > 0, axis=0).astype(np.float32)
        else:
            proj = np.sum(mask > 0, axis=1).astype(np.float32)

        if proj.size == 0:
            return []

        proj = smooth_1d(proj, k=17)

        max_val = float(np.max(proj))

        if max_val <= 1e-6:
            return []

        threshold = max_val * threshold_ratio

        raw_lines = []
        start = None

        for i, val in enumerate(proj):
            if val >= threshold and start is None:
                start = i

            elif val < threshold and start is not None:
                end = i - 1
                segment = proj[start:end + 1]

                if segment.size > 0:
                    peak = int(np.argmax(segment))
                    raw_lines.append(start + peak)

                start = None

        if start is not None:
            end = len(proj) - 1
            segment = proj[start:end + 1]

            if segment.size > 0:
                peak = int(np.argmax(segment))
                raw_lines.append(start + peak)

        if not raw_lines:
            return []

        raw_lines = sorted(raw_lines)

        merged = []
        group = [raw_lines[0]]

        for p in raw_lines[1:]:
            if abs(p - group[-1]) <= min_gap:
                group.append(p)
            else:
                merged.append(int(sum(group) / len(group)))
                group = [p]

        if group:
            merged.append(int(sum(group) / len(group)))

        return sorted(merged)

    def normalize_grid_lines(lines, length, edge_ratio=0.12):
        lines = sorted(list(lines))

        if len(lines) == 0:
            return [0, length - 1]

        if lines[0] > length * edge_ratio:
            lines.insert(0, 0)

        if lines[-1] < length * (1.0 - edge_ratio):
            lines.append(length - 1)

        return sorted(lines)

    def remove_close_lines(lines, min_cell_size):
        if not lines:
            return []

        lines = sorted(lines)
        result = [lines[0]]

        for line in lines[1:]:
            if line - result[-1] >= min_cell_size:
                result.append(line)

        return result

    # =========================
    # RESIZE
    # =========================
    img = cv2.resize(img, (960, 720))
    h, w = img.shape[:2]

    # =========================
    # ROI
    # ตัดท้องฟ้า / ล้อ / พื้นถนน
    # เหลือเฉพาะพื้นที่สินค้าบนรถ
    # =========================
    y1 = int(h * 0.30)
    y2 = int(h * 0.75)
    x1 = int(w * 0.02)
    x2 = int(w * 0.97)

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return 0

    rh, rw = roi.shape[:2]

    # =========================
    # LIGHT NORMALIZATION
    # =========================
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

    roi_norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    hsv = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2GRAY
    )

    # =========================
    # CREAM / BEIGE MASK
    # จับกรงครีม / ลัง / ไม้ / น้ำตาลอ่อน
    # =========================
    cream_mask_1 = cv2.inRange(
        hsv,
        (5, 18, 50),
        (45, 210, 255)
    )

    cream_mask_2 = cv2.inRange(
        hsv,
        (0, 0, 95),
        (50, 110, 245)
    )

    cream_mask = cv2.bitwise_or(
        cream_mask_1,
        cream_mask_2
    )

    # ลบ label / กระดาษขาวจัด
    white_mask = cv2.inRange(
        hsv,
        (0, 0, 190),
        (180, 55, 255)
    )

    cream_mask = cv2.bitwise_and(
        cream_mask,
        cv2.bitwise_not(white_mask)
    )

    # =========================
    # GREEN MASK
    # =========================
    green_mask = cv2.inRange(
        hsv,
        (35, 35, 35),
        (95, 255, 255)
    )

    # =========================
    # CLEAN MASKS
    # =========================
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    kernel_mid = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    cream_clean = cv2.morphologyEx(
        cream_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    cream_clean = cv2.morphologyEx(
        cream_clean,
        cv2.MORPH_CLOSE,
        kernel_mid,
        iterations=1
    )

    green_clean = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2
    )

    green_clean = cv2.morphologyEx(
        green_clean,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # =========================
    # FIND MAIN CREAM BLOCK
    # หา block หลักของกรงครีม ไม่เอาช่องว่างซ้าย/หลังคา
    # =========================
    cream_for_block = cream_clean.copy()

    cream_for_block[:int(rh * 0.03), :] = 0
    cream_for_block[int(rh * 0.96):, :] = 0

    cream_for_block = cv2.morphologyEx(
        cream_for_block,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (23, 13)),
        iterations=2
    )

    x_proj = np.sum(
        cream_for_block > 0,
        axis=0
    ).astype(np.float32)

    y_proj = np.sum(
        cream_for_block > 0,
        axis=1
    ).astype(np.float32)

    x_region = find_dense_region(
        x_proj,
        min_len=int(rw * 0.35),
        threshold_ratio=0.18,
        prefer_right=True
    )

    y_region = find_dense_region(
        y_proj,
        min_len=int(rh * 0.45),
        threshold_ratio=0.16,
        prefer_right=False
    )

    main_box = None

    if x_region is not None and y_region is not None:
        bx1, bx2 = x_region
        by1, by2 = y_region

        bx = bx1
        by = by1
        bw = bx2 - bx1 + 1
        bh = by2 - by1 + 1

        # ถ้า block เริ่มซ้ายเกินไป ให้ดึงเข้าขวา
        # เพื่อไม่รวม green rack ซ้ายหรือช่องว่าง
        if bx < rw * 0.18:
            search_start = int(rw * 0.18)
            search_proj = x_proj[search_start:]

            if search_proj.size > 0:
                local_max = float(np.max(search_proj))

                if local_max > 0:
                    strong = np.where(search_proj > local_max * 0.25)[0]

                    if strong.size > 0:
                        new_bx = search_start + int(strong[0])
                        old_right = bx + bw
                        bx = new_bx
                        bw = old_right - bx

        pad_x = int(bw * 0.025)
        pad_y = int(bh * 0.035)

        bx = clamp(bx - pad_x, 0, rw - 1)
        by = clamp(by - pad_y, 0, rh - 1)
        bw = clamp(bw + pad_x * 2, 1, rw - bx)
        bh = clamp(bh + pad_y * 2, 1, rh - by)

        aspect = bw / float(max(1, bh))

        if (
            bw >= rw * 0.35 and
            bh >= rh * 0.40 and
            1.4 <= aspect <= 5.8
        ):
            main_box = (bx, by, bw, bh)

    # fallback ถ้าหา block ไม่เจอ
    if main_box is None:
        main_box = (
            int(rw * 0.22),
            int(rh * 0.04),
            int(rw * 0.72),
            int(rh * 0.90)
        )

    bx, by, bw, bh = main_box

    # =========================
    # GRID DETECTION INSIDE MAIN CREAM BLOCK
    # =========================
    crop_cream = cream_clean[by:by + bh, bx:bx + bw]
    crop_green = green_clean[by:by + bh, bx:bx + bw]
    crop_gray = gray[by:by + bh, bx:bx + bw]

    crop_edges = cv2.Canny(
        cv2.GaussianBlur(crop_gray, (5, 5), 0),
        45,
        135
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(25, int(bh * 0.20)))
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(35, int(bw * 0.10)), 3)
    )

    vertical_mask = cv2.morphologyEx(
        crop_cream,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    horizontal_mask = cv2.morphologyEx(
        crop_cream,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    edge_on_cream = cv2.bitwise_and(
        crop_edges,
        cv2.bitwise_or(crop_cream, crop_green)
    )

    vertical_mask = cv2.bitwise_or(
        vertical_mask,
        edge_on_cream
    )

    horizontal_mask = cv2.bitwise_or(
        horizontal_mask,
        edge_on_cream
    )

    v_lines = find_line_positions(
        vertical_mask,
        axis=0,
        min_gap=max(12, int(bw * 0.035)),
        threshold_ratio=0.22
    )

    h_lines = find_line_positions(
        horizontal_mask,
        axis=1,
        min_gap=max(10, int(bh * 0.055)),
        threshold_ratio=0.22
    )

    v_lines = normalize_grid_lines(
        v_lines,
        bw,
        edge_ratio=0.12
    )

    h_lines = normalize_grid_lines(
        h_lines,
        bh,
        edge_ratio=0.12
    )

    v_lines = remove_close_lines(
        v_lines,
        min_cell_size=int(bw * 0.105)
    )

    h_lines = remove_close_lines(
        h_lines,
        min_cell_size=int(bh * 0.180)
    )

    col_count = max(1, len(v_lines) - 1)
    row_count = max(1, len(h_lines) - 1)

    # =========================
    # DYNAMIC FALLBACK ESTIMATION
    # ไม่ fix 5x3 แต่ estimate จากขนาด block
    # =========================
    if col_count < 2 or col_count > 8:
        estimated_cell_w = rw * 0.145
        col_count = int(round(bw / max(1.0, estimated_cell_w)))
        col_count = clamp(col_count, 1, 8)

    if row_count < 2 or row_count > 5:
        estimated_cell_h = rh * 0.285
        row_count = int(round(bh / max(1.0, estimated_cell_h)))
        row_count = clamp(row_count, 1, 5)

    # สร้าง grid ใหม่จากจำนวน dynamic ที่ได้
    v_lines_final = []

    for c in range(col_count + 1):
        v_lines_final.append(
            int(c * bw / float(col_count))
        )

    h_lines_final = []

    for r in range(row_count + 1):
        h_lines_final.append(
            int(r * bh / float(row_count))
        )

    candidates = []

    # =========================
    # CREATE CREAM CELL CANDIDATES
    # สำคัญ: green ที่อยู่ใน cream block ให้นับเป็น cream cell
    # ไม่แยกนับเป็น green ซ้ำ
    # =========================
    for r in range(row_count):
        for c in range(col_count):

            cx1 = bx + v_lines_final[c]
            cx2 = bx + v_lines_final[c + 1]
            cy1 = by + h_lines_final[r]
            cy2 = by + h_lines_final[r + 1]

            cw = cx2 - cx1
            ch = cy2 - cy1

            if cw <= 0 or ch <= 0:
                continue

            if cw < rw * 0.065 or ch < rh * 0.120:
                continue

            if cw > rw * 0.300 or ch > rh * 0.460:
                continue

            aspect = cw / float(max(1, ch))

            if aspect < 0.40 or aspect > 3.20:
                continue

            cell_area = float(cw * ch)

            local_x1 = max(0, cx1 - bx)
            local_x2 = min(bw, cx2 - bx)
            local_y1 = max(0, cy1 - by)
            local_y2 = min(bh, cy2 - by)

            cell_cream = cream_clean[cy1:cy2, cx1:cx2]
            cell_green = green_clean[cy1:cy2, cx1:cx2]
            cell_edges = crop_edges[local_y1:local_y2, local_x1:local_x2]

            cream_ratio = cv2.countNonZero(cell_cream) / max(1.0, cell_area)
            green_ratio = cv2.countNonZero(cell_green) / max(1.0, cell_area)
            edge_ratio = cv2.countNonZero(cell_edges) / max(1.0, cell_area)

            strip = max(3, int(min(cw, ch) * 0.08))

            top_strip = cell_cream[:strip, :]
            bottom_strip = cell_cream[max(0, ch - strip):, :]
            left_strip = cell_cream[:, :strip]
            right_strip = cell_cream[:, max(0, cw - strip):]

            border_pixels = (
                cv2.countNonZero(top_strip) +
                cv2.countNonZero(bottom_strip) +
                cv2.countNonZero(left_strip) +
                cv2.countNonZero(right_strip)
            )

            border_area = (
                top_strip.size +
                bottom_strip.size +
                left_strip.size +
                right_strip.size
            )

            border_ratio = border_pixels / max(1.0, float(border_area))

            has_cell = (
                cream_ratio >= 0.014 or
                border_ratio >= 0.016 or
                edge_ratio >= 0.018 or
                green_ratio >= 0.030
            )

            if not has_cell:
                continue

            score = (
                cream_ratio * 100.0 +
                border_ratio * 120.0 +
                edge_ratio * 40.0 +
                green_ratio * 25.0
            )

            candidates.append(
                {
                    "box": (cx1, cy1, cw, ch),
                    "score": score,
                    "type": "cream",
                    "source": "grid"
                }
            )

    # =========================
    # GREEN OUTSIDE CREAM BLOCK
    # นับเฉพาะ rack/pallet เขียวที่อยู่นอก cream block
    # =========================
    green_contours, _ = cv2.findContours(
        green_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    green_candidates = []

    for cnt in green_contours:

        area = cv2.contourArea(cnt)

        if area < rh * rw * 0.0025:
            continue

        x, y, gw, gh = cv2.boundingRect(cnt)

        if gw <= 0 or gh <= 0:
            continue

        aspect = gw / float(gh)

        gx = x + gw / 2.0
        gy = y + gh / 2.0

        inside_cream_block = (
            bx <= gx <= bx + bw and
            by <= gy <= by + bh
        )

        if inside_cream_block:
            continue

        # กันเส้นเขียวล่างรถ
        if y > rh * 0.78:
            continue

        passed_size = (
            gw >= rw * 0.075 and
            gh >= rh * 0.120 and
            area >= rh * rw * 0.0025
        )

        passed_max = (
            gw <= rw * 0.340 and
            gh <= rh * 0.650 and
            area <= rh * rw * 0.120
        )

        passed_aspect = (
            0.35 <= aspect <= 4.50
        )

        if not (passed_size and passed_max and passed_aspect):
            continue

        score = float(area) / float(rh * rw) * 1000.0

        green_candidates.append(
            {
                "box": (x, y, gw, gh),
                "score": score + 50.0,
                "type": "green",
                "source": "green"
            }
        )

    # =========================
    # MERGE GREEN RACK CANDIDATES
    # ถ้า rack เขียวแตกเป็นหลาย contour ให้รวมเป็น 1
    # =========================
    if len(green_candidates) > 1:

        selected_green = nms(
            green_candidates,
            iou_threshold=0.10
        )

        if len(selected_green) > 1:

            xs = []
            ys = []
            xes = []
            yes = []

            for g in selected_green:
                x, y, gw, gh = g["box"]

                xs.append(x)
                ys.append(y)
                xes.append(x + gw)
                yes.append(y + gh)

            union_box = (
                min(xs),
                min(ys),
                max(xes) - min(xs),
                max(yes) - min(ys)
            )

            ux, uy, uw, uh = union_box

            if (
                uw <= rw * 0.35 and
                uh <= rh * 0.70 and
                uy < rh * 0.78
            ):
                green_candidates = [
                    {
                        "box": union_box,
                        "score": 100.0,
                        "type": "green",
                        "source": "green"
                    }
                ]
            else:
                green_candidates = selected_green

    candidates.extend(green_candidates)

    # =========================
    # FINAL NMS
    # กันนับซ้อนทับ
    # =========================
    selected = nms(
        candidates,
        iou_threshold=0.15
    )

    final_selected = []

    for cand in selected:

        x, y, cw, ch = cand["box"]
        cx = x + cw / 2.0
        cy = y + ch / 2.0

        duplicate = False

        for old in final_selected:
            ox, oy, ow, oh = old["box"]
            ocx = ox + ow / 2.0
            ocy = oy + oh / 2.0

            center_close = (
                abs(cx - ocx) < min(cw, ow) * 0.50 and
                abs(cy - ocy) < min(ch, oh) * 0.50
            )

            if center_close:
                duplicate = True
                break

        if not duplicate:
            final_selected.append(cand)

    selected = final_selected

    # =========================
    # COUNT RESULT
    # =========================
    cream_count = 0
    green_count = 0

    for cand in selected:
        if cand["type"] == "green":
            green_count += 1
        else:
            cream_count += 1

    pallet_count = cream_count + green_count

    # =========================
    # DEBUG DRAW
    # =========================
    debug_box = roi.copy()
    debug_contour = roi.copy()

    color_layer = roi.copy()

    color_layer[cream_clean > 0] = (
        0,
        255,
        0
    )

    color_layer[green_clean > 0] = (
        255,
        255,
        0
    )

    debug_overlay = cv2.addWeighted(
        roi,
        0.78,
        color_layer,
        0.22,
        0
    )

    # main cream block
    cv2.rectangle(
        debug_contour,
        (bx, by),
        (bx + bw, by + bh),
        (0, 255, 255),
        3
    )

    # grid lines
    for vx in v_lines_final:
        x = bx + int(vx)

        cv2.line(
            debug_contour,
            (x, by),
            (x, by + bh),
            (0, 255, 255),
            1
        )

    for hy in h_lines_final:
        y = by + int(hy)

        cv2.line(
            debug_contour,
            (bx, y),
            (bx + bw, y),
            (0, 255, 255),
            1
        )

    cv2.putText(
        debug_contour,
        f"GRID {col_count}x{row_count}",
        (bx, max(25, by - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    c_no = 0
    g_no = 0

    for cand in selected:

        x, y, cw, ch = cand["box"]

        if cand["type"] == "green":
            g_no += 1
            color = (255, 255, 0)
            label = f"G{g_no}"
        else:
            c_no += 1
            color = (0, 255, 0)
            label = f"C{c_no}"

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + cw, y + ch),
            color,
            3
        )

        cv2.putText(
            debug_box,
            label,
            (x + 6, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        debug_box,
        f"CREAM={cream_count} GREEN={green_count} TOTAL={pallet_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("INBOUND PALLET DYNAMIC DEBUG")
    print(f"ROI SIZE: {rw}x{rh}")
    print(f"MAIN BOX: {(bx, by, bw, bh)}")
    print(f"GRID: {col_count}x{row_count}")
    print(f"CANDIDATES: {len(candidates)}")
    print(f"SELECTED: {len(selected)}")
    print(f"CREAM COUNT: {cream_count}")
    print(f"GREEN COUNT: {green_count}")
    print(f"PALLET COUNT: {pallet_count}")
    print("=" * 50)

    # =========================
    # SAVE DEBUG
    # ใช้ชื่อไฟล์เดิม ไม่ต้องแก้ route อื่น
    # =========================
    if debug:

        save_debug(
            "debug_original.jpg",
            roi
        )

        save_debug(
            "debug_normalized.jpg",
            roi_norm
        )

        save_debug(
            "debug_brown.jpg",
            cream_clean
        )

        save_debug(
            "debug_green.jpg",
            green_clean
        )

        block_mask = np.zeros_like(cream_clean)

        cv2.rectangle(
            block_mask,
            (bx, by),
            (bx + bw, by + bh),
            255,
            thickness=-1
        )

        save_debug(
            "debug_container.jpg",
            block_mask
        )

        final_mask = cv2.bitwise_or(
            cream_clean,
            green_clean
        )

        save_debug(
            "debug_cargo.jpg",
            final_mask
        )

        save_debug(
            "debug_overlay.jpg",
            debug_overlay
        )

        save_debug(
            "debug_overlay_contour.jpg",
            debug_contour
        )

        save_debug(
            "debug_empty.jpg",
            debug_box
        )

        save_debug(
            "debug_pallet_mask.jpg",
            final_mask
        )

        save_debug(
            "debug_pallet_box.jpg",
            debug_box
        )

    return pallet_count

    
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
                "ID": row_id,
                "TFR AI": volume_text,
                "status": "Done"
            }
        ]
    }

    try:
        r = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=20
        )

        print("APPSHEET STATUS:", r.status_code)
        print("APPSHEET RESPONSE:", r.text[:300])

    except Exception as e:
        print("APPSHEET ERROR:", e)


# =========================
# HEALTH CHECK
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "container-volume-ai"
    })


# =========================
# DEBUG VIEW
# =========================
@app.route("/debug/<filename>", methods=["GET"])
def debug_file(filename):

    allowed = {
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_overlay_light.jpg",
        "debug_overlay_contour.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_blue.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    }

    if filename not in allowed:
        return jsonify({"error": "file not allowed"}), 403

    path = os.path.join(DEBUG_DIR, filename)

    if not os.path.exists(path):
        return jsonify({"error": "debug file not found"}), 404

    return send_file(path, mimetype="image/jpeg")


@app.route("/debug-list", methods=["GET"])
def debug_list():

    files = [
        "debug_original.jpg",
        "debug_normalized.jpg",
        "debug_container.jpg",
        "debug_cargo.jpg",
        "debug_empty.jpg",
        "debug_overlay.jpg",
        "debug_overlay_light.jpg",
        "debug_overlay_contour.jpg",
        "debug_green.jpg",
        "debug_brown.jpg",
        "debug_blue.jpg",
        "debug_dark.jpg",
        "debug_texture.jpg"
    ]

    base_url = request.host_url.rstrip("/")

    return jsonify({
        "status": "ok",
        "files": [
            {
                "file": f,
                "url": f"{base_url}/debug/{f}",
                "exists": os.path.exists(os.path.join(DEBUG_DIR, f))
            }
            for f in files
        ]
    })


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    row_id = None

    try:
        data = request.get_json(silent=True)

        print("=" * 50)
        print("REQUEST JSON =", data)
        print("=" * 50)

        if not data:
            return jsonify({"error": "no json"}), 400

        image_url = data.get("link")
        row_id = data.get("id")
        project = str(data.get("project", "outbound")).strip().lower()

        debug = bool(data.get("debug", True))
        return_empty = bool(data.get("return_empty", False))

        if not image_url or not row_id:
            return jsonify({"error": "missing data"}), 400

        print("PROJECT =", repr(project))

        with lock:
            if row_id in processed_ids:
                return jsonify({"status": "skipped", "id": row_id}), 200

            processed_ids.add(row_id)

        img = download_image(image_url)

        if img is None:
            with lock:
                processed_ids.discard(row_id)
            return jsonify({"error": "image fail"}), 400

        if project == "outbound":

            volume = gen_volume(
                img,
                debug=debug,
                return_empty=return_empty
            )

            result_text = f"{volume}%"

            print("OUTBOUND:", result_text)

        elif project == "inbound":

            pallet_count = gen_pallet(
                img,
                debug=debug
            )

            result_text = str(pallet_count)

            print("INBOUND:", result_text)

        else:
            with lock:
                processed_ids.discard(row_id)

            return jsonify({
                "error": f"unknown project: {project}"
            }), 400

        update_appsheet(
            row_id,
            result_text
        )

        base_url = request.host_url.rstrip("/")

        if project == "inbound":
            debug_urls = {
                "pallet_mask": f"{base_url}/debug/debug_pallet_mask.jpg",
                "pallet_box": f"{base_url}/debug/debug_pallet_box.jpg",
                "list": f"{base_url}/debug-list"
            }
        else:
            debug_urls = {
                "overlay": f"{base_url}/debug/debug_overlay.jpg",
                "overlay_light": f"{base_url}/debug/debug_overlay_light.jpg",
                "overlay_contour": f"{base_url}/debug/debug_overlay_contour.jpg",
                "cargo": f"{base_url}/debug/debug_cargo.jpg",
                "empty": f"{base_url}/debug/debug_empty.jpg",
                "container": f"{base_url}/debug/debug_container.jpg",
                "blue": f"{base_url}/debug/debug_blue.jpg",
                "list": f"{base_url}/debug-list"
            }

        return jsonify({
            "status": "success",
            "project": project,
            "id": row_id,
            "result": result_text,
            "mode": "empty" if return_empty else "filled",
            "debug": debug,
            "debug_urls": debug_urls
        })

    except Exception:
        if row_id:
            with lock:
                processed_ids.discard(row_id)

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
