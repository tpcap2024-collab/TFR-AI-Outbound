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
# DYNAMIC PALLET DETECTION + NON-OVERLAP NMS
# =========================
def gen_pallet(img, debug=True):

    if img is None or img.size == 0:
        return 0

    # =========================
    # LOCAL HELPER FUNCTIONS
    # =========================
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

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

        area_a = aw * ah
        area_b = bw * bh

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / float(union)

    def nms_candidates(candidates, iou_threshold=0.25):
        """
        candidates item:
        {
            "box": (x, y, w, h),
            "score": float,
            "type": "cream" or "green"
        }
        """
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

            for sel in selected:
                iou = box_iou(cand["box"], sel["box"])

                if iou > iou_threshold:
                    keep = False
                    break

            if keep:
                selected.append(cand)

        return selected

    def merge_close_lines(lines, min_gap):
        if not lines:
            return []

        lines = sorted(lines)
        merged = []
        group = [lines[0]]

        for x in lines[1:]:
            if abs(x - group[-1]) <= min_gap:
                group.append(x)
            else:
                merged.append(int(sum(group) / len(group)))
                group = [x]

        if group:
            merged.append(int(sum(group) / len(group)))

        return merged

    def find_projection_lines(mask, axis, min_strength_ratio, min_gap):
        """
        axis=0 => vertical lines from column projection
        axis=1 => horizontal lines from row projection
        """
        if axis == 0:
            proj = np.sum(mask > 0, axis=0).astype(np.float32)
        else:
            proj = np.sum(mask > 0, axis=1).astype(np.float32)

        if proj.size == 0:
            return []

        # smooth projection
        k = 15
        kernel = np.ones(k, dtype=np.float32) / float(k)
        proj_smooth = np.convolve(proj, kernel, mode="same")

        max_val = float(np.max(proj_smooth))

        if max_val <= 1e-6:
            return []

        threshold = max_val * min_strength_ratio

        raw_lines = []

        in_run = False
        start = 0

        for i, val in enumerate(proj_smooth):
            if val >= threshold and not in_run:
                start = i
                in_run = True
            elif val < threshold and in_run:
                end = i - 1

                segment = proj_smooth[start:end + 1]

                if segment.size > 0:
                    peak = int(np.argmax(segment))
                    raw_lines.append(start + peak)

                in_run = False

        if in_run:
            end = len(proj_smooth) - 1
            segment = proj_smooth[start:end + 1]

            if segment.size > 0:
                peak = int(np.argmax(segment))
                raw_lines.append(start + peak)

        lines = merge_close_lines(
            raw_lines,
            min_gap=min_gap
        )

        return lines

    def add_border_lines(lines, length, max_edge_gap_ratio=0.10):
        lines = list(lines)

        if len(lines) == 0:
            return [0, length - 1]

        lines = sorted(lines)

        if lines[0] > length * max_edge_gap_ratio:
            lines.insert(0, 0)

        if lines[-1] < length * (1.0 - max_edge_gap_ratio):
            lines.append(length - 1)

        return sorted(lines)

    def remove_too_close_lines(lines, min_cell_size):
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
    # ตัดท้องฟ้า ล้อ และพื้นถนนออก
    # เหลือโซนสินค้าบนรถ
    # =========================
    y1 = int(h * 0.23)
    y2 = int(h * 0.74)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)

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
    # MASK: CREAM / BEIGE / BROWN
    # สำหรับกรงครีม ลัง กระดาษ ไม้
    # =========================
    cream_mask_1 = cv2.inRange(
        hsv,
        (5, 18, 55),
        (45, 200, 255)
    )

    cream_mask_2 = cv2.inRange(
        hsv,
        (0, 0, 95),
        (50, 105, 245)
    )

    cream_mask = cv2.bitwise_or(
        cream_mask_1,
        cream_mask_2
    )

    # ลบ label/กระดาษขาวจัด
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
    # MASK: GREEN
    # สำหรับ rack/pallet เขียว
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
    # EDGE / LINE MASK
    # ใช้ช่วยหาโครงกรง ไม่ใช่นับตู้ใหญ่
    # =========================
    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray_blur,
        45,
        135
    )

    combined_for_lines = cv2.bitwise_or(
        cream_clean,
        green_clean
    )

    combined_for_lines = cv2.bitwise_or(
        combined_for_lines,
        edges
    )

    # ตัดขอบบน/ล่างที่มักเป็นหลังคาและคานรถ
    top_cut = int(rh * 0.05)
    bottom_cut = int(rh * 0.04)

    combined_for_lines[:top_cut, :] = 0
    combined_for_lines[rh - bottom_cut:, :] = 0

    # =========================
    # FIND DYNAMIC GRID LINES
    # =========================
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(25, int(rh * 0.18)))
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(35, int(rw * 0.08)), 3)
    )

    vertical_line_mask = cv2.morphologyEx(
        combined_for_lines,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    horizontal_line_mask = cv2.morphologyEx(
        combined_for_lines,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    v_lines = find_projection_lines(
        vertical_line_mask,
        axis=0,
        min_strength_ratio=0.25,
        min_gap=max(12, int(rw * 0.025))
    )

    h_lines = find_projection_lines(
        horizontal_line_mask,
        axis=1,
        min_strength_ratio=0.25,
        min_gap=max(10, int(rh * 0.035))
    )

    # เพิ่มขอบภาพเฉพาะถ้าจำเป็น
    v_lines = add_border_lines(
        v_lines,
        rw,
        max_edge_gap_ratio=0.08
    )

    h_lines = add_border_lines(
        h_lines,
        rh,
        max_edge_gap_ratio=0.10
    )

    # กรองเส้นที่ใกล้กันเกินไป เพื่อไม่ให้ cell เล็กผิดปกติ
    min_cell_w = int(rw * 0.075)
    min_cell_h = int(rh * 0.120)

    v_lines = remove_too_close_lines(
        v_lines,
        min_cell_w
    )

    h_lines = remove_too_close_lines(
        h_lines,
        min_cell_h
    )

    # =========================
    # CREATE CELL CANDIDATES FROM GRID
    # =========================
    candidates = []
    rejected_boxes = []

    combined_mask = cv2.bitwise_or(
        cream_clean,
        green_clean
    )

    for yi in range(len(h_lines) - 1):
        for xi in range(len(v_lines) - 1):

            cx1 = int(v_lines[xi])
            cx2 = int(v_lines[xi + 1])
            cy1 = int(h_lines[yi])
            cy2 = int(h_lines[yi + 1])

            cw = cx2 - cx1
            ch = cy2 - cy1

            if cw <= 0 or ch <= 0:
                continue

            cell_area = float(cw * ch)

            # =========================
            # SIZE FILTER
            # ขนาด cell ต้องใกล้เคียง pallet/cage
            # =========================
            too_small = (
                cw < rw * 0.075 or
                ch < rh * 0.120 or
                cell_area < rh * rw * 0.006
            )

            too_big = (
                cw > rw * 0.260 or
                ch > rh * 0.420 or
                cell_area > rh * rw * 0.100
            )

            aspect = cw / float(ch)

            bad_aspect = (
                aspect < 0.45 or
                aspect > 2.80
            )

            if too_small or too_big or bad_aspect:
                rejected_boxes.append(
                    {
                        "box": (cx1, cy1, cw, ch),
                        "reason": "SIZE"
                    }
                )
                continue

            cell_cream = cream_clean[cy1:cy2, cx1:cx2]
            cell_green = green_clean[cy1:cy2, cx1:cx2]
            cell_combined = combined_mask[cy1:cy2, cx1:cx2]
            cell_edges = edges[cy1:cy2, cx1:cx2]

            cream_pixels = cv2.countNonZero(cell_cream)
            green_pixels = cv2.countNonZero(cell_green)
            combined_pixels = cv2.countNonZero(cell_combined)
            edge_pixels = cv2.countNonZero(cell_edges)

            cream_ratio = cream_pixels / max(1.0, cell_area)
            green_ratio = green_pixels / max(1.0, cell_area)
            combined_ratio = combined_pixels / max(1.0, cell_area)
            edge_ratio = edge_pixels / max(1.0, cell_area)

            # =========================
            # OCCUPANCY CHECK
            # พาเลท/กรงตาข่ายมี pixel ไม่เต็มช่อง จึงใช้ threshold ต่ำ
            # =========================
            has_cream = cream_ratio >= 0.012
            has_green = green_ratio >= 0.018
            has_structure = edge_ratio >= 0.018
            has_content = combined_ratio >= 0.018

            if not ((has_cream or has_green) and (has_structure or has_content)):
                rejected_boxes.append(
                    {
                        "box": (cx1, cy1, cw, ch),
                        "reason": "EMPTY"
                    }
                )
                continue

            score = (
                cream_ratio * 80.0 +
                green_ratio * 120.0 +
                edge_ratio * 30.0 +
                combined_ratio * 60.0
            )

            pallet_type = "green" if green_ratio > cream_ratio * 1.25 else "cream"

            candidates.append(
                {
                    "box": (cx1, cy1, cw, ch),
                    "score": score,
                    "type": pallet_type,
                    "source": "grid"
                }
            )

    # =========================
    # GREEN CONTOUR CANDIDATES
    # สำหรับ rack เขียวที่ไม่เข้ากับ grid
    # =========================
    green_contours, _ = cv2.findContours(
        green_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in green_contours:

        area = cv2.contourArea(cnt)

        if area < rh * rw * 0.0025:
            continue

        x, y, gw, gh = cv2.boundingRect(cnt)

        if gw <= 0 or gh <= 0:
            continue

        aspect = gw / float(gh)

        passed_size = (
            gw >= rw * 0.060 and
            gh >= rh * 0.100 and
            area >= rh * rw * 0.0025
        )

        passed_max = (
            gw <= rw * 0.300 and
            gh <= rh * 0.500 and
            area <= rh * rw * 0.120
        )

        passed_aspect = (
            0.35 <= aspect <= 4.00
        )

        # กันเส้นยาวด้านล่างรถ
        not_bottom_bar = y < rh * 0.78

        if passed_size and passed_max and passed_aspect and not_bottom_bar:
            score = float(area) / float(rh * rw) * 1000.0

            candidates.append(
                {
                    "box": (x, y, gw, gh),
                    "score": score + 20.0,
                    "type": "green",
                    "source": "contour"
                }
            )
        else:
            rejected_boxes.append(
                {
                    "box": (x, y, gw, gh),
                    "reason": "GREEN_REJECT"
                }
            )

    # =========================
    # CREAM CONTOUR CANDIDATES
    # สำหรับพาเลทครีมที่ grid line หาไม่ครบ
    # =========================
    cream_block = cv2.morphologyEx(
        cream_clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)),
        iterations=1
    )

    cream_contours, _ = cv2.findContours(
        cream_block,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in cream_contours:

        area = cv2.contourArea(cnt)

        if area < rh * rw * 0.003:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw <= 0 or ch <= 0:
            continue

        aspect = cw / float(ch)

        passed_size = (
            cw >= rw * 0.070 and
            ch >= rh * 0.100 and
            area >= rh * rw * 0.003
        )

        passed_max = (
            cw <= rw * 0.260 and
            ch <= rh * 0.420 and
            area <= rh * rw * 0.090
        )

        passed_aspect = (
            0.45 <= aspect <= 3.00
        )

        if passed_size and passed_max and passed_aspect:
            score = float(area) / float(rh * rw) * 800.0

            candidates.append(
                {
                    "box": (x, y, cw, ch),
                    "score": score,
                    "type": "cream",
                    "source": "contour"
                }
            )

    # =========================
    # NON-OVERLAP NMS
    # ตัดกรอบซ้อนทับกัน ไม่ให้นับซ้ำ
    # =========================
    selected = nms_candidates(
        candidates,
        iou_threshold=0.22
    )

    # =========================
    # FINAL SANITY FILTER
    # กันกรอบที่แทบซ้อนกันแต่ IoU ต่ำจากการเหลื่อมขอบ
    # =========================
    final_selected = []

    for cand in selected:
        x, y, bw, bh = cand["box"]

        cx = x + bw / 2.0
        cy = y + bh / 2.0

        duplicate = False

        for old in final_selected:
            ox, oy, ow, oh = old["box"]

            ocx = ox + ow / 2.0
            ocy = oy + oh / 2.0

            center_close = (
                abs(cx - ocx) < min(bw, ow) * 0.55 and
                abs(cy - ocy) < min(bh, oh) * 0.55
            )

            if center_close:
                duplicate = True
                break

        if not duplicate:
            final_selected.append(cand)

    selected = final_selected

    pallet_count = len(selected)

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

    # วาดเส้น grid ที่ระบบหาได้
    for vx in v_lines:
        cv2.line(
            debug_contour,
            (int(vx), 0),
            (int(vx), rh),
            (0, 255, 255),
            1
        )

    for hy in h_lines:
        cv2.line(
            debug_contour,
            (0, int(hy)),
            (rw, int(hy)),
            (0, 255, 255),
            1
        )

    # วาด rejected บางส่วนแบบจาง ๆ ไม่ให้รกเกิน
    max_reject_draw = 40

    for item in rejected_boxes[:max_reject_draw]:
        x, y, bw, bh = item["box"]

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + bw, y + bh),
            (0, 0, 120),
            1
        )

    # วาดกล่องที่นับจริง
    cream_count = 0
    green_count = 0

    for idx, cand in enumerate(selected, start=1):

        x, y, bw, bh = cand["box"]
        pallet_type = cand["type"]
        source = cand["source"]

        if pallet_type == "green":
            green_count += 1
            color = (255, 255, 0)
            label = f"G{green_count}"
        else:
            cream_count += 1
            color = (0, 255, 0)
            label = f"C{cream_count}"

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + bw, y + bh),
            color,
            3
        )

        cv2.putText(
            debug_box,
            label,
            (x + 6, max(24, y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            debug_box,
            source,
            (x + 6, min(rh - 6, y + bh - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
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
    print(f"V_LINES: {v_lines}")
    print(f"H_LINES: {h_lines}")
    print(f"CANDIDATES: {len(candidates)}")
    print(f"SELECTED/NMS: {len(selected)}")
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

        # cream mask
        save_debug(
            "debug_brown.jpg",
            cream_clean
        )

        # green mask
        save_debug(
            "debug_green.jpg",
            green_clean
        )

        # line mask
        save_debug(
            "debug_container.jpg",
            combined_for_lines
        )

        # combined mask
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

        # ภาพผลลัพธ์หลัก
        save_debug(
            "debug_empty.jpg",
            debug_box
        )

        # ชื่อเดิมของ inbound
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
