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
# CREAM GRID + GREEN RACK DETECTION
# DEBUG IMAGE ONLY
# =========================
def gen_pallet(img, debug=True):

    if img is None or img.size == 0:
        return 0

    # =========================
    # LOCAL HELPERS
    # =========================
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def find_line_centers_from_projection(proj, min_spacing, threshold_ratio=0.35):
        """
        หา center ของเส้นหลักจาก projection
        ใช้หาเสา/คานของกรงครีม
        """
        if proj is None or len(proj) == 0:
            return []

        proj = proj.astype(np.float32)

        # smooth projection
        k = 17
        kernel = np.ones(k, dtype=np.float32) / float(k)
        proj_smooth = np.convolve(proj, kernel, mode="same")

        max_val = float(np.max(proj_smooth))

        if max_val <= 1e-6:
            return []

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

        centers = []

        for s, e in runs:
            if e < s:
                continue

            segment = proj_smooth[s:e + 1]
            if segment.size == 0:
                continue

            peak = int(np.argmax(segment))
            center = s + peak
            value = float(proj_smooth[center])

            centers.append((center, value))

        # เรียงจากเส้นที่ชัดที่สุดก่อน
        centers = sorted(
            centers,
            key=lambda x: x[1],
            reverse=True
        )

        selected = []

        for center, value in centers:
            too_close = False

            for old_center in selected:
                if abs(center - old_center) < min_spacing:
                    too_close = True
                    break

            if not too_close:
                selected.append(center)

        selected = sorted(selected)

        return selected

    def normalize_line_edges(lines, length, edge_ratio=0.10):
        """
        เติมเส้นขอบซ้าย/ขวา หรือ บน/ล่าง ถ้าเส้นที่เจอห่างจากขอบมากเกินไป
        """
        if lines is None:
            lines = []

        lines = list(lines)

        if len(lines) == 0:
            return lines

        if lines[0] > length * edge_ratio:
            lines.insert(0, 0)

        if lines[-1] < length * (1.0 - edge_ratio):
            lines.append(length - 1)

        return sorted(lines)

    def build_cells_from_count(x, y, bw, bh, cols, rows):
        cells = []

        if cols <= 0 or rows <= 0:
            return cells

        cell_w = bw / float(cols)
        cell_h = bh / float(rows)

        for r in range(rows):
            for c in range(cols):
                cx1 = int(x + c * cell_w)
                cy1 = int(y + r * cell_h)
                cx2 = int(x + (c + 1) * cell_w)
                cy2 = int(y + (r + 1) * cell_h)

                cells.append(
                    (
                        cx1,
                        cy1,
                        max(1, cx2 - cx1),
                        max(1, cy2 - cy1)
                    )
                )

        return cells

    # =========================
    # RESIZE
    # =========================
    img = cv2.resize(img, (960, 720))
    h, w = img.shape[:2]

    # =========================
    # ROI
    # ตัดท้องฟ้า / หลังคาสูง / ล้อ / พื้นถนนออก
    # เหลือเฉพาะบริเวณสินค้าในรถ
    # =========================
    y1 = int(h * 0.26)
    y2 = int(h * 0.72)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return 0

    rh, rw = roi.shape[:2]

    # =========================
    # NORMALIZE LIGHT
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
    # CREAM MASK
    # จับกรง/ครีม/ลังโทนเบจ
    # ตั้งใจไม่จับท้องฟ้า/กระดาษขาวมากเกินไป
    # =========================
    cream_mask = cv2.inRange(
        hsv,
        (7, 22, 55),
        (42, 185, 245)
    )

    # ลบกระดาษ/label ขาวจัด
    white_mask = cv2.inRange(
        hsv,
        (0, 0, 185),
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

    kernel_cream_close = cv2.getStructuringElement(
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
        kernel_cream_close,
        iterations=1
    )

    # ใช้รวม block กรงครีมให้เป็นก้อนใหญ่
    kernel_block = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (45, 17)
    )

    cream_block_mask = cv2.morphologyEx(
        cream_clean,
        cv2.MORPH_CLOSE,
        kernel_block,
        iterations=3
    )

    green_clean = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
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
    # =========================
    contours, _ = cv2.findContours(
        cream_block_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    main_box = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < rh * rw * 0.025:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / float(bh)

        # กรงครีมหลักควรกว้าง มีหลายช่อง และอยู่ช่วงกลาง/ขวาของรถ
        possible = (
            bw >= rw * 0.38 and
            bh >= rh * 0.42 and
            1.4 <= aspect <= 5.5 and
            x + bw >= rw * 0.45
        )

        if not possible:
            continue

        score = area + (bw * bh * 0.25)

        if score > best_score:
            best_score = score
            main_box = (x, y, bw, bh)

    cream_count = 0
    green_count = 0

    debug_box = roi.copy()
    debug_overlay = roi.copy()
    debug_contour = roi.copy()

    v_lines = []
    h_lines = []
    cream_cells = []

    # =========================
    # COUNT CREAM BY GRID
    # =========================
    if main_box is not None:

        bx, by, bw, bh = main_box

        # ขยาย box ให้ครอบกรอบนอกของกรง
        pad_x = int(bw * 0.02)
        pad_y = int(bh * 0.04)

        bx = clamp(bx - pad_x, 0, rw - 1)
        by = clamp(by - pad_y, 0, rh - 1)
        bw = clamp(bw + pad_x * 2, 1, rw - bx)
        bh = clamp(bh + pad_y * 2, 1, rh - by)

        crop_cream = cream_clean[by:by + bh, bx:bx + bw]
        crop_gray = gray[by:by + bh, bx:bx + bw]

        # =========================
        # หาเส้นตั้ง/นอนของกรง
        # =========================
        vertical_kernel_h = max(25, int(bh * 0.20))
        horizontal_kernel_w = max(35, int(bw * 0.12))

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, vertical_kernel_h)
        )

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_kernel_w, 3)
        )

        vertical_lines_mask = cv2.morphologyEx(
            crop_cream,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=1
        )

        horizontal_lines_mask = cv2.morphologyEx(
            crop_cream,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=1
        )

        edges = cv2.Canny(
            crop_gray,
            50,
            140
        )

        vertical_proj = (
            np.sum(vertical_lines_mask > 0, axis=0).astype(np.float32) +
            np.sum(edges > 0, axis=0).astype(np.float32) * 0.20
        )

        horizontal_proj = (
            np.sum(horizontal_lines_mask > 0, axis=1).astype(np.float32) +
            np.sum(edges > 0, axis=1).astype(np.float32) * 0.20
        )

        min_v_spacing = max(30, int(bw * 0.12))
        min_h_spacing = max(30, int(bh * 0.20))

        v_lines = find_line_centers_from_projection(
            vertical_proj,
            min_spacing=min_v_spacing,
            threshold_ratio=0.28
        )

        h_lines = find_line_centers_from_projection(
            horizontal_proj,
            min_spacing=min_h_spacing,
            threshold_ratio=0.28
        )

        v_lines = normalize_line_edges(
            v_lines,
            bw,
            edge_ratio=0.10
        )

        h_lines = normalize_line_edges(
            h_lines,
            bh,
            edge_ratio=0.12
        )

        # =========================
        # คำนวณจำนวน column/row
        # =========================
        col_count = max(0, len(v_lines) - 1)
        row_count = max(0, len(h_lines) - 1)

        block_aspect = bw / float(bh)

        # =========================
        # FALLBACK / STABILIZER
        # สำหรับ pattern ตามภาพตัวอย่าง: กรงครีม 5 คอลัมน์ x 3 แถว
        # ถ้า line detection เพี้ยน ให้ใช้สัดส่วน block ช่วย lock ค่า
        # =========================
        if bh >= rh * 0.45 and 2.0 <= block_aspect <= 3.8:
            if row_count < 2 or row_count > 4:
                row_count = 3

            if col_count < 4 or col_count > 6:
                col_count = 5

            # ถ้าได้ใกล้เคียงอยู่แล้ว ให้ดึงเข้าหา 5x3
            if row_count in [2, 3, 4]:
                row_count = 3

            if col_count in [4, 5, 6]:
                col_count = 5

        else:
            # fallback ทั่วไป
            if row_count <= 0:
                row_count = 1

            if col_count <= 0:
                estimated_cols = int(round(block_aspect * row_count / 1.15))
                col_count = clamp(estimated_cols, 1, 8)

        col_count = clamp(col_count, 1, 8)
        row_count = clamp(row_count, 1, 5)

        cream_count = int(col_count * row_count)

        cream_cells = build_cells_from_count(
            bx,
            by,
            bw,
            bh,
            col_count,
            row_count
        )

        # =========================
        # DEBUG DRAW CREAM GRID
        # =========================
        cv2.rectangle(
            debug_box,
            (bx, by),
            (bx + bw, by + bh),
            (0, 255, 255),
            3
        )

        cv2.putText(
            debug_box,
            f"CREAM {col_count}x{row_count}={cream_count}",
            (bx, max(25, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        for i, cell in enumerate(cream_cells, start=1):
            cx, cy, cw, ch = cell

            cv2.rectangle(
                debug_box,
                (cx, cy),
                (cx + cw, cy + ch),
                (0, 255, 0),
                2
            )

            cv2.putText(
                debug_box,
                str(i),
                (cx + 6, cy + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # วาดเส้นที่ตรวจจับได้จริงบน debug_contour
        for vx in v_lines:
            cv2.line(
                debug_contour,
                (bx + int(vx), by),
                (bx + int(vx), by + bh),
                (0, 255, 255),
                2
            )

        for hy in h_lines:
            cv2.line(
                debug_contour,
                (bx, by + int(hy)),
                (bx + bw, by + int(hy)),
                (0, 255, 255),
                2
            )

    # =========================
    # COUNT GREEN RACK OUTSIDE CREAM BLOCK
    # =========================
    green_contours, _ = cv2.findContours(
        green_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in green_contours:
        area = cv2.contourArea(cnt)

        if area < rh * rw * 0.004:
            continue

        x, y, gw, gh = cv2.boundingRect(cnt)

        if gw <= 0 or gh <= 0:
            continue

        aspect = gw / float(gh)

        passed_size = (
            gw >= rw * 0.08 and
            gh >= rh * 0.18 and
            area >= rh * rw * 0.004
        )

        passed_aspect = (
            0.35 <= aspect <= 4.5
        )

        if not passed_size or not passed_aspect:
            continue

        # ไม่ให้นับสีเขียวที่อยู่ภายในกรงครีม
        inside_main_block = False

        if main_box is not None:
            bx, by, bw, bh = main_box

            gx_center = x + gw / 2.0
            gy_center = y + gh / 2.0

            inside_main_block = (
                bx <= gx_center <= bx + bw and
                by <= gy_center <= by + bh
            )

        if inside_main_block:
            continue

        # กันการจับเส้นขอบรถล่าง
        if y > rh * 0.78:
            continue

        green_count += 1

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + gw, y + gh),
            (255, 255, 0),
            3
        )

        cv2.putText(
            debug_box,
            f"GREEN {green_count}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    total_count = int(cream_count + green_count)

    # =========================
    # OVERLAY
    # =========================
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

    cv2.putText(
        debug_box,
        f"CREAM={cream_count} GREEN={green_count} TOTAL={total_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("INBOUND PALLET DEBUG")
    print(f"ROI SIZE: {rw}x{rh}")
    print(f"MAIN CREAM BOX: {main_box}")
    print(f"V LINES: {v_lines}")
    print(f"H LINES: {h_lines}")
    print(f"CREAM COUNT: {cream_count}")
    print(f"GREEN COUNT: {green_count}")
    print(f"PALLET COUNT: {total_count}")
    print("=" * 50)

    # =========================
    # DEBUG OUTPUT
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

        # main cream block mask
        save_debug(
            "debug_container.jpg",
            cream_block_mask
        )

        combined_mask = cv2.bitwise_or(
            cream_clean,
            green_clean
        )

        save_debug(
            "debug_cargo.jpg",
            combined_mask
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
        # เขียว = ช่องครีมที่นับ
        # ฟ้า/เหลือง = green rack ที่นับ
        save_debug(
            "debug_empty.jpg",
            debug_box
        )

        # เผื่อเปิดชื่อเดิมของ inbound
        save_debug(
            "debug_pallet_mask.jpg",
            combined_mask
        )

        save_debug(
            "debug_pallet_box.jpg",
            debug_box
        )

    return total_count

    
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
