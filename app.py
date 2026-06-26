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
# ใส่ค่าจริงเอง
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
            timeout=20,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        print("IMAGE STATUS:", r.status_code)
        print("IMAGE CONTENT TYPE:", r.headers.get("Content-Type"))

        if r.status_code != 200:
            print("IMAGE HTTP ERROR:", r.status_code)
            print(r.text[:300])
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            print("IMAGE DECODE FAILED")
            return None

        print("IMAGE SHAPE:", img.shape)

        return img

    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        print(traceback.format_exc())
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(name, img):
    try:
        if img is None:
            print("SAVE DEBUG ERROR:", name, "image is None")
            return False

        path = os.path.join(DEBUG_DIR, name)

        ok = cv2.imwrite(path, img)

        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0

        print(
            f"SAVE DEBUG {name}: "
            f"ok={ok} exists={exists} size={size} path={path}"
        )

        return ok

    except Exception as e:
        print("SAVE DEBUG ERROR:", name, e)
        print(traceback.format_exc())
        return False


# =========================
# OUTBOUND VOLUME MODEL
# =========================
def gen_volume(img, debug=True, return_empty=False):

    if img is None or img.size == 0:
        return 0

    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    roi = img[
        int(h * 0.25):int(h * 0.80),
        int(w * 0.10):int(w * 0.90)
    ]

    if roi.size == 0:
        return 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    green = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
    brown = cv2.inRange(hsv, (5, 40, 40), (35, 255, 255))
    blue = cv2.inRange(hsv, (85, 40, 40), (130, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

    edges = cv2.Canny(gray, 40, 120)

    cargo = cv2.bitwise_or(green, brown)
    cargo = cv2.bitwise_or(cargo, blue)
    cargo = cv2.bitwise_or(cargo, dark)
    cargo = cv2.bitwise_or(cargo, edges)

    cargo = cv2.morphologyEx(
        cargo,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1
    )

    filled = cv2.countNonZero(cargo) / float(max(1, cargo.size)) * 100
    filled = min(100, filled)

    result = (100 - filled) if return_empty else filled
    result = int(round(result / 5) * 5)
    result = max(0, min(100, result))

    if debug:
        overlay = roi.copy()
        overlay[cargo > 0] = (0, 255, 0)

        empty = cv2.bitwise_not(cargo)

        save_debug("debug_original.jpg", roi)
        save_debug("debug_cargo.jpg", cargo)
        save_debug("debug_empty.jpg", empty)
        save_debug("debug_overlay.jpg", overlay)

    print("=" * 50)
    print("VOLUME RESULT:", result)
    print("=" * 50)

    return result


# =========================
# INBOUND PALLET MODEL
# Edge / Frame detection: cream + green + wood
# =========================
def gen_pallet(img, debug=True):

    if img is None or img.size == 0:
        return 0

    # =========================
    # HELPERS
    # =========================
    def save_dbg(name, im):
        if debug and im is not None:
            save_debug(name, im)

    def clean(mask, k=5, close_it=1, open_it=1):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if close_it > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                ker,
                iterations=close_it
            )

        if open_it > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                ker,
                iterations=open_it
            )

        return mask

    def color_by_type(t):
        if t == "green":
            return (0, 255, 0)
        if t == "blue":
            return (255, 160, 0)
        if t == "cream":
            return (0, 255, 255)
        if t == "wood":
            return (0, 180, 255)
        return (255, 255, 255)

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

    def nms_iou(boxes, types, iou_th=0.22):
        if not boxes:
            return boxes, types

        items = []

        for i, b in enumerate(boxes):
            x, y, w, h = b
            area = w * h
            items.append((b, types[i], area))

        items = sorted(items, key=lambda v: v[2], reverse=True)

        keep_boxes = []
        keep_types = []

        for b, t, area in items:
            duplicate = False

            for old in keep_boxes:
                if box_iou(b, old) > iou_th:
                    duplicate = True
                    break

                x, y, w, h = b
                ox, oy, ow, oh = old

                ix1 = max(x, ox)
                iy1 = max(y, oy)
                ix2 = min(x + w, ox + ow)
                iy2 = min(y + h, oy + oh)

                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                small_area = min(w * h, ow * oh)

                if small_area > 0 and inter / float(small_area) > 0.60:
                    duplicate = True
                    break

            if not duplicate:
                keep_boxes.append(b)
                keep_types.append(t)

        return keep_boxes, keep_types

    # =========================
    # RESIZE + ROI
    # =========================
    img = cv2.resize(img, (1280, 720))
    H, W = img.shape[:2]

    roi = img[
        int(H * 0.13):int(H * 0.85),
        int(W * 0.02):int(W * 0.98)
    ]

    rh, rw = roi.shape[:2]

    top_cut = int(rh * 0.28)
    bottom_cut = int(rh * 0.82)

    # =========================
    # LIGHT NORMALIZE
    # =========================
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    # =========================
    # COLOR MASKS
    # =========================

    # เขียว
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (95, 255, 255)
    )

    # น้ำเงิน
    blue_mask = cv2.inRange(
        hsv,
        (85, 35, 35),
        (135, 255, 255)
    )

    # ครีม / beige
    cream_mask = cv2.inRange(
        hsv,
        (5, 20, 70),
        (45, 190, 255)
    )

    # ไม้ / carton
    wood_mask = cv2.inRange(
        hsv,
        (5, 35, 45),
        (38, 230, 245)
    )

    for m in [green_mask, blue_mask, cream_mask, wood_mask]:
        m[:top_cut, :] = 0
        m[bottom_cut:, :] = 0
        m[:, :int(rw * 0.01)] = 0
        m[:, int(rw * 0.99):] = 0

    green_mask = clean(green_mask, 5, 1, 1)
    blue_mask = clean(blue_mask, 5, 1, 1)
    cream_mask = clean(cream_mask, 5, 1, 1)
    wood_mask = clean(wood_mask, 5, 1, 1)

    material_mask = cv2.bitwise_or(green_mask, blue_mask)
    material_mask = cv2.bitwise_or(material_mask, cream_mask)
    material_mask = cv2.bitwise_or(material_mask, wood_mask)

    # =========================
    # EDGE / FRAME DETECTION
    # =========================
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(
        blur,
        35,
        120
    )

    sobel_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=3)

    _, sx = cv2.threshold(
        sobel_x,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, sy = cv2.threshold(
        sobel_y,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edge_base = cv2.bitwise_or(canny, sx)
    edge_base = cv2.bitwise_or(edge_base, sy)

    edge_base[:top_cut, :] = 0
    edge_base[bottom_cut:, :] = 0

    # เส้นตั้งของกรง / rack
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(20, int(rh * 0.08)))
    )

    vertical_lines = cv2.morphologyEx(
        edge_base,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    # เส้นนอนของกรง / rack
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, int(rw * 0.045)), 3)
    )

    horizontal_lines = cv2.morphologyEx(
        edge_base,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    line_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # เก็บเส้นที่อยู่ใกล้วัสดุ
    material_dilate = cv2.dilate(
        material_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 21)),
        iterations=1
    )

    line_near_material = cv2.bitwise_and(
        line_mask,
        material_dilate
    )

    # อนุญาตเส้นกรงช่วงล่าง
    lower_zone = np.zeros_like(line_mask)
    lower_zone[int(rh * 0.34):bottom_cut, :] = 255

    line_lower = cv2.bitwise_and(
        line_mask,
        lower_zone
    )

    frame_mask = cv2.bitwise_or(
        line_near_material,
        line_lower
    )

    frame_mask = cv2.morphologyEx(
        frame_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)),
        iterations=1
    )

    # =========================
    # PALLET SEED
    # =========================
    pallet_seed = cv2.bitwise_or(
        material_mask,
        frame_mask
    )

    pallet_seed = cv2.morphologyEx(
        pallet_seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
        iterations=2
    )

    pallet_seed = cv2.morphologyEx(
        pallet_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # =========================
    # FIND BLOCKS
    # =========================
    contours, _ = cv2.findContours(
        pallet_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < rw * rh * 0.003:
            continue

        if w < rw * 0.04:
            continue

        if h < rh * 0.06:
            continue

        if y + h > bottom_cut:
            continue

        if h > rh * 0.58:
            continue

        box_area = float(max(1, w * h))

        mat_den = cv2.countNonZero(
            material_mask[y:y + h, x:x + w]
        ) / box_area

        frame_den = cv2.countNonZero(
            frame_mask[y:y + h, x:x + w]
        ) / box_area

        if mat_den < 0.004 and frame_den < 0.004:
            continue

        blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda v: (v[1], v[0]))

    # =========================
    # CLASSIFY COLOR CELL
    # =========================
    def classify_color_cell(x1, y1, x2, y2):
        area = float(max(1, (x2 - x1) * (y2 - y1)))

        g = cv2.countNonZero(green_mask[y1:y2, x1:x2]) / area
        b = cv2.countNonZero(blue_mask[y1:y2, x1:x2]) / area
        c = cv2.countNonZero(cream_mask[y1:y2, x1:x2]) / area
        w = cv2.countNonZero(wood_mask[y1:y2, x1:x2]) / area
        f = cv2.countNonZero(frame_mask[y1:y2, x1:x2]) / area

        # เขียวต้องมาก่อน เพราะ rack เขียวชัดที่สุด
        if g > 0.025:
            return "green"

        # น้ำเงินต้องชัดจริงเท่านั้น
        # กันครีม / เงา / ผนังถูกมองเป็นน้ำเงิน
        if b > 0.060 and b > c * 1.6 and b > g * 1.6:
            return "blue"

        # ไม้ / carton
        if w > 0.035 and w > c:
            return "wood"

        # ถ้ามีขอบกรง แต่สีไม่ชัด ให้ถือเป็นครีม
        if c > 0.012 or f > 0.006:
            return "cream"

        return "unknown"

    # =========================
    # BUILD SQUARE-LIKE CELLS
    # =========================
    cells = []
    cell_types = []

    debug_grid = roi.copy()
    debug_box = roi.copy()

    for (x, y, w, h) in blocks:

        block_area = float(max(1, w * h))

        g_ratio = cv2.countNonZero(
            green_mask[y:y + h, x:x + w]
        ) / block_area

        b_ratio = cv2.countNonZero(
            blue_mask[y:y + h, x:x + w]
        ) / block_area

        c_ratio = cv2.countNonZero(
            cream_mask[y:y + h, x:x + w]
        ) / block_area

        wd_ratio = cv2.countNonZero(
            wood_mask[y:y + h, x:x + w]
        ) / block_area

        frame_ratio = cv2.countNonZero(
            frame_mask[y:y + h, x:x + w]
        ) / block_area

        ratios = {
            "green": g_ratio,
            "blue": b_ratio,
            "cream": c_ratio,
            "wood": wd_ratio
        }

        block_type = max(ratios, key=ratios.get)

        if ratios[block_type] < 0.006 and frame_ratio >= 0.006:
            block_type = "cream"

        if block_type == "green":
            est_cell_w = rw * 0.135
            est_cell_h = rh * 0.27
        elif block_type == "blue":
            est_cell_w = rw * 0.125
            est_cell_h = rh * 0.25
        elif block_type == "cream":
            est_cell_w = rw * 0.125
            est_cell_h = rh * 0.24
        else:
            est_cell_w = rw * 0.140
            est_cell_h = rh * 0.28

        cols = int(round(w / max(1.0, est_cell_w)))
        rows = int(round(h / max(1.0, est_cell_h)))

        cols = max(1, min(6, cols))
        rows = max(1, min(3, rows))

        if w > rw * 0.25 and cols < 2:
            cols = 2

        if h > rh * 0.34 and rows < 2:
            rows = 2

        cv2.rectangle(
            debug_grid,
            (x, y),
            (x + w, y + h),
            color_by_type(block_type),
            3
        )

        cv2.putText(
            debug_grid,
            f"{block_type} {cols}x{rows}",
            (x + 5, max(25, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_by_type(block_type),
            2,
            cv2.LINE_AA
        )

        for r in range(rows):
            for c in range(cols):

                x1 = x + int(c * w / cols)
                x2 = x + int((c + 1) * w / cols)
                y1 = y + int(r * h / rows)
                y2 = y + int((r + 1) * h / rows)

                cw = x2 - x1
                ch = y2 - y1

                if cw < rw * 0.045:
                    continue

                if ch < rh * 0.060:
                    continue

                if y2 > bottom_cut:
                    continue

                area = float(max(1, cw * ch))

                g = cv2.countNonZero(
                    green_mask[y1:y2, x1:x2]
                ) / area

                b = cv2.countNonZero(
                    blue_mask[y1:y2, x1:x2]
                ) / area

                cr = cv2.countNonZero(
                    cream_mask[y1:y2, x1:x2]
                ) / area

                wd = cv2.countNonZero(
                    wood_mask[y1:y2, x1:x2]
                ) / area

                li = cv2.countNonZero(
                    frame_mask[y1:y2, x1:x2]
                ) / area

                material_score = max(g, b, cr, wd)
                frame_score = max(material_score, li)

                # เช็กครึ่งล่างเพื่อตัดผนัง / เพดาน
                lower_y1 = y1 + int(ch * 0.45)
                lower_area = float(max(1, (y2 - lower_y1) * cw))

                lg = cv2.countNonZero(
                    green_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lb = cv2.countNonZero(
                    blue_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lcr = cv2.countNonZero(
                    cream_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lwd = cv2.countNonZero(
                    wood_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lli = cv2.countNonZero(
                    frame_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lower_score = max(lg, lb, lcr, lwd, lli)

                # ต้องมีขอบหรือวัสดุจริง
                if frame_score < 0.006:
                    continue

                if lower_score < 0.004:
                    continue

                # กันโซนสูงที่เป็นผนัง
                if y1 < rh * 0.35 and frame_score < 0.018:
                    continue

                t = classify_color_cell(
                    x1,
                    y1,
                    x2,
                    y2
                )

                # ถ้าจำแนกไม่ได้ แต่มี frame ให้ใช้ block_type
                if t == "unknown" and li >= 0.006:
                    t = block_type

                cells.append((x1, y1, cw, ch))
                cell_types.append(t)

                cv2.rectangle(
                    debug_grid,
                    (x1, y1),
                    (x2, y2),
                    color_by_type(t),
                    1
                )

    # =========================
    # FILTER SQUARE-LIKE CELLS
    # =========================
    filtered_cells = []
    filtered_types = []

    for i, (x, y, w, h) in enumerate(cells):

        t = cell_types[i]
        aspect = w / float(max(1, h))

        # กรอบพาเลทควรไม่เป็นเส้นแคบ
        if w < rw * 0.050:
            continue

        if h < rh * 0.065:
            continue

        # กันกล่องใหญ่ครอบทั้งชุด
        if w > rw * 0.34:
            continue

        if h > rh * 0.45:
            continue

        # ลักษณะใกล้สี่เหลี่ยม / rectangle pallet
        if aspect < 0.35:
            continue

        if aspect > 3.80:
            continue

        filtered_cells.append((x, y, w, h))
        filtered_types.append(t)

    cells = filtered_cells
    cell_types = filtered_types

    # =========================
    # SPLIT OVERSIZE CELLS
    # แตกกรอบใหญ่เป็น pallet ย่อย
    # =========================
    def split_oversize_cells(cells, cell_types):
        new_cells = []
        new_types = []

        for i, (x, y, w, h) in enumerate(cells):
            original_type = cell_types[i]

            target_w = rw * 0.13
            target_h = rh * 0.25

            cols = 1
            rows = 1

            # ถ้ากว้างเกิน ให้แตกแนว X
            if w > target_w * 1.65:
                cols = int(round(w / target_w))
                cols = max(2, min(cols, 4))

            # ถ้าสูงเกิน ให้แตกแนว Y
            if h > target_h * 1.55:
                rows = int(round(h / target_h))
                rows = max(2, min(rows, 3))

            for r in range(rows):
                for c in range(cols):
                    x1 = x + int(c * w / cols)
                    x2 = x + int((c + 1) * w / cols)
                    y1 = y + int(r * h / rows)
                    y2 = y + int((r + 1) * h / rows)

                    cw = x2 - x1
                    ch = y2 - y1

                    if cw < rw * 0.045:
                        continue

                    if ch < rh * 0.060:
                        continue

                    if y2 > bottom_cut:
                        continue

                    area = float(max(1, cw * ch))

                    g = cv2.countNonZero(green_mask[y1:y2, x1:x2]) / area
                    b = cv2.countNonZero(blue_mask[y1:y2, x1:x2]) / area
                    c_ratio = cv2.countNonZero(cream_mask[y1:y2, x1:x2]) / area
                    w_ratio = cv2.countNonZero(wood_mask[y1:y2, x1:x2]) / area
                    f = cv2.countNonZero(frame_mask[y1:y2, x1:x2]) / area

                    if max(g, b, c_ratio, w_ratio, f) < 0.006:
                        continue

                    cell_type = classify_color_cell(
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    if cell_type == "unknown":
                        cell_type = original_type

                    new_cells.append((x1, y1, cw, ch))
                    new_types.append(cell_type)

        return new_cells, new_types

    cells, cell_types = split_oversize_cells(
        cells,
        cell_types
    )

    # =========================
    # NMS
    # =========================
    cells, cell_types = nms_iou(
        cells,
        cell_types,
        iou_th=0.22
    )

    # =========================
    # FINAL DRAW + COUNT
    # =========================
    counts = {
        "green": 0,
        "blue": 0,
        "cream": 0,
        "wood": 0,
        "unknown": 0
    }

    for i, (x, y, w, h) in enumerate(cells, 1):

        t = cell_types[i - 1]

        if t not in counts:
            counts[t] = 0

        counts[t] += 1

        if t == "green":
            prefix = "G"
        elif t == "blue":
            prefix = "B"
        elif t == "cream":
            prefix = "C"
        elif t == "wood":
            prefix = "W"
        else:
            prefix = "U"

        cv2.rectangle(
            debug_box,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            debug_box,
            f"{prefix}{counts[t]}",
            (x + 6, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    total = len(cells)

    cv2.putText(
        debug_box,
        f"T={total} G={counts['green']} B={counts['blue']} C={counts['cream']} W={counts['wood']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("PALLET FRAME SQUARE DETECTION + OVERSIZE SPLIT")
    print(f"BLOCKS : {len(blocks)}")
    print(f"TOTAL  : {total}")
    print(f"GREEN  : {counts['green']}")
    print(f"BLUE   : {counts['blue']}")
    print(f"CREAM  : {counts['cream']}")
    print(f"WOOD   : {counts['wood']}")
    print("=" * 50)

    # =========================
    # DEBUG SAVE
    # =========================
    if debug:
        save_dbg("debug_original.jpg", roi)
        save_dbg("debug_normalized.jpg", norm)
        save_dbg("debug_green_mask.jpg", green_mask)
        save_dbg("debug_blue_mask.jpg", blue_mask)
        save_dbg("debug_cream_mask.jpg", cream_mask)
        save_dbg("debug_wood_mask.jpg", wood_mask)
        save_dbg("debug_edges.jpg", edge_base)
        save_dbg("debug_line_mask.jpg", line_mask)
        save_dbg("debug_frame_mask.jpg", frame_mask)
        save_dbg("debug_cargo.jpg", pallet_seed)
        save_dbg("debug_grid.jpg", debug_grid)
        save_dbg("debug_pallet_box.jpg", debug_box)

    return total

# =========================
# APPSHEET
# =========================
def update_appsheet(row_id, text):
    if not APP_ID or not ACCESS_KEY:
        return

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{TABLE_NAME}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action":"Edit",
        "Rows":[
            {"ID":row_id,"TFR AI":text,"status":"Done"}
        ]
    }

    try:
        requests.post(url,json=payload,headers=headers,timeout=10)
    except:
        pass


# =========================
# DEBUG SERVER
# =========================
@app.route("/debug/<path:filename>")
def debug_file(filename):
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.exists(path):
        return "Not found",404
    return send_file(path)


@app.route("/debug-list")
def debug_list():
    return jsonify(os.listdir(DEBUG_DIR))


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    print("="*40)
    print("REQUEST:", data)
    print("="*40)

    image_url = data.get("link")
    row_id = data.get("id")

    project = str(data.get("project","")).strip().lower()

    if project not in ["inbound", "outbound"]:
        print("⚠️ INVALID PROJECT → FORCE INBOUND")
        project = "inbound"

    print("PROJECT =", project)

    if not image_url or not row_id:
        return {"error":"missing"},400

    img = download_image(image_url)
    if img is None:
        return {"error":"image fail"},400

    if project == "inbound":
        print("RUN PALLET MODEL")
        result = str(gen_pallet(img))

    else:
        print("RUN VOLUME MODEL")
        result = f"{gen_volume(img)}%"

    update_appsheet(row_id, result)

    return {
        "status":"success",
        "project": project,
        "result": result
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
