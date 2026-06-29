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
            return (255, 130, 0)
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

    def nms_iou(boxes, types, iou_th=0.18):
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

                if small_area > 0 and inter / float(small_area) > 0.55:
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
        int(H * 0.12):int(H * 0.86),
        int(W * 0.02):int(W * 0.98)
    ]

    rh, rw = roi.shape[:2]

    top_cut = int(rh * 0.15)
    bottom_cut = int(rh * 0.84)

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
    green_mask = cv2.inRange(
        hsv,
        (30, 28, 28),
        (100, 255, 255)
    )

    blue_mask = cv2.inRange(
        hsv,
        (85, 30, 30),
        (140, 255, 255)
    )

    cream_mask = cv2.inRange(
        hsv,
        (4, 18, 65),
        (48, 200, 255)
    )

    wood_mask = cv2.inRange(
        hsv,
        (5, 30, 35),
        (38, 240, 255)
    )

    # ตัดหลังคา / พื้นล่าง / ขอบภาพ
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
    # EDGE / FRAME MASK
    # =========================
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_base = cv2.Canny(
        blur,
        30,
        120
    )

    edge_base[:top_cut, :] = 0
    edge_base[bottom_cut:, :] = 0

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(18, int(rh * 0.075)))
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(28, int(rw * 0.040)), 3)
    )

    vertical_lines = cv2.morphologyEx(
        edge_base,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    horizontal_lines = cv2.morphologyEx(
        edge_base,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    line_mask = cv2.bitwise_or(
        vertical_lines,
        horizontal_lines
    )

    # =========================
    # FRAME SEED ONLY
    # สำคัญ: ห้ามใช้ material_mask ทั้งก้อนเป็น object seed
    # =========================

    # สีโครงที่มีโอกาสเป็น frame
    # ไม่ใส่ wood_mask เพราะ wood มักเป็นกล่องด้านใน ไม่ใช่ขอบพาเลท
    color_frame_mask = cv2.bitwise_or(
        green_mask,
        blue_mask
    )

    # cream ใช้เฉพาะบริเวณที่มีเส้น/ขอบร่วมด้วย
    cream_edge = cv2.bitwise_and(
        cream_mask,
        line_mask
    )

    color_frame_mask = cv2.bitwise_or(
        color_frame_mask,
        cream_edge
    )

    # edge เฉพาะที่อยู่ใกล้ material จริงเท่านั้น
    material_dilate = cv2.dilate(
        material_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25)),
        iterations=1
    )

    line_near_material = cv2.bitwise_and(
        line_mask,
        material_dilate
    )

    # สำหรับเส้นช่วงล่าง อนุญาตเฉพาะที่อยู่ใกล้ material เท่านั้น
    lower_zone = np.zeros_like(line_mask)
    lower_zone[int(rh * 0.28):bottom_cut, :] = 255

    line_lower = cv2.bitwise_and(
        line_mask,
        lower_zone
    )

    line_lower_near_material = cv2.bitwise_and(
        line_lower,
        material_dilate
    )

    frame_mask = cv2.bitwise_or(
        line_near_material,
        line_lower_near_material
    )

    frame_mask = cv2.bitwise_or(
        frame_mask,
        color_frame_mask
    )

    frame_mask[:top_cut, :] = 0
    frame_mask[bottom_cut:, :] = 0
    frame_mask[:, :int(rw * 0.01)] = 0
    frame_mask[:, int(rw * 0.99):] = 0

    frame_mask = cv2.morphologyEx(
        frame_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1
    )

    frame_mask = cv2.morphologyEx(
        frame_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # =========================
    # OBJECT SEED - FRAME ONLY
    # ใช้เฉพาะขอบ / frame / line เท่านั้น
    # =========================
    object_seed = frame_mask.copy()

    # =========================
    # CLASSIFY BY BORDER
    # =========================
    def classify_by_border(x, y, w, h):
        if w <= 0 or h <= 0:
            return "unknown"

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        border = max(6, int(min(w, h) * 0.13))

        border_mask = np.zeros((h, w), dtype=np.uint8)

        border_mask[:border, :] = 255
        border_mask[h - border:, :] = 255
        border_mask[:, :border] = 255
        border_mask[:, w - border:] = 255

        border_area = float(max(1, cv2.countNonZero(border_mask)))

        g_crop = green_mask[y1:y2, x1:x2]
        b_crop = blue_mask[y1:y2, x1:x2]
        c_crop = cream_mask[y1:y2, x1:x2]
        w_crop = wood_mask[y1:y2, x1:x2]
        f_crop = frame_mask[y1:y2, x1:x2]

        g = cv2.countNonZero(
            cv2.bitwise_and(g_crop, border_mask)
        ) / border_area

        bl = cv2.countNonZero(
            cv2.bitwise_and(b_crop, border_mask)
        ) / border_area

        c = cv2.countNonZero(
            cv2.bitwise_and(c_crop, border_mask)
        ) / border_area

        wd = cv2.countNonZero(
            cv2.bitwise_and(w_crop, border_mask)
        ) / border_area

        fr = cv2.countNonZero(
            cv2.bitwise_and(f_crop, border_mask)
        ) / border_area

        # ครึ่งล่างต้องมีหลักฐานจริง
        lower_y1 = y + int(h * 0.45)
        lower_area = float(max(1, (y + h - lower_y1) * w))

        lg = cv2.countNonZero(
            green_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lb = cv2.countNonZero(
            blue_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lc = cv2.countNonZero(
            cream_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lw = cv2.countNonZero(
            wood_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lf = cv2.countNonZero(
            frame_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lower_score = max(lg, lb, lc, lw, lf)

        if lower_score < 0.004:
            return "unknown"

        # blue ต้องชัดจากขอบจริง
        if bl > 0.030 and bl > g * 1.25 and bl > c * 1.35:
            return "blue"

        # green
        if g > 0.022 and g > bl * 1.15 and g > c * 1.15:
            return "green"

        # wood / carton
        if wd > 0.050 and wd > c * 1.20 and wd > g:
            return "wood"

        # cream / cage / frame
        if c > 0.010 or fr > 0.008:
            return "cream"

        return "unknown"

    # =========================
    # VALID PALLET BOX
    # ห้ามเล็ก / แบน / ใหญ่เกิน
    # =========================
    def valid_pallet_box(x, y, w, h):

        if y < top_cut:
            return False

        if y + h > bottom_cut:
            return False

        # ไม่เล็ก
        if w < rw * 0.055:
            return False

        if h < rh * 0.080:
            return False

        # ไม่ใหญ่เกิน
        if w > rw * 0.48:
            return False

        if h > rh * 0.62:
            return False

        aspect = w / float(max(1, h))

        # ไม่เป็นเสา ไม่เป็นคานแบน
        if aspect < 0.32:
            return False

        if aspect > 4.20:
            return False

        area = float(max(1, w * h))

        frame_den = cv2.countNonZero(
            frame_mask[y:y + h, x:x + w]
        ) / area

        material_den = cv2.countNonZero(
            material_mask[y:y + h, x:x + w]
        ) / area

        # ต้องมีขอบพอจริง
        if frame_den < 0.004 and material_den < 0.004:
            return False

        # ไม่ให้จับผนังเรียบที่ไม่มี content ในครึ่งล่าง
        lower_y1 = y + int(h * 0.45)
        lower_area = float(max(1, (y + h - lower_y1) * w))

        lower_frame = cv2.countNonZero(
            frame_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        lower_material = cv2.countNonZero(
            material_mask[lower_y1:y + h, x:x + w]
        ) / lower_area

        if max(lower_frame, lower_material) < 0.003:
            return False

        return True

    # =========================
    # FIND COMPONENTS FROM FRAME ONLY
    # =========================
    contours, _ = cv2.findContours(
        object_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_candidates = []
    raw_types = []

    debug_reject = roi.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        if area < rw * rh * 0.0012:
            continue

        if w < rw * 0.030:
            continue

        if h < rh * 0.045:
            continue

        if not valid_pallet_box(x, y, w, h):
            cv2.rectangle(
                debug_reject,
                (x, y),
                (x + w, y + h),
                (80, 80, 80),
                1
            )
            continue

        t = classify_by_border(x, y, w, h)

        if t == "unknown":
            cv2.rectangle(
                debug_reject,
                (x, y),
                (x + w, y + h),
                (200, 200, 200),
                1
            )
            continue

        raw_candidates.append((x, y, w, h))
        raw_types.append(t)

    candidates, candidate_types = nms_iou(
        raw_candidates,
        raw_types,
        iou_th=0.18
    )

    # =========================
    # FINAL DRAW
    # =========================
    debug_box = roi.copy()

    counts = {
        "green": 0,
        "blue": 0,
        "cream": 0,
        "wood": 0,
        "unknown": 0
    }

    for i, (x, y, w, h) in enumerate(candidates, 1):
        t = candidate_types[i - 1]

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

    total = len(candidates)

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
    print("PALLET FRAME ONLY DETECTION - NO GRID")
    print(f"RAW    : {len(raw_candidates)}")
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
        save_dbg("debug_color_frame.jpg", color_frame_mask)
        save_dbg("debug_cargo.jpg", object_seed)
        save_dbg("debug_reject.jpg", debug_reject)
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
