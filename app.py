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
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if r.status_code != 200:
            return None

        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img

    except:
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(name, img):
    if img is None:
        return
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)


# =========================
# OUTBOUND (ง่าย version)
# =========================
def gen_volume(img, debug=True, return_empty=False):

    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    roi = img[int(h*0.25):int(h*0.80), int(w*0.1):int(w*0.9)]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    green = cv2.inRange(hsv, (35, 50, 50), (90,255,255))
    brown = cv2.inRange(hsv, (5,40,40), (35,255,255))
    blue  = cv2.inRange(hsv, (85,40,40), (130,255,255))
    dark  = cv2.inRange(hsv, (0,0,0), (180,255,60))

    edges = cv2.Canny(gray, 40, 120)

    cargo = green | brown | blue | dark | edges

    filled = cv2.countNonZero(cargo) / cargo.size * 100
    filled = min(100, filled)

    result = (100-filled) if return_empty else filled
    result = int(round(result/5)*5)

    if debug:
        overlay = roi.copy()
        overlay[cargo > 0] = (0,255,0)
        save_debug("debug_overlay.jpg", overlay)

    return result


def gen_pallet(img, debug=True):

    if img is None or img.size == 0:
        return 0

    def save_dbg(name, im):
        if debug and im is not None:
            save_debug(name, im)

    def clean(mask, k=5, close_it=1, open_it=1):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if close_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=close_it)

        if open_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=open_it)

        return mask

    def color_by_type(t):
        if t == "green":
            return (0, 255, 0)
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

    def nms_iou(cells, types, iou_th=0.22):
        if not cells:
            return cells, types

        items = []

        for i, b in enumerate(cells):
            x, y, w, h = b
            items.append((b, types[i], w * h))

        items = sorted(items, key=lambda v: v[2], reverse=True)

        keep_boxes = []
        keep_types = []

        for b, t, area in items:
            duplicate = False

            for old in keep_boxes:
                if box_iou(b, old) > iou_th:
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

    # =========================
    # NORMALIZE
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
    # MATERIAL MASKS
    # =========================

    # green rack
    green_mask = cv2.inRange(
        hsv,
        (30, 30, 30),
        (95, 255, 255)
    )

    # cream/beige cargo/rack
    # เพิ่ม S ขั้นต่ำเพื่อลดการจับผนังเทา/เพดาน
    cream_mask = cv2.inRange(
        hsv,
        (5, 32, 70),
        (45, 190, 255)
    )

    # wood/carton
    wood_mask = cv2.inRange(
        hsv,
        (5, 35, 45),
        (38, 230, 245)
    )

    # =========================
    # CUT ROOF / FLOOR / WHEEL
    # =========================
    top_cut = int(rh * 0.18)
    bottom_cut = int(rh * 0.80)

    for m in [green_mask, cream_mask, wood_mask]:
        m[:top_cut, :] = 0
        m[bottom_cut:, :] = 0
        m[:, :int(rw * 0.01)] = 0
        m[:, int(rw * 0.99):] = 0

    green_mask = clean(green_mask, 5, 1, 1)
    cream_mask = clean(cream_mask, 5, 1, 1)
    wood_mask = clean(wood_mask, 5, 1, 1)

    material_mask = cv2.bitwise_or(green_mask, cream_mask)
    material_mask = cv2.bitwise_or(material_mask, wood_mask)

    # =========================
    # EDGE ใช้ช่วยเฉพาะใกล้ material
    # =========================
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        50,
        150
    )

    edges[:top_cut, :] = 0
    edges[bottom_cut:, :] = 0

    material_dilate = cv2.dilate(
        material_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11)),
        iterations=1
    )

    edge_on_material = cv2.bitwise_and(edges, material_dilate)

    cargo_mask = cv2.bitwise_or(material_mask, edge_on_material)

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
        iterations=2
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # =========================
    # FIND BLOCKS
    # =========================
    contours, _ = cv2.findContours(
        cargo_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h < rw * rh * 0.004:
            continue

        if w < rw * 0.045:
            continue

        if h < rh * 0.070:
            continue

        if y + h > bottom_cut:
            continue

        if h > rh * 0.55:
            continue

        block_material = material_mask[y:y + h, x:x + w]
        block_density = cv2.countNonZero(block_material) / float(max(1, w * h))

        if block_density < 0.010:
            continue

        blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    cells = []
    cell_types = []

    debug_grid = roi.copy()
    debug_box = roi.copy()
    debug_reject = roi.copy()

    # =========================
    # SPLIT BLOCKS
    # =========================
    for (x, y, w, h) in blocks:

        block_area = float(max(1, w * h))

        g_ratio = cv2.countNonZero(green_mask[y:y+h, x:x+w]) / block_area
        c_ratio = cv2.countNonZero(cream_mask[y:y+h, x:x+w]) / block_area
        w_ratio = cv2.countNonZero(wood_mask[y:y+h, x:x+w]) / block_area

        if g_ratio >= max(c_ratio, w_ratio):
            block_type = "green"
        elif c_ratio >= max(g_ratio, w_ratio):
            block_type = "cream"
        else:
            block_type = "wood"

        # =========================
        # ESTIMATE GRID
        # =========================
        if block_type == "green":
            est_cell_w = rw * 0.135
            est_cell_h = rh * 0.28
        elif block_type == "cream":
            est_cell_w = rw * 0.125
            est_cell_h = rh * 0.25
        else:
            est_cell_w = rw * 0.14
            est_cell_h = rh * 0.28

        cols = int(round(w / max(1.0, est_cell_w)))
        rows = int(round(h / max(1.0, est_cell_h)))

        cols = max(1, min(6, cols))
        rows = max(1, min(3, rows))

        if w > rw * 0.25 and cols < 2:
            cols = 2

        if h > rh * 0.35 and rows < 2:
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

        # =========================
        # CREATE CELLS
        # =========================
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

                cell_area = float(max(1, cw * ch))

                # full cell material
                cell_green = cv2.countNonZero(
                    green_mask[y1:y2, x1:x2]
                ) / cell_area

                cell_cream = cv2.countNonZero(
                    cream_mask[y1:y2, x1:x2]
                ) / cell_area

                cell_wood = cv2.countNonZero(
                    wood_mask[y1:y2, x1:x2]
                ) / cell_area

                material_score = max(cell_green, cell_cream, cell_wood)

                # =========================
                # สำคัญมาก:
                # ตรวจครึ่งล่างของ cell
                # ถ้าเป็นผนัง/เพดาน จะไม่มี material ในครึ่งล่าง
                # =========================
                lower_y1 = y1 + int(ch * 0.45)
                lower_area = float(max(1, (y2 - lower_y1) * cw))

                lower_green = cv2.countNonZero(
                    green_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lower_cream = cv2.countNonZero(
                    cream_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lower_wood = cv2.countNonZero(
                    wood_mask[lower_y1:y2, x1:x2]
                ) / lower_area

                lower_score = max(lower_green, lower_cream, lower_wood)

                # cell ที่เป็นผนัง/เพดานจะตกเงื่อนไขนี้
                if material_score < 0.012:
                    cv2.rectangle(debug_reject, (x1, y1), (x2, y2), (80, 80, 80), 1)
                    continue

                if lower_score < 0.010:
                    cv2.rectangle(debug_reject, (x1, y1), (x2, y2), (150, 150, 150), 1)
                    continue

                # extra reject: cell อยู่สูงเกินและ material ต่ำ
                if y1 < rh * 0.35 and material_score < 0.025:
                    cv2.rectangle(debug_reject, (x1, y1), (x2, y2), (200, 200, 200), 1)
                    continue

                if cell_green >= max(cell_cream, cell_wood):
                    t = "green"
                elif cell_cream >= max(cell_green, cell_wood):
                    t = "cream"
                else:
                    t = "wood"

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
    # FILTER BAD CELLS
    # =========================
    def filter_bad_cells(cells, cell_types):
        new_cells = []
        new_types = []

        for i, (x, y, w, h) in enumerate(cells):
            t = cell_types[i]
            aspect = w / float(max(1, h))

            if w < rw * 0.050:
                continue

            if h < rh * 0.070:
                continue

            if w > rw * 0.32:
                continue

            if h > rh * 0.42:
                continue

            if t in ["cream", "wood"]:
                if aspect < 0.40 or aspect > 3.50:
                    continue

            new_cells.append((x, y, w, h))
            new_types.append(t)

        return new_cells, new_types

    # =========================
    # JIGSAW NORMALIZE
    # =========================
    def normalize_jigsaw_cells(cells, cell_types):
        if not cells:
            return cells, cell_types

        widths = []
        heights = []

        for x, y, w, h in cells:
            if w > rw * 0.06:
                widths.append(w)
            if h > rh * 0.10:
                heights.append(h)

        ref_w = int(np.median(widths)) if widths else int(rw * 0.13)
        ref_h = int(np.median(heights)) if heights else int(rh * 0.25)

        new_cells = []
        new_types = []

        for i, (x, y, w, h) in enumerate(cells):
            t = cell_types[i]

            cx = x + w / 2.0
            cy = y + h / 2.0

            nw = w
            nh = h

            if h < ref_h * 0.55:
                nh = ref_h

            if w < ref_w * 0.55:
                nw = ref_w

            nx = int(cx - nw / 2)
            ny = int(cy - nh / 2)

            nx = max(0, nx)
            ny = max(top_cut, ny)

            if nx + nw > rw:
                nx = rw - nw

            if ny + nh > bottom_cut:
                ny = bottom_cut - nh

            nx = max(0, nx)
            ny = max(top_cut, ny)

            x1 = int(nx)
            y1 = int(ny)
            x2 = int(nx + nw)
            y2 = int(ny + nh)

            area = float(max(1, (x2 - x1) * (y2 - y1)))

            g = cv2.countNonZero(green_mask[y1:y2, x1:x2]) / area
            c = cv2.countNonZero(cream_mask[y1:y2, x1:x2]) / area
            wmat = cv2.countNonZero(wood_mask[y1:y2, x1:x2]) / area

            material_score = max(g, c, wmat)

            lower_y1 = y1 + int((y2 - y1) * 0.45)
            lower_area = float(max(1, (y2 - lower_y1) * (x2 - x1)))

            lg = cv2.countNonZero(green_mask[lower_y1:y2, x1:x2]) / lower_area
            lc = cv2.countNonZero(cream_mask[lower_y1:y2, x1:x2]) / lower_area
            lw = cv2.countNonZero(wood_mask[lower_y1:y2, x1:x2]) / lower_area

            lower_score = max(lg, lc, lw)

            if material_score < 0.010:
                continue

            if lower_score < 0.008:
                continue

            new_cells.append((x1, y1, x2 - x1, y2 - y1))
            new_types.append(t)

        return new_cells, new_types

    # =========================
    # POST PROCESSING
    # =========================
    cells, cell_types = normalize_jigsaw_cells(cells, cell_types)
    cells, cell_types = filter_bad_cells(cells, cell_types)
    cells, cell_types = nms_iou(cells, cell_types, iou_th=0.22)

    # =========================
    # FINAL DRAW
    # =========================
    counts = {
        "green": 0,
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
        f"T={total} G={counts['green']} C={counts['cream']} W={counts['wood']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    print("=" * 50)
    print("PALLET MATERIAL GRID + LOWER MATERIAL CHECK")
    print(f"BLOCKS : {len(blocks)}")
    print(f"TOTAL  : {total}")
    print(f"GREEN  : {counts['green']}")
    print(f"CREAM  : {counts['cream']}")
    print(f"WOOD   : {counts['wood']}")
    print("=" * 50)

    if debug:
        save_dbg("debug_original.jpg", roi)
        save_dbg("debug_normalized.jpg", norm)
        save_dbg("debug_green_mask.jpg", green_mask)
        save_dbg("debug_cream_mask.jpg", cream_mask)
        save_dbg("debug_wood_mask.jpg", wood_mask)
        save_dbg("debug_edges.jpg", edges)
        save_dbg("debug_cargo.jpg", cargo_mask)
        save_dbg("debug_grid.jpg", debug_grid)
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
