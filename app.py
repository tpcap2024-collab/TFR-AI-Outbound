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
# แนะนำให้ตั้งใน Render Environment Variables
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
        if img is None:
            print("SAVE DEBUG ERROR:", filename, "image is None")
            return False

        path = os.path.join(DEBUG_DIR, filename)

        ok = cv2.imwrite(path, img)

        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0

        print(
            f"SAVE DEBUG {filename}: "
            f"ok={ok} exists={exists} size={size} path={path}"
        )

        return ok

    except Exception as e:
        print("SAVE DEBUG ERROR:", filename, e)
        print(traceback.format_exc())
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
# VOLUME MODEL - OUTBOUND
# =========================
def gen_volume(img, debug=True, return_empty=False):

    if img is None or img.size == 0:
        return 0

    orig_h, orig_w = img.shape[:2]
    view_type = "rear" if orig_h > orig_w else "side"

    img = cv2.resize(img, (640, 480))

    if view_type == "side":
        h, w = img.shape[:2]
        target_h = int(w * 9 / 16)
        top = max(0, (h - target_h) // 2)
        img = img[top:top + target_h, :]

    h, w = img.shape[:2]

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

    container_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(l)

    roi_norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    h_channel, s_channel, v_channel = cv2.split(hsv)

    v_mean = float(v_channel.mean())
    s_mean = float(s_channel.mean())

    green_mask = cv2.inRange(hsv, (35, 45, 45), (95, 255, 255))
    brown_mask = cv2.inRange(hsv, (5, 45, 45), (35, 255, 230))
    blue_mask = cv2.inRange(hsv, (85, 35, 35), (125, 255, 255))

    red_mask_1 = cv2.inRange(hsv, (0, 60, 50), (12, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 60, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    dark_mask = cv2.inRange(hsv, (0, 55, 0), (180, 255, 65))

    adaptive_texture = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    strong_saturation_mask = cv2.inRange(s_channel, 70, 255)
    strong_low_value_mask = cv2.inRange(v_channel, 0, 75)

    texture_candidate = cv2.bitwise_and(
        strong_saturation_mask,
        strong_low_value_mask
    )

    texture_mask = cv2.bitwise_and(
        adaptive_texture,
        texture_candidate
    )

    edges = cv2.Canny(gray_blur, 40, 120)

    edge_density = cv2.blur(
        edges.astype(np.float32),
        (15, 15)
    )

    edge_mask = cv2.inRange(edge_density, 10, 255)

    texture_mask = cv2.bitwise_and(
        texture_mask,
        edge_mask
    )

    top_suppress_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    top_cut_ratio = 0.12 if view_type == "rear" else 0.16
    top_cut = int(rh * top_cut_ratio)

    top_suppress_mask[:top_cut, :] = 0

    texture_mask = cv2.bitwise_and(texture_mask, top_suppress_mask)
    dark_mask = cv2.bitwise_and(dark_mask, top_suppress_mask)

    color_cargo_mask = cv2.bitwise_or(green_mask, brown_mask)
    color_cargo_mask = cv2.bitwise_or(color_cargo_mask, blue_mask)
    color_cargo_mask = cv2.bitwise_or(color_cargo_mask, red_mask)

    cargo_mask = cv2.bitwise_or(color_cargo_mask, dark_mask)
    cargo_mask = cv2.bitwise_or(cargo_mask, texture_mask)
    cargo_mask = cv2.bitwise_and(cargo_mask, container_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

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

        if 0.20 <= aspect_ratio <= 6.50:
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                thickness=-1
            )

    cargo_mask = filtered_mask

    raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    if raw_cargo_ratio > 0.95:
        print("WARNING: cargo over-detected, fallback to color only")

        cargo_mask = color_cargo_mask.copy()
        cargo_mask = cv2.bitwise_or(cargo_mask, dark_mask)
        cargo_mask = cv2.bitwise_and(cargo_mask, container_mask)

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

    empty_mask = cv2.bitwise_and(
        container_mask,
        cv2.bitwise_not(cargo_mask)
    )

    y = np.linspace(0, 1, rh, dtype=np.float32).reshape(rh, 1)

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

    filled_ratio = float(np.clip(filled_ratio, 0, 1))
    empty_ratio = float(np.clip(empty_ratio, 0, 1))

    filled_volume = (filled_ratio ** 0.95) * 100
    filled_volume = filled_volume * 0.95
    filled_volume = float(np.clip(filled_volume, 0, 100))

    empty_volume = 100 - filled_volume

    output_volume = empty_volume if return_empty else filled_volume
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

    if debug:
        color_layer = roi_norm.copy()

        color_layer[cargo_mask > 0] = (0, 255, 0)
        color_layer[empty_mask > 0] = (255, 0, 0)

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

        empty_contours, _ = cv2.findContours(
            empty_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

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


def gen_pallet(img, debug=True):
    save_dbg(name, im):    cells = []
        if not debug or im is None:
            return
        try:
            save_debug(name, im)
        except Exception:
            cv2.imwrite(os.path.join(DEBUG_DIR, name), im)

    def add_cell(box, cell_type):
        cells.append(box)
        cell_types.append(cell_type)

    def color_by_type(t):
        if t == "cream":
            return (0, 255, 0)
        if t == "blue":
            return (255, 180, 0)
        if t == "wood":
            return (0, 180, 255)
        return (0, 255, 255)

    def draw_split(draw, box, cols, rows, cell_type, prefix):
        x, y, w, h = box
        color = color_by_type(cell_type)

        for c in range(cols + 1):
            gx = x + int(c * w / cols)
            cv2.line(draw, (gx, y), (gx, y + h), (255, 255, 0), 1)

        for r in range(rows + 1):
            gy = y + int(r * h / rows)
            cv2.line(draw, (x, gy), (x + w, gy), (255, 255, 0), 1)

        no = 1

        for r in range(rows):
            for c in range(cols):
                x1 = x + int(c * w / cols)
                x2 = x + int((c + 1) * w / cols)
                y1 = y + int(r * h / rows)
                y2 = y + int((r + 1) * h / rows)

                add_cell(
                    (x1, y1, x2 - x1, y2 - y1),
                    cell_type
                )

                cv2.rectangle(
                    draw,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    draw,
                    f"{prefix}{no}",
                    (x1 + 6, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )

                no += 1

    if img is not None and img.size > 0:
        img = cv2.resize(img, (1280, 720))
        H, W = img.shape[:2]

        # ROI ตามที่ใช้งานได้แล้ว
        roi = img[
            int(H * 0.17):int(H * 0.84),
            int(W * 0.01):int(W * 0.99)
        ]

        if roi.size > 0:
            rh, rw = roi.shape[:2]

            debug_blocks = roi.copy()
            debug_grid = roi.copy()
            debug_count = roi.copy()

            # =========================
            # FIXED LAYOUT FROM IMAGE
            # =========================
            cargo_y1 = int(rh * 0.13)
            cargo_y2 = int(rh * 0.88)
            cargo_h = cargo_y2 - cargo_y1

            # ซ้าย = กล่องไม้ / กระดาษไม้ 3
            wood_x1 = int(rw * 0.00)
            wood_x2 = int(rw * 0.245)

            # กลางซ้าย = Rack น้ำเงิน 2
            blue_x1 = int(rw * 0.245)
            blue_x2 = int(rw * 0.420)

            # ขวา = Cage ครีม 4x2 = 8
            cream_x1 = int(rw * 0.420)
            cream_x2 = int(rw * 0.995)

            wood_box = (
                wood_x1,
                cargo_y1,
                wood_x2 - wood_x1,
                cargo_h
            )

            blue_box = (
                blue_x1,
                cargo_y1,
                blue_x2 - blue_x1,
                cargo_h
            )

            cream_box = (
                cream_x1,
                cargo_y1,
                cream_x2 - cream_x1,
                cargo_h
            )

            # =========================
            # DRAW BLOCKS
            # =========================
            cv2.rectangle(
                debug_blocks,
                (wood_box[0], wood_box[1]),
                (wood_box[0] + wood_box[2], wood_box[1] + wood_box[3]),
                color_by_type("wood"),
                3
            )

            cv2.putText(
                debug_blocks,
                "WOOD=3",
                (wood_box[0] + 6, wood_box[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_by_type("wood"),
                2,
                cv2.LINE_AA
            )

            cv2.rectangle(
                debug_blocks,
                (blue_box[0], blue_box[1]),
                (blue_box[0] + blue_box[2], blue_box[1] + blue_box[3]),
                color_by_type("blue"),
                3
            )

            cv2.putText(
                debug_blocks,
                "BLUE=2",
                (blue_box[0] + 6, blue_box[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_by_type("blue"),
                2,
                cv2.LINE_AA
            )

            cv2.rectangle(
                debug_blocks,
                (cream_box[0], cream_box[1]),
                (cream_box[0] + cream_box[2], cream_box[1] + cream_box[3]),
                color_by_type("cream"),
                3
            )

            cv2.putText(
                debug_blocks,
                "CREAM=8",
                (cream_box[0] + 6, cream_box[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_by_type("cream"),
                2,
                cv2.LINE_AA
            )

            # =========================
            # COUNT FIXED
            # =========================

            # กล่องไม้ = 3
            draw_split(
                debug_grid,
                wood_box,
                cols=1,
                rows=3,
                cell_type="wood",
                prefix="W"
            )

            # น้ำเงิน = 2
            draw_split(
                debug_grid,
                blue_box,
                cols=1,
                rows=2,
                cell_type="blue",
                prefix="B"
            )

            # ครีม = 8
            draw_split(
                debug_grid,
                cream_box,
                cols=4,
                rows=2,
                cell_type="cream",
                prefix="C"
            )

            # =========================
            # FINAL DRAW
            # =========================
            cream_count = sum(1 for t in cell_types if t == "cream")
            blue_count = sum(1 for t in cell_types if t == "blue")
            wood_count = sum(1 for t in cell_types if t == "wood")

            type_running = {
                "cream": 0,
                "blue": 0,
                "wood": 0
            }

            for x, y, w, h in cells:
                t = cell_types[cells.index((x, y, w, h))]
                type_running[t] += 1

                color = color_by_type(t)

                if t == "cream":
                    label = f"C{type_running[t]}"
                elif t == "blue":
                    label = f"B{type_running[t]}"
                else:
                    label = f"W{type_running[t]}"

                cv2.rectangle(
                    debug_count,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                cv2.putText(
                    debug_count,
                    label,
                    (x + 7, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    color,
                    2,
                    cv2.LINE_AA
                )

            cv2.putText(
                debug_count,
                f"CREAM={cream_count} BLUE={blue_count} WOOD={wood_count} TOTAL={len(cells)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                debug_blocks,
                "FIXED LAYOUT: WOOD=3 BLUE=2 CREAM=8",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                debug_grid,
                "WOOD 1x3 | BLUE 1x2 | CREAM 4x2",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 0),
                3,
                cv2.LINE_AA
            )

            print("=" * 50)
            print("PALLET FIXED LAYOUT DEBUG")
            print(f"ROI SIZE : {rw}x{rh}")
            print(f"CREAM    : {cream_count}")
            print(f"BLUE     : {blue_count}")
            print(f"WOOD     : {wood_count}")
            print(f"TOTAL    : {len(cells)}")
            print("=" * 50)

            # =========================
            # SAVE DEBUG
            # =========================
            save_dbg("debug_original.jpg", roi)
            save_dbg("debug_blocks.jpg", debug_blocks)
            save_dbg("debug_grid.jpg", debug_grid)
            save_dbg("debug_pallet_box.jpg", debug_count)

            # สร้าง mask dummy เพื่อให้ route เดิมไม่ error
            dummy_mask = np.zeros((rh, rw), dtype=np.uint8)

            cv2.rectangle(
                dummy_mask,
                (wood_box[0], wood_box[1]),
                (wood_box[0] + wood_box[2], wood_box[1] + wood_box[3]),
                255,
                -1
            )

            cv2.rectangle(
                dummy_mask,
                (blue_box[0], blue_box[1]),
                (blue_box[0] + blue_box[2], blue_box[1] + blue_box[3]),
                255,
                -1
            )

            cv2.rectangle(
                dummy_mask,
                (cream_box[0], cream_box[1]),
                (cream_box[0] + cream_box[2], cream_box[1] + cream_box[3]),
                255,
                -1
            )

            save_dbg("debug_pallet_mask.jpg", dummy_mask)
            save_dbg("debug_cargo.jpg", dummy_mask)
            save_dbg("debug_overlay.jpg", debug_count)

    return len(cells)
    cell_types = []




# =========================
# UPDATE APPSHEET
# =========================
def update_appsheet(row_id, volume_text):

    if not APP_ID or not ACCESS_KEY:
        print("APPSHEET SKIP: missing APP_ID or ACCESS_KEY")
        return

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
@app.route("/debug/<path:filename>", methods=["GET"])
def debug_file(filename):
    try:
        filename = os.path.basename(filename)

        if not filename.startswith("debug_"):
            return jsonify({
                "error": "file not allowed",
                "reason": "filename must start with debug_",
                "filename": filename
            }), 403

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return jsonify({
                "error": "file not allowed",
                "reason": "only jpg/jpeg/png allowed",
                "filename": filename
            }), 403

        path = os.path.join(DEBUG_DIR, filename)

        if not os.path.exists(path):
            return jsonify({
                "error": "debug file not found",
                "filename": filename,
                "path": path,
                "debug_dir": DEBUG_DIR,
                "files_in_debug_dir": sorted(os.listdir(DEBUG_DIR))
            }), 404

        mimetype = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

        return send_file(
            path,
            mimetype=mimetype,
            as_attachment=False,
            max_age=0
        )

    except Exception as e:
        print("DEBUG FILE ERROR:", e)
        print(traceback.format_exc())

        return jsonify({
            "error": "debug server error",
            "message": str(e)
        }), 500


@app.route("/debug-list", methods=["GET"])
def debug_list():
    try:
        base_url = request.host_url.rstrip("/")

        expected_files = [
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
            "debug_texture.jpg",
            "debug_pallet_mask.jpg",
            "debug_pallet_box.jpg",
            "debug_box_mask.jpg",
            "debug_blocks.jpg",
            "debug_grid.jpg",
            "debug_cream.jpg",
            "debug_white.jpg",
            "debug_green_inbound.jpg",
            "debug_blue_inbound.jpg",
            "debug_carton.jpg"
        ]

        actual_files = []

        if os.path.exists(DEBUG_DIR):
            actual_files = sorted([
                f for f in os.listdir(DEBUG_DIR)
                if f.startswith("debug_") and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        all_files = sorted(set(expected_files + actual_files))

        return jsonify({
            "status": "ok",
            "debug_dir": DEBUG_DIR,
            "actual_files_count": len(actual_files),
            "actual_files": actual_files,
            "files": [
                {
                    "file": f,
                    "url": f"{base_url}/debug/{f}",
                    "exists": os.path.exists(os.path.join(DEBUG_DIR, f))
                }
                for f in all_files
            ]
        })

    except Exception as e:
        print("DEBUG LIST ERROR:", e)
        print(traceback.format_exc())

        return jsonify({
            "error": "debug list server error",
            "message": str(e)
        }), 500


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
                return jsonify({
                    "status": "skipped",
                    "id": row_id
                }), 200

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
                "overlay": f"{base_url}/debug/debug_overlay.jpg",
                "pallet_mask": f"{base_url}/debug/debug_pallet_mask.jpg",
                "pallet_box": f"{base_url}/debug/debug_pallet_box.jpg",
                "blocks": f"{base_url}/debug/debug_blocks.jpg",
                "grid": f"{base_url}/debug/debug_grid.jpg",
                "cargo": f"{base_url}/debug/debug_cargo.jpg",
                "carton": f"{base_url}/debug/debug_carton.jpg",
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

        return jsonify({
            "error": "server error"
        }), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )
