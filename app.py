from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import traceback

app = Flask(__name__)

@app.route("/")
def home():
    return "TFR AI Running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("=" * 80)

        data = request.get_json(silent=True)
        print("JSON:", data)

        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 400

        image_url = data.get("link")
        print("IMAGE URL:", image_url)

        if not image_url:
            return jsonify({"status": "error", "message": "link empty"}), 400

        # ==============================
        # DOWNLOAD IMAGE
        # ==============================
        response = requests.get(
            image_url,
            timeout=60,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": f"download failed {response.status_code}"
            }), 400

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "decode failed"}), 400

        print("IMAGE OK")

        # ==============================
        # RESIZE (เหมือน Colab)
        # ==============================
        img_resized = cv2.resize(img, (800, 600))
        h, w = img_resized.shape[:2]

        # ==============================
        # HSV
        # ==============================
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5,5), 0)

        # ==============================
        # COLOR RANGE (Colab)
        # ==============================
        lower_egg = np.array([10, 30, 60])
        upper_egg = np.array([45, 255, 255])

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 70, 255])

        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])

        # ==============================
        # MASKS
        # ==============================
        mask_egg = cv2.inRange(hsv, lower_egg, upper_egg)

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # ==============================
        # COMBINE
        # ==============================
        combined_mask = cv2.bitwise_or(
            mask_egg,
            cv2.bitwise_or(
                mask_red,
                cv2.bitwise_or(
                    mask_blue,
                    cv2.bitwise_or(mask_green, mask_white)
                )
            )
        )

        # ==============================
        # CLEAN DARK NOISE
        # ==============================
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        texture_edges = cv2.Canny(gray, 50, 150)

        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        smooth_dark = cv2.bitwise_and(mask_dark, cv2.bitwise_not(texture_edges))

        combined_mask = cv2.bitwise_and(
            combined_mask,
            cv2.bitwise_not(smooth_dark)
        )

        # ==============================
        # ROI (สำคัญมาก)
        # ==============================
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        y1 = int(h * 0.18)
        y2 = int(h * 0.80)

        crop_mask = combined_mask[y1:y2, x1:x2]

        # ==============================
        # REMOVE CEILING
        # ==============================
        ceiling_cut = int(crop_mask.shape[0] * 0.12)
        crop_mask[0:ceiling_cut, :] = 0

        # ==============================
        # MORPHOLOGY CLEAN
        # ==============================
        kernel_open = np.ones((5,5), np.uint8)
        crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, kernel_open)

        kernel = np.ones((7,7), np.uint8)
        dilated = cv2.dilate(crop_mask, kernel, iterations=1)

        kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (25,9))
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_fill)

        # ==============================
        # CONNECTED COMPONENTS
        # ==============================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        clean_mask = np.zeros_like(dilated)

        ground_threshold = int(dilated.shape[0] * 0.75)

        total_object_area = 0
        valid_y_coords = []

        for label in range(1, num_labels):

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w_box = stats[label, cv2.CC_STAT_WIDTH]
            h_box = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            bottom_y = y + h_box

            if area < 1500:
                continue

            if bottom_y < ground_threshold and area < 5000:
                continue

            if w_box < 20 and h_box > 150:
                continue

            clean_mask[labels == label] = 255
            total_object_area += area
            valid_y_coords.append(y)

        dilated = clean_mask

        # ==============================
        # HEIGHT CALC
        # ==============================
        highest_point_y = crop_mask.shape[0]

        if len(valid_y_coords) > 0:
            highest_point_y = int(np.percentile(valid_y_coords, 15))

        height_fill_rate = ((crop_mask.shape[0] - highest_point_y) / crop_mask.shape[0]) * 100
        volume_percent = (total_object_area / (crop_mask.shape[0] * crop_mask.shape[1])) * 100

        height_fill_rate = int(round(height_fill_rate / 5) * 5)
        volume_percent = int(round(volume_percent / 5) * 5)

        height_fill_rate = max(0, min(100, height_fill_rate))
        volume_percent = max(0, min(100, volume_percent))

        if height_fill_rate >= 85:
            height_fill_rate = 100
        if volume_percent >= 85:
            volume_percent = 100

        # ==============================
        # RETURN RESULT
        # ==============================
        print("HEIGHT:", height_fill_rate)
        print("VOLUME:", volume_percent)

        return jsonify({
            "status": "success",
            "height_fill_rate": height_fill_rate,
            "volume_percent": volume_percent
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
