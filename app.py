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
        print("HEADERS:")
        print(dict(request.headers))

        print("=" * 80)
        print("RAW DATA:")
        print(request.data)

        data = request.get_json(silent=True)

        print("=" * 80)
        print("JSON:")
        print(data)

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON received"
            }), 400

        # รองรับหลายชื่อ field
        image_url = (
            data.get("link")
            or data.get("linkapp")
            or data.get("url")
            or data.get("image_url")
        )

        print("=" * 80)
        print("IMAGE URL:")
        print(image_url)

        if not image_url:
            return jsonify({
                "status": "error",
                "message": "Image URL not found"
            }), 400

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            image_url,
            headers=headers,
            timeout=30,
            allow_redirects=True
        )

        print("=" * 80)
        print("DOWNLOAD STATUS:", response.status_code)
        print("FINAL URL:", response.url)
        print("CONTENT TYPE:", response.headers.get("Content-Type"))
        print("CONTENT LENGTH:", len(response.content))

        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": f"Download failed ({response.status_code})"
            }), 400

        img_array = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        print("=" * 80)
        print("IMAGE OBJECT:", img is not None)

        if img is None:

            preview = response.text[:500]

            print("RESPONSE PREVIEW:")
            print(preview)

            return jsonify({
                "status": "error",
                "message": "Cannot decode image",
                "content_type": response.headers.get("Content-Type")
            }), 400

        h, w = img.shape[:2]

        print("=" * 80)
        print("IMAGE SIZE:", w, "x", h)

        return jsonify({
            "status": "success",
            "width": int(w),
            "height": int(h)
        })

    except Exception as e:

        print("=" * 80)
        print("EXCEPTION")

        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
