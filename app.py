from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        print("=" * 80)
        print(data)

        image_url = data.get("link")

        print("IMAGE URL:", image_url)

        response = requests.get(
            image_url,
            timeout=60,
            allow_redirects=True
        )

        print("STATUS:", response.status_code)
        print("FINAL URL:", response.url)
        print("CONTENT TYPE:", response.headers.get("Content-Type"))
        print("SIZE:", len(response.content))

        img_array = np.frombuffer(
            response.content,
            np.uint8
        )

        img = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        print("IMAGE OBJECT:", img is not None)

        if img is None:

            return jsonify({
                "status": "error",
                "message": "decode failed"
            })

        h, w = img.shape[:2]

        print("IMAGE SIZE:", w, "x", h)

        return jsonify({
            "status": "success",
            "width": int(w),
            "height": int(h)
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
