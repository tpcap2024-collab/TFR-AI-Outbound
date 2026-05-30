@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("HEADERS:", request.headers)

        print("RAW DATA:", request.data)

        print("JSON:", request.get_json(silent=True))

        return jsonify({
            "status":"received"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
