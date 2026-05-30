@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    return {
        "debug_data": data
    }
