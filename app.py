@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    return {
        "status": "received",
        "data": data
    }
