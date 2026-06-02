@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    print("="*80)
    print(data)

    return jsonify({
        "status":"success"
    })
