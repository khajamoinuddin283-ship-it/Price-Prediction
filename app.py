from flask import Flask, render_template, request, jsonify
from pricing import predict_product_price_with_range_and_profit, ChipClassifier  # Import both

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract values from the request
        production_cost = float(data.get("production_cost", 0))
        labour_cost = float(data.get("labour_cost", 0))
        raw_material_cost = float(data.get("raw_material_cost", 0))
        rent = float(data.get("rent", 0))
        advertising = float(data.get("advertising", 0))
        transportation_cost_percentage = float(data.get("transportation_cost_percentage", 0))
        packet_size = float(data.get("packet_size", 0))
        gst_percentage = float(data.get("gst_percentage", 0))  # Will receive 0.18 or 0.12

        # Get predictions
        predicted_price, lower_bound, upper_bound = predict_product_price_with_range_and_profit(
            production_cost, labour_cost,
            raw_material_cost, rent, advertising,
            transportation_cost_percentage, packet_size, gst_percentage
        )

        # Classify packet size
        classifier = ChipClassifier()
        pack_type = classifier.classify(packet_size)

        # Return response
        return jsonify({
            "predicted_price": round(predicted_price, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "pack_type": pack_type
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/classify", methods=["POST"])
def classify():
    try:
        grams = float(request.form["grams"])
        classifier = ChipClassifier()
        category = classifier.classify(grams)
        return render_template("index.html", category=category)
    except Exception as e:
        return render_template("index.html", category="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
