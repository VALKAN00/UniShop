from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model



app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load components
pipeline = joblib.load("full_pipeline.joblib")
model = load_model("deep_learning_model.keras")
label_encoder = joblib.load("label_encoder.joblib")  # Make sure this was saved during training

# Expected input features
expected_features = [
    "price", "college", "Sub Category", "rating", "stock", "is_out_of_stock",
    "views_count", "wishlist_count", "add_to_cart_count", "buyers_count",
    "discount_pct", "final_price", "stock_range", "seasonality",
    "effective_price", "cart_to_view_ratio", "wishlist_to_view_ratio"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the incoming POST request
        data = request.get_json()

        # Ensure all required fields are present
        if not all(feature in data for feature in expected_features):
            return jsonify({"error": "Missing one or more required input fields."}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])[expected_features]

        # Transform using pipeline
        processed_data = pipeline.transform(input_df)

        # Predict using the model
        probs = model.predict(processed_data)
        pred_num = np.argmax(probs, axis=1)  # Get the predicted class
        pred_label = label_encoder.inverse_transform(pred_num)  # Convert to original label

        # Return the prediction
        return jsonify({
            "predicted_price_range_category": pred_label[0]  # Send the predicted category
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
