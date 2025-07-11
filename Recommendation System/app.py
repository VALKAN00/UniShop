from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load data and similarity matrix
df = joblib.load('products.pkl')
similarity_matrix = joblib.load('similarity.pkl')

# Recommendation function
def recommend(product_id, top_n=5):
    if product_id not in df['id'].values:
        return []
    
    product_index = df[df['id'] == product_id].index[0]
    sim_scores = list(enumerate(similarity_matrix[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]

    return df.iloc[recommended_indices].to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend_get():
    try:
        product_id = int(request.args.get('id'))
        top_n = int(request.args.get('top_n', 5))
        results = recommend(product_id, top_n)
        if not results:
            return jsonify({'error': 'Product ID not found'}), 404
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# example: { "id": 1234, "top_n": 5 }
@app.route('/recommend', methods=['POST'])
def recommend_post():
    try:
        data = request.get_json()
        product_id = int(data.get('id'))
        top_n = int(data.get('top_n', 5))
        results = recommend(product_id, top_n)
        if not results:
            return jsonify({'error': 'Product ID not found'}), 404
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
