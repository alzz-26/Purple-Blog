from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Add the backend/model directory to the Python path
ml_path = os.path.join(os.path.dirname(__file__), 'backend', 'ml')
if ml_path not in sys.path:
    sys.path.append(ml_path)
    
from predict import predict_engagement

app = Flask(__name__)
CORS(app) # Enable CORS

@app.route('/')
def home():
    return "Flask server is running."

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        
        # Get all three inputs from the request
        description = data['description']
        followers = int(data['followers'])
        views = int(data['views'])
        
        if not description or followers < 0 or views <= 0:
            return jsonify({'error': 'Invalid input provided.'}), 400

        # Pass all three arguments to the function
        prediction = predict_engagement(description, followers, views)

        return jsonify(prediction)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred on the server.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)