from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import sys
print("Starting Flask server...")

app = Flask(__name__)
CORS(app)  # enable CORS for all domains for testing

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

@app.route('/api/predict', methods=['POST'])
def predict():
    features = request.get_json()  # expect JSON with inputs for predict.py

    # Run the predict.py script with the data via stdin
    proc = subprocess.Popen(
        [sys.executable, 'predict.py'], 
        cwd=MODEL_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(json.dumps(features))

    if proc.returncode != 0:
        return jsonify({'error': stderr}), 500
    
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON output from prediction script'}), 500
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
