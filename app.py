from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/fish_model.keras')
IMG_SIZE = (224, 224)  # ← just hard-code it for now
CLASS_NAMES = [
    'devario_malabaricus',
    'devario_memorialis', 
    'devario_micronema',
    'devario_monticola',
    'devario_pathirana',
]
THRESHOLD = float(os.getenv('THRESHOLD', '0.75'))
MIN_GAP = float(os.getenv('MIN_GAP', '0.20'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PORT = int(os.getenv('PORT', '5002'))
HOST = os.getenv('HOST', '0.0.0.0')

# Load data
def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        app.logger.error(f"Data file not found: {filepath}")
        return {}

SPECIES_INFO = load_json_data('data/species_info.json')
ENV_REQUIREMENTS = load_json_data('data/env_requirements.json')
PREDEFINED_LOCATIONS = load_json_data('data/predefined_locations.json')

# Load model
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    img = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fish(image):
    if model is None:
        return {'status': 'error', 'message': 'Model not loaded'}
    
    img_array = preprocess_image(image)
    pred = model.predict(img_array, verbose=0)[0]
    
    sorted_indices = np.argsort(pred)[::-1]
    top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
    top1_conf, top2_conf = float(pred[top1_idx]), float(pred[top2_idx])
    
    predictions = [
        {'species': CLASS_NAMES[idx], 'confidence': float(pred[idx])}
        for idx in sorted_indices
    ]
    
    if top1_conf < THRESHOLD or (top1_conf - top2_conf) < MIN_GAP:
        return {
            'status': 'unknown',
            'message': 'Species could not be identified with sufficient confidence',
            'predicted_species': None,
            'confidence': top1_conf,
            'all_predictions': predictions[:3]
        }
    else:
        return {
            'status': 'success',
            'predicted_species': CLASS_NAMES[top1_idx],
            'confidence': top1_conf,
            'all_predictions': predictions[:3]
        }

def calculate_survival_probability(species, measurements):
    if species not in ENV_REQUIREMENTS:
        return {'probability': 0.0, 'issues': ['Unknown species requirements']}
    
    req = ENV_REQUIREMENTS[species]
    score = 100.0
    issues = []
    
    # pH
    if 'pH' in measurements:
        ph = measurements['pH']
        r = req['pH']
        if ph < r['min'] or ph > r['max']:
            score *= 0.3
            issues.append(f"pH {ph} outside range {r['min']}-{r['max']}")
        elif not (r['optimal'][0] <= ph <= r['optimal'][1]):
            score *= 0.75
            issues.append(f"pH {ph} suboptimal (optimal: {r['optimal']})")
    
    # Temperature
    if 'temperature_C' in measurements:
        temp = measurements['temperature_C']
        r = req['temperature_C']
        if temp < r['min'] or temp > r['max']:
            score *= 0.25
            issues.append(f"Temp {temp}°C outside {r['min']}-{r['max']}°C")
        elif not (r['optimal'][0] <= temp <= r['optimal'][1]):
            score *= 0.7
            issues.append(f"Temp {temp}°C suboptimal")
    
    # Dissolved Oxygen (most critical)
    if 'dissolved_oxygen_mgL' in measurements:
        do = measurements['dissolved_oxygen_mgL']
        r = req['dissolved_oxygen_mgL']
        if do < r['min']:
            score *= max(0.1, do / r['min'])
            issues.append(f"DO {do} mg/L CRITICAL (< {r['min']})")
        elif do < r['optimal']:
            score *= 0.8
            issues.append(f"DO {do} mg/L below optimal {r['optimal']}")
    
    return {
        'probability': round(max(0.0, min(100.0, score)), 1),
        'issues': issues[:3],  # Top 3 issues
        'status': 'good' if score > 80 else 'warning' if score > 50 else 'critical'
    }

# === ROUTES ===

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '🐟 Fish Recognition & Survival API v2.0',
        'endpoints': {
            'POST /predict': 'Image → Species',
            'GET /species': 'List species',
            'GET /species/info/<name>': 'Species details',
            'GET /locations': 'Predefined locations (Sri Lanka)',
            'POST /assess/survival': 'Custom measurements → Survival %',
            'POST /assess/survival/location': 'Location → Survival %',
            'GET /health': 'Status check'
        },
        'species_count': len(CLASS_NAMES),
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'species_count': len(CLASS_NAMES),
        'data_files': {
            'species_info': bool(SPECIES_INFO),
            'env_requirements': bool(ENV_REQUIREMENTS),
            'locations': bool(PREDEFINED_LOCATIONS)
        }
    })

@app.route('/species', methods=['GET'])
def get_species():
    return jsonify({
        'species': CLASS_NAMES,
        'count': len(CLASS_NAMES),
        'endangered_species': [s for s in CLASS_NAMES if 'micronema' in s or 'memorialis' in s]
    })

@app.route('/species/info/<species_name>', methods=['GET'])
def get_species_info(species_name):
    species_name = species_name.lower().replace('-', '_')
    if species_name not in SPECIES_INFO:
        return jsonify({
            'error': 'Species not found',
            'available': list(SPECIES_INFO.keys())
        }), 404
    return jsonify({
        'status': 'success',
        'species': species_name,
        'info': SPECIES_INFO[species_name]
    })

@app.route('/locations', methods=['GET'])
def get_locations():
    return jsonify({
        'locations': PREDEFINED_LOCATIONS,
        'count': len(PREDEFINED_LOCATIONS),
        'recommended_for_srilanka': ['Kandy', 'Knuckles', 'Sinharaja']
    })

# Prediction Endpoints (Your original code enhanced)
@app.route('/predict', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(file.stream)
        result = predict_fish(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        result = predict_fish(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/url', methods=['POST'])
def predict_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL'}), 400
    
    try:
        response = requests.get(data['url'], timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        result = predict_fish(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: Threat Assessment Endpoints (Your Flowchart!)
@app.route('/assess/survival', methods=['POST'])
def assess_survival():
    data = request.get_json()
    if not data or 'species' not in data or 'measurements' not in data:
        return jsonify({'error': 'Need species & measurements'}), 400
    
    species = data['species'].lower().replace('-', '_')
    measurements = data['measurements']
    
    result = calculate_survival_probability(species, measurements)
    
    return jsonify({
        'status': 'success',
        'species': species,
        'survival_probability_percent': result['probability'],
        'health_status': result['status'],
        'measurements_used': measurements,
        'critical_issues': result['issues'],
        'recommendation': 'GOOD' if result['probability'] > 80 else 'MONITOR' if result['probability'] > 50 else 'HIGH RISK - Relocate fish'
    })

@app.route('/assess/survival/location', methods=['POST'])
def assess_location():
    data = request.get_json()
    if not data or 'species' not in data or 'location' not in data:
        return jsonify({'error': 'Need species & location'}), 400
    
    species = data['species'].lower().replace('-', '_')
    location = data['location']
    
    if location not in PREDEFINED_LOCATIONS:
        return jsonify({'error': f'Location not found. Available: {list(PREDEFINED_LOCATIONS.keys())}'}), 404
    
    measurements = PREDEFINED_LOCATIONS[location].copy()
    
    # Allow overrides
    if 'overrides' in data:
        measurements.update(data['overrides'])
    
    result = calculate_survival_probability(species, measurements)
    
    return jsonify({
        'status': 'success',
        'species': species,
        'location': location,
        'measurements_used': measurements,
        'survival_probability_percent': result['probability'],
        'health_status': result['status'],
        'critical_issues': result['issues'],
        'location_notes': PREDEFINED_LOCATIONS[location].get('notes', '')
    })

if __name__ == '__main__':
    print(f"🚀 Starting Fish API on {HOST}:{PORT}")
    print(f"📊 Species: {len(CLASS_NAMES)} | Model: {'✅' if model else '❌'}")
    app.run(host=HOST, port=PORT, debug=True)