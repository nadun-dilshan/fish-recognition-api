from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import json
import cv2
from tensorflow.keras import backend as K
import matplotlib.cm as cm
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

def get_last_conv_layer_name(model):
    """Find the last conv layer dynamically (works for most CNNs)"""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("Could not find a Conv2D layer in the model")


LAST_CONV_LAYER_NAME = None  # will be set on first use

def get_gradcam_heatmap(img_array, model, class_idx, last_conv_layer_name):
    """Generate raw Grad-CAM heatmap (0–1)"""
    global LAST_CONV_LAYER_NAME
    if LAST_CONV_LAYER_NAME is None:
        LAST_CONV_LAYER_NAME = last_conv_layer_name or get_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Weight the conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.expand_dims(pooled_grads, axis=0) * conv_outputs, axis=-1)

    # ReLU + normalize
    heatmap = tf.nn.relu(heatmap)
    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


def heatmap_to_colored_image(heatmap, original_img, alpha=0.55, colormap=cv2.COLORMAP_TURBO):
    orig_np = np.array(original_img)
    h, w = orig_np.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), cv2.INTER_CUBIC)
    heatmap_norm = np.clip(heatmap_resized / np.max(heatmap_resized + 1e-8), 0, 1)

    # ── Stronger foreground mask ───────────────────────────────────────
    lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB)
    a_channel = lab[:,:,1].astype(np.float32)   # red-green
    b_channel = lab[:,:,2].astype(np.float32)   # yellow-blue

    # Many tropical fish have strong a/b channel contrast
    color_contrast = np.sqrt(a_channel**2 + b_channel**2)
    color_contrast = cv2.GaussianBlur(color_contrast, (0,0), 4)
    color_contrast = (color_contrast - color_contrast.min()) / (color_contrast.max() - color_contrast.min() + 1e-6)

    luminosity = lab[:,:,0].astype(np.float32) / 255.0
    mask = 0.65 * color_contrast + 0.35 * luminosity
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = np.clip(mask * 1.6, 0, 1)           # aggressive boost

    # Final sharpening of transition
    mask = np.power(mask, 1.35)

    # ── Compose ─────────────────────────────────────────────────────────
    colored = (255 * heatmap_norm[..., np.newaxis] * mask[..., np.newaxis]).astype(np.uint8)
    colored_mapped = cv2.applyColorMap(colored, colormap)
    colored_mapped = cv2.cvtColor(colored_mapped, cv2.COLOR_BGR2RGB)

    # Where mask is low → original image, where high → colored heatmap
    output = orig_np * (1 - mask[..., np.newaxis]) + colored_mapped * mask[..., np.newaxis]
    output = np.clip(output, 0, 255).astype(np.uint8)

    return Image.fromarray(output)

def generate_gradcam_variants(image: Image.Image, img_array, model, top_class_idx):
    """Return original + improved heatmap-focused variants"""
    heatmap = get_gradcam_heatmap(img_array, model, top_class_idx, None)

    # ── Main improved overlay ───────────────────────────────────────
    heatmap_overlay_img = heatmap_to_colored_image(
        heatmap, 
        image, 
        alpha=0.45, 
        colormap=cv2.COLORMAP_JET   # ← you can also try PLASMA, INFERNO, MAGMA, TURBO
    )

    # ── Optional: high-contrast version (sometimes better for fish) ──
    high_contrast_overlay = heatmap_to_colored_image(
        heatmap, 
        image, 
        alpha=0.6, 
        colormap=cv2.COLORMAP_TURBO   # often gives nicer separation
    )

    # ── Keep simple outline variant (only one) ───────────────────────
    orig_np = np.array(image)
    gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_rgb[edges > 0] = [80, 255, 120]          # softer green
    outlined = cv2.addWeighted(orig_np, 0.75, edges_rgb, 0.45, 0)
    outlined_img = Image.fromarray(outlined)

    # ── Combined variant (heatmap + subtle outline) ──────────────────
    combined_np = np.array(heatmap_overlay_img)
    edges_combined = cv2.Canny(cv2.cvtColor(combined_np, cv2.COLOR_RGB2GRAY), 60, 180)
    edges_rgb = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2RGB)
    edges_rgb[edges_combined > 0] = [100, 255, 140]
    combined = cv2.addWeighted(combined_np, 0.92, edges_rgb, 0.35, 0)
    combined_img = Image.fromarray(combined)

    def pil_to_base64(pil_img):
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "original":          pil_to_base64(image),
        "heatmap":           pil_to_base64(heatmap_overlay_img),     # ← main improved one
        "heatmap_contrast":  pil_to_base64(high_contrast_overlay),   # optional extra
        "outline":           pil_to_base64(outlined_img),
        "combined":          pil_to_base64(combined_img),
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
@app.route('/predict', methods=['GET', 'POST'])  # allow GET just in case someone tests
def predict_file():
    if request.method == 'GET':
        return jsonify({"message": "Use POST to send image"}), 405

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(file.stream)
        img_array = preprocess_image(image)   # already have this function
        
        pred_result = predict_fish(image)      # your original function
        
        response = pred_result  # start with original dict
        
        # ── Grad-CAM support ────────────────────────────────
        if request.args.get('gradcam', 'false').lower() in ('true', '1', 'yes'):
            if model is None:
                response['gradcam_error'] = 'Model not loaded'
            elif pred_result.get('status') != 'success':
                response['gradcam_error'] = 'No confident prediction → no Grad-CAM'
            else:
                top_species = pred_result['predicted_species']
                top_idx = CLASS_NAMES.index(top_species)
                
                try:
                    variants = generate_gradcam_variants(image, img_array, model, top_idx)
                    response['gradcam'] = variants
                    response['gradcam_layer'] = LAST_CONV_LAYER_NAME  # optional
                    response['gradcam_class'] = top_species           # optional, helpful for UI
                except Exception as e:
                    response['gradcam_error'] = f"Grad-CAM failed: {str(e)}"
        
        return jsonify(response)
    
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