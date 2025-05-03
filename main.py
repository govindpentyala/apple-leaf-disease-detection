from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__, static_folder='static', template_folder='templates')

# ========== MODEL SETUP ==========
# Fruit detection model (YOLO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, 'static', 'models', 'best.pt')
fruit_model = YOLO(FRUIT_MODEL_PATH)
fruit_class_names = fruit_model.names

# Leaf detection model (ResNet)
resnet_model = models.resnet18()
num_inftr = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_inftr, 4)
resnet_model.load_state_dict(torch.load(os.path.join('static', 'models', 'fix_resnet18.pth'), map_location=torch.device('cpu')))
resnet_model.eval()
leaf_class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

# ========== SHARED ROUTES ==========
@app.route('/')
@app.route('/home')
def home():
    return render_template('shared/home.html')

@app.route('/apple')
def apple():
    return render_template('index.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf_index.html')

@app.route('/videos')
def video():
    return render_template('shared/video.html', template_folder='templates/shared')

@app.route('/contact-us')
def contact():
    return render_template('shared/contact.html', template_folder='templates/shared')

@app.route('/join-us')
def join():
    return render_template('shared/join.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# ========== FRUIT PREDICTION (from app.py) ==========
@app.route('/predict-fruit', methods=['POST'])
def predict_fruit():
    app.logger.info('Predict route called for apple fruit detection')
    app.logger.info(f'Files in request: {request.files}')
    
    # Check for 'image' field first (used in index.html)
    if 'image' in request.files:
        file = request.files['image']
    # Check for 'file' field as fallback (used in leaf_index.html)
    elif 'file' in request.files:
        file = request.files['file']
    else:
        app.logger.error('No file part in request')
        return "<span class='text-danger'>No file uploaded</span>", 400
    
    app.logger.info(f'File name: {file.filename}')
    if file.filename == '':
        app.logger.error('No selected file')
        return "<span class='text-danger'>No selected file</span>", 400
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        
        results = fruit_model(img)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append((fruit_class_names[cls], conf))
        
        if detections:
            best_match = max(detections, key=lambda x: x[1])
            predicted_class, confidence = best_match
            
            app.logger.info(f'Raw model prediction: {predicted_class} with confidence {confidence}')
            
            # Map the model's output classes to the expected frontend disease names
            disease_mapping = {
                'BLOTCH': 'Apple Blotch',
                'HEALTHY': 'Healthy',
                'ROT': 'Black Rot',
                'SCAB': 'Apple Scab'
            }
            
            # Convert to title case first (in case model returns lowercase)
            normalized_class = predicted_class.upper()
            
            # Map to the frontend expected name
            if normalized_class in disease_mapping:
                frontend_class = disease_mapping[normalized_class]
            else:
                # Fallback if the mapping is missing
                frontend_class = predicted_class
                
            app.logger.info(f'Mapped to frontend class: {frontend_class}')
            
            # Format for consistent handling in JavaScript
            if 'HEALTHY' in normalized_class:
                return f"<span class='text-success'><i class='fas fa-check-circle me-2'></i>Detected: <b>{frontend_class}</b> with {confidence*100:.2f}% confidence</span>"
            else:
                return f"<span class='text-danger'><i class='fas fa-virus me-2'></i>Detected: <b>{frontend_class}</b> with {confidence*100:.2f}% confidence</span>"
        else:
            return "<span class='text-danger'>No apple disease detected. Try another image.</span>"
    except Exception as e:
        app.logger.error(f'Error in prediction: {str(e)}')
        return f"<span class='text-danger'>Error: {str(e)}</span>", 500

# ========== LEAF PREDICTION (from leaf.py) ==========
@app.route('/predict-leaf', methods=['POST'])
def predict_leaf():
    app.logger.info('Predict route called for leaf detection')
    app.logger.info(f'Files in request: {request.files}')
    
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return "<span class='text-danger'>No file part in the request</span>"
    
    file = request.files['file']
    app.logger.info(f'File name: {file.filename}')
    
    if file.filename == '':
        app.logger.error('No selected file')
        return "<span class='text-danger'>No selected file</span>"
    
    try:
        img_bytes = file.read()
        if len(img_bytes) == 0:
            app.logger.error('Empty file uploaded')
            return "<span class='text-danger'>Empty file uploaded</span>"
            
        result, confidence = get_leaf_prediction(img_bytes)
        app.logger.info(f'Prediction result: {result}, confidence: {confidence}')
        
        if result == "Healthy":
            return f"<span class='text-success'><i class='fas fa-check-circle me-2'></i>Detected: <b>{result}</b> with {confidence:.2f}% confidence</span>"
        else:
            return f"<span class='text-danger'><i class='fas fa-virus me-2'></i>Detected: <b>{result}</b> with {confidence:.2f}% confidence</span>"
    except Exception as e:
        app.logger.error(f'Error in prediction: {str(e)}')
        return f"<span class='text-danger'>Error: {str(e)}</span>", 500

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return my_transforms(image).unsqueeze(0)

def get_leaf_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        outputs = resnet_model.forward(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probabilities, 1)
    return leaf_class_names[prediction], confidence.item() * 100

if __name__ == '__main__':
    app.run(debug=False, port=3000)