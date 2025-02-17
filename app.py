from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import torch
import cv2
import numpy as np
from rembg import remove

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, "skin-compact-x1.pth")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define CompactNet (Placeholder for actual architecture)
class CompactNet(torch.nn.Module):
    def __init__(self):
        super(CompactNet, self).__init__()
        # Define layers (Placeholder, update with actual model architecture)
        pass

    def forward(self, x):
        return x  # Update with the real forward pass

# Load the upscaler model
def load_model(model_path):
    model = CompactNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Initialize Model (Load from models folder)
upscaler_model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['file']
    target_format = request.form['format']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    output_path = os.path.join(UPLOAD_FOLDER, f"converted.{target_format}")
    try:
        img = Image.open(file_path)
        img.save(output_path, format=target_format.upper())
    except Exception as e:
        return f"Error during conversion: {e}"
    
    return send_file(output_path, as_attachment=True)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        with open(file_path, "rb") as input_file:
            output = remove(input_file.read())
        output_path = os.path.join(UPLOAD_FOLDER, f"no_bg_{file.filename}")
        with open(output_path, "wb") as output_file:
            output_file.write(output)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/upscale', methods=['POST'])
def upscale():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Failed to load the image."

        # Preprocess image for model
        input_tensor = preprocess_image(img)
        
        with torch.no_grad():
            output_tensor = upscaler_model(input_tensor)
        
        output_image = postprocess_tensor(output_tensor)
        
        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, output_image)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred during upscaling: {e}"

def preprocess_image(img):
    return torch.from_numpy(img).float().unsqueeze(0)  # Update preprocessing based on model needs

def postprocess_tensor(tensor):
    return tensor.squeeze(0).numpy().astype(np.uint8)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for Render
    app.run(host='0.0.0.0', port=port, debug=True)
