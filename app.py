from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import torch
import cv2
import numpy as np
from rembg import remove

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, "smbss-2x.pth")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define CompactNet (Placeholder for actual architecture)
class CompactNet(torch.nn.Module):
    def __init__(self):
        super(CompactNet, self).__init__()
        # Define layers (Needs actual implementation)
        pass

    def forward(self, x):
        # Define forward pass (Needs actual implementation)
        return x

# Load the smbss-2x model
def load_smbss_model(model_path):
    model = CompactNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Initialize the smbss-2x model
smbss_model = load_smbss_model(MODEL_PATH)

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
    if target_format in ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'ico', 'gif']:
        img = Image.open(file_path)
        img.save(output_path, format=target_format.upper())
    elif target_format == 'pdf':
        img = Image.open(file_path).convert("RGB")
        img.save(output_path, "PDF")
    else:
        return "Unsupported format. SVG conversion is not implemented."
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
            output_tensor = smbss_model(input_tensor)
        
        output_image = postprocess_tensor(output_tensor)
        
        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, output_image)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred during upscaling: {e}"

def preprocess_image(img):
    # Implement required preprocessing (normalize, reshape, etc.)
    return torch.from_numpy(img).float().unsqueeze(0)

def postprocess_tensor(tensor):
    # Convert tensor back to image format
    return tensor.squeeze(0).numpy().astype(np.uint8)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
