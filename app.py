from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import torch
import cv2
import numpy as np
from rembg import remove
import cairosvg  # For SVG conversion

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, "skin-compact-x1.pth")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define a lightweight model for upscaling (Placeholder for actual architecture)
class CompactNet(torch.nn.Module):
    def __init__(self):
        super(CompactNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

# Load the upscaling model
def load_upscale_model(model_path):
    model = CompactNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Initialize the upscaling model
upscale_model = load_upscale_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

# ✅ **Updated File Converter**
@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['file']
    target_format = request.form['format'].lower()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    output_filename = f"converted.{target_format}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    try:
        if target_format in ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'ico', 'gif']:
            img = Image.open(file_path)
            img.save(output_path, format=target_format.upper())
        elif target_format == 'pdf':
            img = Image.open(file_path).convert("RGB")
            img.save(output_path, "PDF")
        elif target_format == 'svg':
            png_output = os.path.join(UPLOAD_FOLDER, "converted.png")
            cairosvg.svg2png(url=file_path, write_to=png_output)
            return send_file(png_output, as_attachment=True)
        else:
            return "Unsupported format. Only PNG, JPG, BMP, PDF, and SVG are supported.", 400
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Error during conversion: {e}", 500

# ✅ **Background Remover**
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

# ✅ **Upscaling with `skin-compact-x1.pth`**
@app.route('/upscale', methods=['POST'])
def upscale():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Failed to load the image."

        input_tensor = preprocess_image(img)
        with torch.no_grad():
            output_tensor = upscale_model(input_tensor)
        output_image = postprocess_tensor(output_tensor)

        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, output_image)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"An error occurred during upscaling: {e}"

# Helper Functions for Image Processing
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize
    return img

def postprocess_tensor(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))  # Use dynamic port for Render
    app.run(host='0.0.0.0', port=port, debug=True)
