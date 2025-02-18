from flask import Flask, request, render_template, send_file
from PIL import Image
from rembg import remove
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Mapping for format conversion
FORMAT_MAP = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "bmp": "BMP",
    "pdf": "PDF",
    "svg": "SVG",
    "ico": "ICO",
    "webp": "WEBP",
    "gif": "GIF",
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/templates/about.html')
def about():
    return render_template('about.html')


@app.route('/templates/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/convert', methods=['POST'])
def convert_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    format = request.form.get("format", "png").lower()
    
    if format not in FORMAT_MAP:
        return "Invalid format selected"
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    image = Image.open(filepath)
    converted_filepath = os.path.join(PROCESSED_FOLDER, f"converted.{format}")
    image.save(converted_filepath, FORMAT_MAP[format])
    
    return send_file(converted_filepath, as_attachment=True)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    image = Image.open(filepath)
    processed_image = remove(image)
    processed_filepath = os.path.join(PROCESSED_FOLDER, "no_bg.png")
    processed_image.save(processed_filepath)
    
    return send_file(processed_filepath, as_attachment=True)

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    factor = int(request.form.get("factor", 2))
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    image = Image.open(filepath)
    width, height = image.size
    new_size = (width * factor, height * factor)
    upscaled_image = image.resize(new_size, Image.LANCZOS)
    processed_filepath = os.path.join(PROCESSED_FOLDER, "upscaled.png")
    upscaled_image.save(processed_filepath)
    
    return send_file(processed_filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
