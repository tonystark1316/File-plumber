from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import cv2
from rembg import remove

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Error: {e}"

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
        return f"Error: {e}"

@app.route('/upscale', methods=['POST'])
def upscale():
    file = request.files['file']
    upscale_factor = int(request.form.get('factor', 2))  # Default factor is 2x
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Invalid image format."

        new_width = img.shape[1] * upscale_factor
        new_height = img.shape[0] * upscale_factor
        upscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        output_path = os.path.join(UPLOAD_FOLDER, "upscaled.png")
        cv2.imwrite(output_path, upscaled_img)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Upscaling failed: {e}"

if __name__ == '__main__':
    app.run(debug=True)
