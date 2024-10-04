import os
from flask import Flask, render_template, request
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from Controller.controller import ImageController, FileController

app = Flask(__name__)

# Initialize controllers
image_controller = ImageController()
file_controller = FileController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def handle_convert_images():
    files = request.files.getlist('images')
    return image_controller.process_images(files)

@app.route('/download', methods=['POST'])
def download():
    content = request.json.get('content', '')
    return file_controller.download_content(content)

@app.route('/save', methods=['POST'])
def save():
    labels = request.json.get('labels', [])
    return file_controller.save_labels(labels)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)