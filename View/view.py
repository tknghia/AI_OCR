import os
from flask import Flask, render_template, request
import sys
# import threading
import time
from queue import Queue
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from Controller.controller import ImageController, FileController

app = Flask(__name__)

# Initialize controllers
image_controller = ImageController()
file_controller = FileController()

# Training queue and thread
# training_queue = Queue()
# training_thread = None

import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# def train_model():
#     while True:
#         # Wait for a training task
#         task = training_queue.get()
#         if task is None:
#             break
        
#         print("Starting model training...")
#         # Ở đây, bạn sẽ chạy mã training từ Jupyter notebook của bạn
        
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(current_dir)
#         notebook_path = os.path.join(project_root, 'idvn-ocr.ipynb')
#         # Đọc nội dung notebook
#         with open(notebook_path, 'r', encoding='utf-8') as file:
#             notebook_content = nbformat.read(file, as_version=4)

#         # Sử dụng ExecutePreprocessor để thực thi notebook
#         ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
#         ep.preprocess(notebook_content, {'metadata': {'path': project_root}})

#         time.sleep(10)  # Giả lập thời gian training
#         print("Model training completed")
        
#         training_queue.task_done()

# def start_training_thread():
#     global training_thread
#     if training_thread is None or not training_thread.is_alive():
#         training_thread = threading.Thread(target=train_model)
#         training_thread.start()

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
    result_id = request.json.get('result_id', '')
    # Lưu labels
    result = file_controller.save_labels(labels, result_id)
    
    # Thêm task training vào queue
    # training_queue.put(True)
    
    # Đảm bảo thread training đang chạy
    # start_training_thread()
    
    return result

if __name__ == '__main__':
    # start_training_thread()
    app.run(host="0.0.0.0", port=5001, debug=False)