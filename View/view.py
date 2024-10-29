import os
from flask import Flask, render_template, request, jsonify,session,url_for,redirect
import sys
#import for chart and report_log
import io  # Thêm 
import base64
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không GUI
import docx
from matplotlib import pyplot as plt
import threading
import time
from queue import Queue
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from Controller.controller import ImageController, FileController , AuthController

app = Flask(__name__)
app.secret_key="thisismysecretkeyforthisapp"
# Initialize controllers
image_controller = ImageController()
file_controller = FileController()
auth_controller=AuthController()
# Training queue and thread
training_queue = Queue()
training_thread = None

import os
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def train_model():
    while True:
        # Wait for a training task
        task = training_queue.get()
        if task is None:
            break
        
        print("Starting model training...")
        # Ở đây, bạn sẽ chạy mã training từ Jupyter notebook của bạn
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        notebook_path = os.path.join(project_root, 'idvn-ocr.ipynb')
        # Đọc nội dung notebook
        with open(notebook_path, 'r', encoding='utf-8') as file:
            notebook_content = nbformat.read(file, as_version=4)

        # Sử dụng ExecutePreprocessor để thực thi notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(notebook_content, {'metadata': {'path': project_root}})

        time.sleep(10)  # Giả lập thời gian training
        print("Model training completed")
        
        training_queue.task_done()

def start_training_thread():
    global training_thread
    if training_thread is None or not training_thread.is_alive():
        training_thread = threading.Thread(target=train_model)
        training_thread.start()

@app.route('/')
def index():
    # Kiểm tra nếu người dùng đã đăng nhập
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html')
    else:
        # Chuyển hướng đến trang đăng nhập nếu chưa đăng nhập
        return redirect(url_for('login')) 
    
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
    training_queue.put(True)
    
    # Đảm bảo thread training đang chạy
    start_training_thread()
    
    return result

# Hàm đọc file Word
def read_word_file(file_path):
    doc = docx.Document(file_path)
    all_tables_data = []  # Danh sách chứa tất cả các bảng
    
    # Đọc từng bảng trong file Word
    for table in doc.tables:
        table_data = []  # Danh sách chứa dữ liệu cho bảng hiện tại
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        all_tables_data.append(table_data)  # Thêm bảng hiện tại vào danh sách
    return all_tables_data  # Trả về danh sách các bảng

@app.route('/report_log')
def report_log():
    # Đọc dữ liệu từ file Word
    file_path = "Logs.docx"  # Thay bằng đường dẫn thật đến file của bạn
    if os.path.exists(file_path):
        word_data = read_word_file(file_path)

        # Render trang HTML với dữ liệu từ file Word
        return render_template('report_log.html', tables=word_data)
    else:
        return "Log file not found.", 404

@app.route('/chart')
def view_chart():
    # Danh sách samples
    samples = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8']
    
    # Dữ liệu cho từng sample
    simple_similarity = [82.72, 98.03, 75.00, 85.50, 90.00, 78.00, 95.00, 88.50]
    levenshtein_similarity = [94.19, 98.35, 90.00, 92.50, 95.00, 80.00, 97.00, 89.00]

    # Vẽ biểu đồ đường
    plt.figure(figsize=(12, 6))  # Thay đổi kích thước nếu cần
    
    # Vẽ đường cho Average Simple Similarity
    plt.plot(samples, simple_similarity, marker='o', label='Average Simple Similarity', color='blue')
    # Vẽ đường cho Average Levenshtein Similarity
    plt.plot(samples, levenshtein_similarity, marker='o', label='Average Levenshtein Similarity', color='red')
    
    plt.xlabel('Samples')
    plt.ylabel('Similarity (%)')
    plt.title('Comparison of Similarity Metrics')
    plt.legend()
    plt.grid(True)
    
    # Lưu biểu đồ vào buffer và chuyển đổi thành base64
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_png = buf.getvalue()
    buf.close()
    graph_url = base64.b64encode(image_png).decode('utf-8')

    return render_template('chart.html', graph_url=graph_url)

@app.route('/register', methods=["POST"])
def register():
    # Lấy dữ liệu JSON từ yêu cầu POST
    data = request.get_json()
    # Gọi hàm register trong auth_controller với dữ liệu người dùng
    result = auth_controller.register(data)

    # Trả về kết quả của hàm register dưới dạng JSON
    return jsonify(result)

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
        else:
            username = request.form.get("username")
            password = request.form.get("password")
        
        # Gọi hàm login trong auth_controller với dữ liệu người dùng
        result = auth_controller.login(username, password)

        if result == "Login successful.":
            # Lưu thông tin đăng nhập vào session
            session['logged_in'] = True
            session['username'] = username
            return jsonify(result)
        else:
            return jsonify({"error": result}), 401
    else:
        return render_template("login.html")


if __name__ == '__main__':
    start_training_thread()
    app.run(host="0.0.0.0", port=5001, debug=False)