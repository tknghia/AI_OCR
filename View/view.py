# View/view.py
import os
import sys
from flask import Flask, render_template, request, redirect, url_for,send_file
from pymongo import MongoClient
import re
from io import BytesIO
from datetime import datetime
# Kết nối đến MongoDB
mongo_uri = "mongodb+srv://user:ocrdeeplearning@ocr.83apy.mongodb.net/"
client = MongoClient(mongo_uri)  # Thay đổi URL nếu cần
db = client['TestingMongoDB']  #Tên cơ sở dữ liệu của bạn
collection = db['TestingMongoDB']  # Tên bộ sưu tập của bạn
db1= client['CCCD']
collectionCCCD= db1['CCCD']
log_collection = db1['Logs']
# Lấy đường dẫn tuyệt đối của thư mục hiện tại (thư mục chứa file này)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn của thư mục gốc dự án (thư mục cha của thư mục hiện tại)
project_root = os.path.dirname(current_dir)

# Thêm thư mục gốc dự án vào sys.path
sys.path.append(project_root)

# Bây giờ bạn có thể import từ Controller
# from Controller.controller import convert_images
from Controller.controller import convert_images
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Controller')))
# from Controller.controller import convert_images

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

import re

def extract_info(data):
    # Trích xuất họ và tên
    name_match = re.search(r'Họ và tên / Full name:\s*(.+)', data)
    name = name_match.group(1) if name_match else None

    # Trích xuất số căn cước
    id_match = re.search(r'Số / No\.:\s*(\d+)', data)
    id_number = id_match.group(1) if id_match else None

    # Trích xuất ngày sinh
    dob_match = re.search(r'Ngày sinh / Date of birth:\s*(\d{2}/\d{2}/\d{4})', data)
    dob = dob_match.group(1) if dob_match else None

    # Trích xuất giới tính
    gender_match = re.search(r'Giới tính / Sex:\s*(\w+)', data)
    gender = gender_match.group(1) if gender_match else None

    return {
        "Họ và tên": name,
        "Số căn cước": id_number,
        "Ngày sinh": dob,
        "Giới tính": gender
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def handle_convert_images():
    # Get the list of uploaded files from the request
    files = request.files.getlist('images')

    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Lấy thời gian hiện tại
    log_entries = []  # Danh sách để lưu các log

    for file in files:
        log_entry = f"{upload_time} - Uploaded file: {file.filename}"
        log_entries.append(log_entry)  # Thêm vào danh sách log

    
    # Call the controller function with the files
    predictions = convert_images(files)

      # Đường dẫn đến thư mục log
    log_dir = os.path.join(project_root, "Logs")  # Tạo đường dẫn đến thư mục Logs

      # Đường dẫn đến file log
    log_file_path = os.path.join(log_dir, "upload_logs.txt")  # Tạo đường dẫn đến file log uploads_log.txt

    # Ghi toàn bộ log vào file
    with open(log_file_path, "a", encoding='utf-8') as log_file:  # Mở file log
        for entry in log_entries:
            log_file.write(entry + "\n")  # Ghi từng dòng log vào file

    mongo_log_entry = {"upload_time": upload_time, "predictions": predictions}
    print(f"Ghi vào MongoDB: {mongo_log_entry}")  # Thêm dòng này
    try:
        log_collection.insert_one(mongo_log_entry)  # Ghi vào MongoDB
        print("Dữ liệu đã được ghi vào MongoDB thành công!")
    except Exception as e:
        print(f"Lỗi khi ghi vào MongoDB: {e}")
    
    collection.insert_one({"predictions": predictions})
    if("CĂN CƯỚC CÔNG DÂN" in predictions):
        collectionCCCD.insert_one({"cccd":extract_info(predictions)})
    return predictions


@app.route('/download', methods=['POST'])
def download():
    data = request.json
    content = data.get('content', '')
    
    # Xử lý nội dung ở đây (ví dụ: chuyển đổi thành chữ hoa)
    processed_content = content.upper()
    
    # Tạo file trong bộ nhớ
    buffer = BytesIO()
    buffer.write(processed_content.encode('utf-8'))
    buffer.seek(0)
    
    # Gửi file để tải xuống
    return send_file(buffer,
                     as_attachment=True,
                     download_name='processed_content.doc',
                     mimetype='application/msword')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
