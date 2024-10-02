# View/view.py
import os
import sys
from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient

# Kết nối đến MongoDB
mongo_uri = "mongodb+srv://user:ocrdeeplearning@ocr.83apy.mongodb.net/"
client = MongoClient(mongo_uri)  # Thay đổi URL nếu cần
db = client['TestingMongoDB']  #Tên cơ sở dữ liệu của bạn
collection = db['TestingMongoDB']  # Tên bộ sưu tập của bạn

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def handle_convert_images():
    # Get the list of uploaded files from the request
    files = request.files.getlist('images')
    
    # Call the controller function with the files
    predictions = convert_images(files)
    
    print(predictions)
    collection.insert_one({"predictions": predictions})
    
    return predictions

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
