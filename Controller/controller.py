import os
import sys
import re
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO
from pymongo import MongoClient
from flask import send_file, jsonify
from bson.objectid import ObjectId
from docx import Document
from docx.shared import Inches
# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import vietocr_module and segmentsPaddle from the model
from Model.module import vietocr_module as vietocr_module
from Model.module import crop_text_line_paddle as segmentsPaddle


class MongoController:
    def __init__(self):
        # MongoDB setup
        MONGO_URI = "mongodb+srv://user:ocrdeeplearning@ocr.83apy.mongodb.net/"
        self.client = MongoClient(MONGO_URI)
        self.db = self.client['TestingMongoDB']
        self.collection = self.db['TestingMongoDB']
        self.db_cccd = self.client['CCCD']
        self.collection_cccd = self.db_cccd['CCCD']
        self.log_collection = self.db_cccd['Logs']
    
    def select_prediction_by_id(self, id):
        try:
            # Convert the string ID to an ObjectId
            object_id = ObjectId(id)
            
            # Find the document in the CCCD collection using the _id
            result = self.collection.find_one({"_id": object_id})
            
            if result:
                return result
            else:
                return f"No prediction found with id: {id}"

        except Exception as e:
            # Log the error and return a meaningful error message
            print(f"Error while retrieving prediction with id {id}: {e}")
            return f"Error while retrieving prediction: {e}"

    def log_predictions(self, predictions, upload_time):
        try:
            # Log predictions to MongoDB
            mongo_log_entry = {"upload_time": upload_time, "predictions": predictions}
            self.log_collection.insert_one(mongo_log_entry)
            result_prediction=self.collection.insert_one({"predictions": predictions})

            # If CCCD is detected, log additional info
            if "CĂN CƯỚC CÔNG DÂN" in predictions:
                extracted_info = ImageController.extract_info(predictions)
                self.collection_cccd.insert_one({"cccd": extracted_info})
            inserted_document = self.collection.find_one({"_id": result_prediction.inserted_id})

            return str(inserted_document["_id"])

        except Exception as e:
            print(f"Error writing to MongoDB: {e}")


class ImageController:
    mongo_controller = MongoController()  # Initialize MongoController

    @staticmethod
    def convert_images(files):
        all_predictions = []

        # Loop through all uploaded files
        for file in files:
            img = Image.open(file)  # Open the image file directly

            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]

            if height > 55:
                # Perform segmentation and OCR prediction
                arr = segmentsPaddle.extract_image_segments(cv_image)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(os.path.dirname(current_dir),'Model','dataset','output_images')
                os.makedirs(output_dir, exist_ok=True)
                beginNumber = len([f for f in os.listdir(output_dir) if not f.endswith('.txt')])
                
                
                for i, img in enumerate(arr):
                    # Create filename
                    filename = f"{beginNumber + i}.png"
                    filepath = os.path.join(output_dir, filename)
                    pil_img = Image.fromarray(np.uint8(img))
                    pil_img.save(filepath)

                print(f"Saved {len(arr)} images to folder {output_dir}")

                for img_segment in arr:
                    np_image = np.asarray(img_segment)
                    image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    str_pred = vietocr_module.vietOCR_prediction(image_pil)
                    all_predictions.append(str_pred)
            else:
                # Use cv_image directly if height <= 55px
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                str_pred = vietocr_module.vietOCR_prediction(image_pil)
                all_predictions.append(str_pred)

        return '\n'.join(all_predictions)

    def process_images(self, files):
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = [f"{upload_time} - Uploaded file: {file.filename}" for file in files]

        predictions = self.convert_images(files)

        # Log to file
        log_file_path = os.path.join(project_root, "Logs", "upload_logs.txt")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write("\n".join(log_entries) + "\n")

        # Use MongoController to log predictions
        prediction_id=self.mongo_controller.log_predictions(predictions, upload_time)
        print(prediction_id)
        return  jsonify({
        'predictions': predictions,
        'prediction_id': prediction_id
    })

    @staticmethod
    def extract_info(data):
        patterns = {
            "Họ và tên": r'Họ và tên / Full name:\s*(.+)',
            "Số căn cước": r'Số / No\.:\s*(\d+)',
            "Ngày sinh": r'Ngày sinh / Date of birth:\s*(\d{2}/\d{2}/\d{4})',
            "Giới tính": r'Giới tính / Sex:\s*(\w+)'
        }
        return {key: re.search(pattern, data).group(1) if re.search(pattern, data) else None 
                for key, pattern in patterns.items()}



import difflib
from io import BytesIO
from flask import jsonify, send_file
import os

class FileController:
    def __init__(self):
        self.mongo_controller = MongoController()

    def save_comparison_to_word(self, comparison_results, avg_simple_similarity, avg_levenshtein_similarity, file_path):
    # Mở tệp DOCX hiện có hoặc tạo mới nếu không tồn tại
        if os.path.exists(file_path):
            doc = Document(file_path)  # Mở tệp nếu đã tồn tại
        else:
            doc = Document()  # Tạo tài liệu mới nếu không có

        # Đếm số mẫu hiện tại dựa trên số tiêu đề cấp 1
        sample_count = len([p for p in doc.paragraphs if p.style.name == 'Heading 1'])

        # Tạo tiêu đề cấp 1 cho mẫu mới
        doc.add_heading(f'Samples - {sample_count + 1}', level=1)

        # Thêm thời gian tải lên với tiêu đề cấp 4
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc.add_heading(f'Upload Time: {upload_time}', level=4)  # Thêm thời gian tải lên với tiêu đề cấp 4

        # Tạo bảng với 4 cột
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'  # Đặt kiểu cho bảng

        # Thêm tiêu đề cho các cột
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Prediction'
        hdr_cells[1].text = 'Correct line'
        hdr_cells[2].text = 'Simple Similarity'
        hdr_cells[3].text = 'Levenshtein Similarity'

        # Định dạng tiêu đề cột
        for cell in hdr_cells:
            cell.bold = True  # Bôi đậm chữ
            cell.paragraphs[0].alignment = 1  # Canh giữa

        # Thêm dữ liệu vào bảng
        for result in comparison_results:
            row_cells = table.add_row().cells
            row_cells[0].text = str(result['wrong'])  # Chuyển đổi thành chuỗi nếu cần
            row_cells[1].text = str(result['correct'])
            row_cells[2].text = f"{result['simple_similarity']:.2%}"
            row_cells[3].text = f"{result['levenshtein_similarity']:.2%}"

            # Định dạng các ô dữ liệu
            for cell in row_cells:
                cell.paragraphs[0].alignment = 1  # Canh giữa

        # Thêm các chỉ số trung bình
        doc.add_paragraph(f"Average Simple Similarity: {avg_simple_similarity:.2%}")
        doc.add_paragraph(f"Average Levenshtein Similarity: {avg_levenshtein_similarity:.2%}")

        # Lưu tệp vào đường dẫn đã chỉ định
        doc.save(file_path)

        return file_path

    @staticmethod
    def levenshtein_similarity(s1, s2):
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    @staticmethod
    def simple_string_similarity(s1, s2):
        return sum(c1 == c2 for c1, c2 in zip(s1, s2)) / max(len(s1), len(s2))

    @classmethod
    def compare_labels(cls, wrong_labels, labels):
        results = []
        for wrong, correct in zip(wrong_labels, labels):
            simple_similarity = cls.simple_string_similarity(wrong, correct)
            levenshtein_similarity_value = cls.levenshtein_similarity(wrong, correct)
            results.append({
                'wrong': wrong,
                'correct': correct,
                'simple_similarity': simple_similarity,
                'levenshtein_similarity': levenshtein_similarity_value
            })
        return results

    @staticmethod
    def download_content(content):
        processed_content = content.upper()
        buffer = BytesIO()
        buffer.write(processed_content.encode('utf-8'))
        buffer.seek(0)

        return send_file(buffer,
                         as_attachment=True,
                         download_name='processed_content.doc',
                         mimetype='application/msword')
    
    def save_labels(self, labels, result_id):
        log_file_path = os.path.join(project_root, "Model","dataset","output_images", "train.txt")
        test_file_path = os.path.join(project_root, "Model","dataset","output_images", "test.txt")

        # Kiểm tra xem thư mục của file có tồn tại không, nếu không thì tạo
        
        # Kiểm tra nếu file tồn tại
        if os.path.exists(log_file_path):
            # Mở file ở chế độ đọc nếu file tồn tại
            with open(log_file_path, 'r', encoding='utf-8') as file:
                existing_lines_count = len(file.readlines())
        else:
            # Nếu file không tồn tại, đặt số dòng ban đầu là 0 hoặc thực hiện hành động khác
            existing_lines_count = 0

        new_entries = [f"{i}.png\t{label}" for i, label in enumerate(labels, start=existing_lines_count)]
        
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write("\n".join(new_entries) + "\n")
        with open(test_file_path, "a", encoding='utf-8') as test_file:
            test_file.write("\n".join(new_entries) + "\n")
        
        old_result = self.mongo_controller.select_prediction_by_id(result_id)
        wrong_labels = old_result['predictions'].split("\n")
        comparison_results = self.compare_labels(wrong_labels, labels)


        # Tính trung bình phần trăm đúng
        avg_simple_similarity = sum(r['simple_similarity'] for r in comparison_results) / len(comparison_results)
        avg_levenshtein_similarity = sum(r['levenshtein_similarity'] for r in comparison_results) / len(comparison_results)

        self.save_comparison_to_word(comparison_results, avg_simple_similarity, avg_levenshtein_similarity,"Logs.docx")

        return jsonify({
            "status": "success",
            "avg_simple_similarity": avg_simple_similarity,
            "avg_levenshtein_similarity": avg_levenshtein_similarity
        }), 200
