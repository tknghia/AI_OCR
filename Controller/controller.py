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

    def log_predictions(self, predictions, upload_time):
        try:
            # Log predictions to MongoDB
            mongo_log_entry = {"upload_time": upload_time, "predictions": predictions}
            self.log_collection.insert_one(mongo_log_entry)
            self.collection.insert_one({"predictions": predictions})

            # If CCCD is detected, log additional info
            if "CĂN CƯỚC CÔNG DÂN" in predictions:
                extracted_info = ImageController.extract_info(predictions)
                self.collection_cccd.insert_one({"cccd": extracted_info})

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

                output_dir = "output_images"
                current_dir = os.path.dirname(os.path.abspath(__file__))
                beginNumber = len(os.listdir(os.path.join(os.path.dirname(current_dir), output_dir)))
                os.makedirs(output_dir, exist_ok=True)
                
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
        self.mongo_controller.log_predictions(predictions, upload_time)

        return predictions

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



class FileController:
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

    @staticmethod
    def save_labels(labels):
        log_file_path = os.path.join(project_root, "Logs", "train.txt")
        
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        with open(log_file_path, 'r', encoding='utf-8') as file:
            existing_lines_count = len(file.readlines())

        new_entries = [f"{i}.png\t{label}" for i, label in enumerate(labels, start=existing_lines_count)]

        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write("\n".join(new_entries) + "\n")

        return jsonify({"status": "success"}), 200
