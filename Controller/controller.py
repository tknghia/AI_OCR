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
import matplotlib.pyplot as plt
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
            result = self.log_collection.find_one({"_id": object_id})
            
            if result:
                return result
            else:
                return f"No prediction found with id: {id}"

        except Exception as e:
            # Log the error and return a meaningful error message
            print(f"Error while retrieving prediction with id {id}: {e}")
            return f"Error while retrieving prediction: {e}"

    def log_predictions(self, predictions, upload_time,list_images):
        try:
            # Log predictions to MongoDB
            mongo_log_entry = {"upload_time": upload_time, "predictions": predictions,"list_images":list_images}
            logs_prediction=self.log_collection.insert_one(mongo_log_entry)
            result_prediction=self.collection.insert_one({"predictions": predictions})

            # If CCCD is detected, log additional info
            if "CĂN CƯỚC CÔNG DÂN" in predictions:
                extracted_info = ImageController.extract_info(predictions)
                self.collection_cccd.insert_one({"cccd": extracted_info})
            inserted_document = self.log_collection.find_one({"_id": logs_prediction.inserted_id})

            return str(inserted_document["_id"])

        except Exception as e:
            print(f"Error writing to MongoDB: {e}")

    def update_labels(self,object_id, list_labels):
        try:
            # Tìm object trong MongoDB với _id đã cho
            query = {"_id": ObjectId(object_id)}
            document = self.log_collection.find_one(query)

            if not document:
                print(f"No document found with id: {object_id}")
                return
            
            # Kiểm tra xem độ dài của list_labels và list_images có bằng nhau không
            if len(list_labels) != len(document['list_images']):
                print("Length of list_labels does not match length of list_images")
                return
            
            # Cập nhật label cho từng phần tử trong list_images
            for i, label in enumerate(list_labels):
                # Sử dụng cú pháp $set với arrayFilters để cập nhật phần tử trong danh sách
                self.log_collection.update_one(
                    {"_id": ObjectId(object_id)},
                    {"$set": {f"list_images.{i}.label": label}}
                )
            
            print("Labels updated successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")


class ImageController:
    mongo_controller = MongoController()  # Initialize MongoController
    # cropped_images_metadata=[]
    # Hàm giảm độ sáng cho hình quá sáng
    def reduce_brightness_contrast(image):
        # Chuyển ảnh sang thang xám (grayscale)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng CLAHE để cải thiện chi tiết ảnh sáng quá
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Giảm độ sáng và tương phản
        alpha = 0.8  # Giảm tương phản
        beta = -30   # Giảm độ sáng
        dark_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)

        return dark_image
    
    # Hàm tăng độ sáng cho hình quá tối
    def enhance_brightness_contrast(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng CLAHE để tăng cường chi tiết
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Tăng độ sáng và tương phản
        alpha = 1.5
        beta = 50
        bright_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)

        return bright_image

    # Hàm kiểm tra độ sáng
    def check_brightness(image):
        # Chuyển ảnh sang thang xám (grayscale)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Tính giá trị trung bình của các pixel
        brightness_mean = np.mean(gray_image)

        return brightness_mean

    # Hàm thay đổi độ sáng và tương phản
    def adjust_image_brightness(image):
        # Kiểm tra độ sáng trung bình
        brightness_mean = ImageController.check_brightness(image)
        
        # Ngưỡng xác định quá tối hoặc quá sáng
        too_dark_threshold = 50  # Ngưỡng cho ảnh quá tối
        too_bright_threshold = 200  # Ngưỡng cho ảnh quá sáng

        # Nếu ảnh quá tối, tăng độ sáng và tương phản
        if brightness_mean < too_dark_threshold:
            print("Image is too dark. Increasing brightness...")
            return ImageController.enhance_brightness_contrast(image)  # Gọi hàm tăng độ sáng

        # Nếu ảnh quá sáng, giảm độ sáng và tương phản
        elif brightness_mean > too_bright_threshold:
            print("Image is too bright. Reducing brightness...")
            return ImageController.reduce_brightness_contrast(image)  # Gọi hàm giảm độ sáng
        
        # Nếu độ sáng bình thường, không làm gì
        else:
            print("Image brightness is normal.")
            return image
        

    def detect_logo_and_rotate(image):
        # Lấy thư mục hiện tại của file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Tạo đường dẫn tới file logo
        logo_path = os.path.join(current_dir, '..', 'View', 'static', 'images', 'logo_cccd.png')
        
        # Tải ảnh logo tham chiếu
        logo_ref = cv2.imread(logo_path, 0)  # Tải dưới dạng ảnh xám
        
        # Tạo bộ dò SIFT
        sift = cv2.SIFT_create()

        # Tìm keypoints và descriptors cho logo tham chiếu
        kp_ref, des_ref = sift.detectAndCompute(logo_ref, None)
        
        #Hàm phát hiện logo và trả về vị trí logo
        def locate_logo(img):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            kp_img, des_img = sift.detectAndCompute(gray, None)
            
            if des_img is None:
                return None  # Trả về None nếu không tìm thấy descriptors
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_ref, des_img, k=2)
            
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) > 5:  # Ngưỡng số lượng matches tốt
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = logo_ref.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Tính toán tâm của logo được phát hiện
                center_x = np.mean(dst[:, 0, 0])
                center_y = np.mean(dst[:, 0, 1])
                
                return (center_x, center_y)
            else:
                return None

        def determine_horizontal_orientation(img):
            if ImageController.is_wider_than_tall(img):
                print("khong xoay ngang")
                return img
            else:
                print("xoay ngang")
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # Đưa ảnh về hướng nằm ngang
        horizontal_img = determine_horizontal_orientation(image)
        rolate_img = cv2.rotate(horizontal_img, cv2.ROTATE_180)
        
        # Xác định vị trí logo
        logo_position = locate_logo(horizontal_img)
        
        if logo_position is None:
            # Nếu không tìm thấy khuôn mặt trên ảnh gốc
            if ImageController.is_face_on_left(image) is None:
                print("khong thay logo va face xoay de kiem tra")
                if ImageController.is_face_on_left(rolate_img) is None:
                    print("Khong thay khuon mat sau khi xoay")
                    return horizontal_img
                else:
                    print("thay face sau khi xoay")
                    return rolate_img
            # Nếu tìm thấy khuôn mặt trên ảnh gốc
            else:
                print("Thay face truoc khi xoay")
                return horizontal_img
                
        height, width = horizontal_img.shape[:2]
        center_x, center_y = logo_position
        
        # Kiểm tra xem logo có ở góc trái trên không
        if center_x < width / 2 and center_y < height / 2:
            print("Logo o goc trai tren, khong xoay")
            return horizontal_img
        else:
            print("Logo o goc phai duoi, xoay 180 do")
            return cv2.rotate(horizontal_img, cv2.ROTATE_180)

    #Hàm kiểm tra chiều rộng hình ảnh có lớn chiều cao không
    def is_wider_than_tall(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Kiểm tra nếu chiều rộng lớn hơn chiều cao
            return w > h
        
        return False  # Nếu không tìm thấy đường viền nào






    def is_face_on_left(img):
        # Khởi tạo face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Chuyển sang ảnh xám nếu cần
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=15,
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            print("Khong phat hien khuon mat")
            return None
            
        # Lấy khuôn mặt lớn nhất (trường hợp có nhiều khuôn mặt)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Tính tâm của khuôn mặt
        face_center_x = x + w/2
        
        # Kiểm tra xem khuôn mặt có nằm bên trái không
        # (so với một nửa chiều rộng của ảnh)
        image_center_x = img.shape[1] / 2
        is_left = face_center_x < image_center_x
        
        print(f"Tam khuon mat: {face_center_x}")
        print(f"Giua hinh anh: {image_center_x}")
        print("Khuon mat nam ben : " + ("trai" if is_left else "phai"))
        
        return is_left





    # Hàm phụ trợ để vẽ khung bao quanh thẻ CCCD (có thể sử dụng để kiểm tra)
    def draw_bounding_box(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return img

    def convert_images(self,files):
        all_predictions = []
        cropped_images_metadata=[]
        # Loop through all uploaded files
        for file in files:
            img = Image.open(file)  # Open the image file directly

            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Gọi hàm xoay hình ảnh
            rotated_image = ImageController.detect_logo_and_rotate(cv_image)

            # Gọi hàm điều chỉnh ánh sáng
            enhanced_image = ImageController.adjust_image_brightness(rotated_image)

            # Kiểm tra ảnh trước khi xoay và viền nhận diện cccd
            # draw = ImageController.draw_bounding_box(cv_image)
            # plt.imshow(draw, cmap='gray')
            # plt.axis('off')  # Tắt hệ trục nếu muốn
            # plt.show()
            
            # Kiểm tra ảnh sau khi xoay và viền nhận diện cccd
            # draw = ImageController.draw_bounding_box(enhanced_image)
            # plt.imshow(draw, cmap='gray')
            # plt.axis('off')  # Tắt hệ trục nếu muốn
            # plt.show()

            # Chuyển sang ảnh dạng màu BGR nếu cần
            if len(enhanced_image.shape) == 1:
                new = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            else:
                new = enhanced_image

            height, width = new.shape[:2]

            if height > 55:
                # Perform segmentation and OCR prediction
                arr = segmentsPaddle.extract_image_segments(new)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(os.path.dirname(current_dir), 'Model', 'dataset', 'output_images')
                os.makedirs(output_dir, exist_ok=True)

                # Save each cropped segment using its unique ID as the filename
                for segment in arr:
                    # Extract ROI and segment ID
                    segment_id = segment["id"]
                    roi = segment["roi"]

                    # Create a timestamp for sorting purposes
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

                    # Create the filename using the timestamp and unique segment ID
                    filename = f"{timestamp}_{segment_id}.png"
                    filepath = os.path.join(output_dir, filename)

                    # Save the image using PIL
                    pil_img = Image.fromarray(np.uint8(roi))
                    pil_img.save(filepath)
                    cropped_images_metadata.append({
                        "filename": filename,
                        "label": None  # Label will be assigned in the /save endpoint
                    })

                print(f"Saved {len(arr)} images to folder {output_dir}")

                    # Perform OCR prediction on each segment and append results
                for segment in arr:
                        np_image = np.asarray(segment["roi"])
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

        return '\n'.join(all_predictions),cropped_images_metadata

    def process_images(self, files):
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = [f"{upload_time} - Uploaded file: {file.filename}" for file in files]

        predictions,list_crop_images = self.convert_images(files)

        # Log to file
        log_file_path = os.path.join(project_root, "Logs", "upload_logs.txt")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        with open(log_file_path, "a", encoding='utf-8') as log_file:
            log_file.write("\n".join(log_entries) + "\n")

        # Use MongoController to log predictions
        prediction_id=self.mongo_controller.log_predictions(predictions,upload_time,list_crop_images)
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
    
    def update_file(self,file_path, old_result, labels):
        # Lấy thông tin từ old_result
        images_belong_to_this_label = old_result["list_images"]

        # Tạo dictionary {filename: label} từ old_result để tra cứu nhanh
        updated_labels_dict = {
            os.path.basename(image["filename"]): label
            for image, label in zip(images_belong_to_this_label, labels)
        }

        # Kiểm tra xem file có tồn tại không
        if os.path.exists(file_path):
            # Đọc nội dung file nếu đã tồn tại
            with open(file_path, "r", encoding='utf-8') as file:
                lines = file.readlines()

            # Cập nhật hoặc giữ nguyên nội dung của file
            updated_lines = []
            existing_filenames = set()  # Lưu các filename đã có trong file

            for line in lines:
                filename, old_label = line.strip().split("\t")
                if filename in updated_labels_dict:
                    # Nếu filename đã tồn tại trong file, cập nhật label mới nếu cần
                    new_label = updated_labels_dict[filename]
                    updated_lines.append(f"{filename}\t{new_label}\n")
                    # Đánh dấu filename này đã xử lý
                    existing_filenames.add(filename)
                else:
                    # Giữ nguyên dòng nếu không cần cập nhật
                    updated_lines.append(line)

            # Thêm các filename chưa tồn tại trong file vào cuối file
            for filename, label in updated_labels_dict.items():
                if filename not in existing_filenames:
                    updated_lines.append(f"{filename}\t{label}\n")

            # Ghi lại nội dung đã cập nhật vào file
            with open(file_path, "w", encoding='utf-8') as file:
                file.writelines(updated_lines)
        else:
            # Nếu file không tồn tại, tạo file mới từ old_result và labels
            new_entries = [f"{os.path.basename(image['filename'])}\t{label}"
                        for image, label in zip(images_belong_to_this_label, labels)]

            # Ghi vào file mới
            with open(file_path, "w", encoding='utf-8') as file:
                file.write("\n".join(new_entries) + "\n")


    
    def save_labels(self, labels, result_id):
        log_file_path = os.path.join(project_root, "Model", "dataset", "output_images", "train.txt")
        test_file_path = os.path.join(project_root, "Model", "dataset", "output_images", "test.txt")
        old_result = self.mongo_controller.select_prediction_by_id(result_id)
        # Ensure the folder exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Use the filenames saved in the `ImageController` during cropping
        # Update filenames with labels and write to train.txt and test.txt
        self.update_file(log_file_path,old_result,labels)

        # Fetch old result for comparison

        if not old_result:
            return jsonify({"status": "error", "message": "Invalid result ID"}), 400

        wrong_labels = old_result['predictions'].split("\n")
        comparison_results = self.compare_labels(wrong_labels, labels)

        # Calculate average correctness percentage
        avg_simple_similarity = sum(r['simple_similarity'] for r in comparison_results) / len(comparison_results)
        avg_levenshtein_similarity = sum(r['levenshtein_similarity'] for r in comparison_results) / len(comparison_results)

        # Save comparison to word document
        self.save_comparison_to_word(comparison_results, avg_simple_similarity, avg_levenshtein_similarity, "Logs.docx")
        self.mongo_controller.update_labels(result_id,labels)
        return jsonify({
            "status": "success",
            "avg_simple_similarity": avg_simple_similarity,
            "avg_levenshtein_similarity": avg_levenshtein_similarity
        }), 200
