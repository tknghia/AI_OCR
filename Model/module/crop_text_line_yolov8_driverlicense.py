from datetime import datetime
import secrets
import string
import cv2
from ultralytics import YOLO
import os

# Lấy thư mục của tệp hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
pj_dir = os.path.dirname(os.path.dirname(current_dir))
MODEL_PATH = os.path.join(pj_dir, "model", "data", "best_blx.pt")

# Tải mô hình YOLOv8 đã huấn luyện từ file của bạn
model = YOLO(MODEL_PATH)

def generate_random_id(length=5):
    characters = string.ascii_letters + string.digits  # Chữ cái hoa, chữ cái thường, và chữ số
    return ''.join(secrets.choice(characters) for _ in range(length))

def extract_image_segments(cv_image):
    # Ensure the image is in RGB format if it's in BGR (OpenCV default is BGR)
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Perform detection using YOLOv8
    results = model(img_rgb)
    
    # Initialize array for segments
    segments = []
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy()
            
            # Convert coordinates to integers and add padding
            x_min = max(0, int(x1))
            y_min = max(0, int(y1))
            x_max = int(x2)
            y_max = int(y2)
            
            # Crop the region of interest (ROI)
            roi = cv_image[y_min:y_max, x_min:x_max]
            
            # Generate a unique ID for the segment
            segment_id = generate_random_id(5)
            
            segments.append({
                "id": segment_id,
                "roi": roi
            })
    
    return segments
