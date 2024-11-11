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

class_names = {
    0: 'address',
    1: 'birth',
    2: 'class',
    3: 'day',
    4: 'expiry',
    5: 'month',
    6: 'name',
    7: 'nationality',
    8: 'number',
    9: 'place',
    10: 'year',
}

# Điều chỉnh desired_order để không bao gồm current_places
# current_places sẽ được xử lý riêng và thêm vào cuối
desired_order = [
    'number', 
    'name', 
    'birth', 
    'nationality', 
    'address', 
    'class', 
    'place',
    'day',
    'month',
    'year',
    'expiry'
]

# Tải mô hình YOLOv8 đã huấn luyện từ file của bạn
model = YOLO(MODEL_PATH)

def generate_random_id(length=5):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def extract_image_segments(cv_image):
    # Ensure the image is in RGB format if it's in BGR
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Perform detection using YOLOv8
    results = model(img_rgb, conf=0.4)

    # Initialize lists and dictionaries
    ordered_segments = []
    detected_segments = {}
    current_places_segments = []
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].cpu().numpy()]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Convert coordinates to integers
            x_min = max(0, int(x1))
            y_min = max(0, int(y1))
            x_max = int(x2)
            y_max = int(y2)
            
            # Crop the region of interest
            roi = cv_image[y_min:y_max, x_min:x_max]
            
            # Generate segment ID
            segment_id = generate_random_id(5)
            
            # Get class name
            class_name = class_names.get(class_id)
            if class_name is None:
                continue
                
            # Create segment info
            segment_info = {
                "id": segment_id,
                "roi": roi,
                "type": class_name,
                "coordinates": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }
            }
            
            # Xử lý riêng cho current_places
            if class_name == 'current_places':
                current_places_segments.append(segment_info)
            else:
                detected_segments[class_name] = segment_info
            # Sắp xếp các trường theo desired_order
    for field in desired_order:
        if field in detected_segments:
            ordered_segments.append(detected_segments[field])
    
    # Thêm tất cả current_places vào cuối
    # Sắp xếp current_places theo tọa độ y để đảm bảo thứ tự từ trên xuống
    current_places_segments.sort(key=lambda x: x['coordinates']['y_min'])
    ordered_segments.extend(current_places_segments)
    
    return ordered_segments