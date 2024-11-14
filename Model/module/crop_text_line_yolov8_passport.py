from datetime import datetime
import secrets
import string
import cv2
from ultralytics import YOLO
import os

# Get current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
pj_dir = os.path.dirname(os.path.dirname(current_dir))
MODEL_PATH = os.path.join(pj_dir, "model", "data", "best_passport.pt")

# Class names mapping - removed mrz1 and mrz2
class_names = {
    0: 'address',
    1: 'code',
    2: 'dob',
    3: 'doi',
    4: 'exp',
    5: 'gender',
    6: 'id',
    7: 'mrz1',
    8: 'mrz2',
    9: 'name',
    10: 'nation',
    11: 'nationality',
    12: 'poi'
}

# Define desired order - removed mrz1 and mrz2
desired_order = [
    'nation', 
    'code', 
    'name', 
    'nationality', 
    'dob', 
    'gender', 
    'address', 
    'id',
    'doi', 
    'exp'
]

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

def generate_random_id(length=5):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def extract_image_segments(cv_image):
    # Convert to RGB
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)

    # Initialize list for ordered segments
    ordered_segments = []
    
    # Create temporary storage for all detected segments
    detected_segments = {}
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get coordinates and convert to numpy
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
            
            # Get class name using integer class_id
            class_name = class_names.get(class_id)
            if class_name is None:
                continue

            # Store segment info
            if class_name not in detected_segments:
                detected_segments[class_name] = []
            
            # Append the segment info as a dictionary to the list
            detected_segments[class_name].append({
                "id": segment_id,
                "roi": roi,
                "confidence": confidence,
                "type": class_name,
                "coordinates": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }
            })
    
    # Post-process: Handle multiple 'name' fields
    if 'name' in detected_segments and detected_segments['name']:
        for i, segment in enumerate(detected_segments['name']):
            if i == 1:
                segment['type'] = 'surname'  # First 'name' becomes 'surname'
            # Second and subsequent 'name' entries remain as 'name'

    # Arrange segments in desired order
    for field in desired_order:
        if field in detected_segments:
            ordered_segments.extend(detected_segments[field])
    
    return ordered_segments