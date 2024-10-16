import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from Model.module import vietocr_module
from Model.module import east_model as east_text_detection  # Giả sử bạn đã lưu code EAST trong module này

def convert_images(files):
    all_predictions = []
    
    # Loop through all uploaded files
    for file in files:
        # Đọc file như là một mảng bytes
        img_bytes = file.read()
        
        # Chuyển đổi bytes thành mảng numpy
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Đọc ảnh sử dụng OpenCV
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Thực hiện phát hiện văn bản sử dụng EAST
        _, text_regions = east_text_detection.east_text_detection(cv_image)
        
        # Xử lý từng vùng văn bản được phát hiện
        for _, region in text_regions:
            # Chuyển đổi region từ BGR sang RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Chuyển đổi sang đối tượng PIL Image
            image_pil = Image.fromarray(region_rgb)
            
            # Thực hiện dự đoán OCR
            str_pred = vietocr_module.vietOCR_prediction(image_pil)
            all_predictions.append(str_pred)

    return '\n'.join(all_predictions)

# Sử dụng hàm
# east_model_path = 'path_to_your_east_model.pb'
# result = convert_images(uploaded_files, east_model_path)
# print(result)