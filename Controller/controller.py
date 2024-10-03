# image_controller.py
import os
from PIL import Image
import numpy as np
import cv2
from Model.module import vietocr_module as vietocr_module

from Model.module import crop_text_line_paddle as segmentsPaddle

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
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(arr):
                # Tạo tên file
                filename = f"segment{i}.png"
                filepath = os.path.join(output_dir, filename)
                pil_img = Image.fromarray(np.uint8(img))
                pil_img.save(filepath)

            print(f"Đã lưu {len(arr)} ảnh vào thư mục {output_dir}")

            for img_segment in arr:
                np_image = np.asarray(img_segment)
                image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                str_pred = vietocr_module.vietOCR_prediction(image_pil)
                all_predictions.append(str_pred)
        else:
            # Sử dụng trực tiếp cv_image nếu chiều cao <= 40px
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            str_pred = vietocr_module.vietOCR_prediction(image_pil)
            all_predictions.append(str_pred)

    return '\n'.join(all_predictions)

