# image_controller.py
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
        
        # Perform segmentation and OCR prediction
        arr = segmentsPaddle.extract_image_segments(cv_image)

        for img_segment in arr:
            np_image = np.asarray(img_segment)
            image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            str_pred = vietocr_module.vietOCR_prediction(image_pil)
            all_predictions.append(str_pred)

    return '\n'.join(all_predictions)
