import numpy as np
import cv2
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(lang="en")

def extract_image_segments(cv_image):
    # Ensure the image is in RGB format if it's in BGR (OpenCV default is BGR)
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Perform OCR on the image using PaddleOCR
    result = ocr.ocr(img_rgb)

    # Initialize arrays for bounding boxes and segments
    bounding_boxes = []
    segments = []

    # Loop through the OCR result to get bounding boxes and extract text
    for i in range(len(result[0])):
        box = result[0][i]  # Bounding box
        text = result[0][i][0]  # Detected text
        score = result[0][i][1]  # Confidence score

        # Coordinates of the top-left and bottom-right corners
        top_left = (int(box[0][0][0]), int(box[0][0][1]))  # Top-left corner
        bottom_right = (int(box[0][2][0]), int(box[0][2][1]))  # Bottom-right corner

        # Draw rectangle around the detected text (optional)
        cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 2)

        # Append bounding box coordinates and text
        bounding_boxes.append((top_left, bottom_right, text, score))

    # Crop the image segments based on bounding boxes
    for box in bounding_boxes:
        top_left = box[0]  # Top-left corner
        bottom_right = box[1]  # Bottom-right corner

        # Get x and y coordinates
        x_min = int(top_left[0]) - 4
        x_max = int(bottom_right[0]) + 4
        y_min = int(top_left[1]) - 4
        y_max = int(bottom_right[1]) + 4

        # Crop the region of interest (ROI)
        roi = cv_image[y_min:y_max, x_min:x_max]
        segments.append(roi)

    # Return the list of image segments
    return segments
