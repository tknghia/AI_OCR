import cv2
import numpy as np
import os

def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# def east_text_detection(image, min_confidence=0.5, width=320, height=320):
#     original = image.copy()
#     (H, W) = image.shape[:2]

#     # Tạo blob từ ảnh đầu vào mà không ép buộc kích thước cố định
#     # Điều chỉnh kích thước theo tỷ lệ của ảnh gốc để tránh bị méo
#     aspect_ratio = W / float(H)
    
#     if W > width or H > height:
#         # Điều chỉnh lại kích thước dựa trên tỷ lệ
#         if aspect_ratio > 1:
#             newW = width
#             newH = int(width / aspect_ratio)
            
#         else:
#             newH = height
#             newW = int(height * aspect_ratio)
#     else:
#         # Nếu ảnh nhỏ hơn kích thước định sẵn, giữ nguyên kích thước ban đầu
#         newW, newH = W, H
#     # Tính toán lại kích thước mới sao cho là bội số của 32
#     newW = (newW // 32) * 32
#     newH = (newH // 32) * 32

#     image = cv2.resize(image, (newW, newH))

#     # Lưu tỷ lệ co để phục hồi sau khi detect
#     rW = W / float(newW)
#     rH = H / float(newH)

#     image = cv2.resize(image, (newW, newH))
#     (H_resized, W_resized) = image.shape[:2]

#     layerNames = [
#         "feature_fusion/Conv_7/Sigmoid",
#         "feature_fusion/concat_3"
#     ]

#     # Load EAST model
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(current_dir, '..', 'data', 'frozen_east_text_detection.pb')
#     net = cv2.dnn.readNet(model_path)

#     blob = cv2.dnn.blobFromImage(image, 1.0, (W_resized, H_resized),
#                                  (123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#     (scores, geometry) = net.forward(layerNames)

#     (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
#     boxes = non_max_suppression(np.array(rects), probs=confidences)

#     results = []
#     for (startX, startY, endX, endY) in boxes:
#         # Scale ngược lại tọa độ bounding box theo tỷ lệ ảnh gốc
#         startX = int(startX * rW)
#         startY = int(startY * rH)
#         endX = int(endX * rW)
#         endY = int(endY * rH)

#         # Thêm padding
#         dX = int((endX - startX) * 0.05)
#         dY = int((endY - startY) * 0.05)
#         startX = max(0, startX - dX)
#         startY = max(0, startY - dY)
#         endX = min(W, endX + (dX * 2))
#         endY = min(H, endY + (dY * 2))

#         # Extract the ROI
#         roi = original[startY:endY, startX:endX]
#         results.append(((startX, startY, endX, endY), roi))

#         # Vẽ bounding box trên ảnh gốc
#         cv2.rectangle(original, (startX, startY), (endX, endY), (0, 255, 0), 2)

#     return original, results

def east_text_detection(image, min_confidence=0.5, width=320*8, height=320*8):
    original = image.copy()
    (H, W) = image.shape[:2]

    # Tạo blob từ ảnh đầu vào
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'data', 'frozen_east_text_detection.pb')
    net = cv2.dnn.readNet(model_path)

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Thêm padding
        dX = int((endX - startX) * 0.05)
        dY = int((endY - startY) * 0.05)
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))

        # Extract the ROI
        roi = original[startY:endY, startX:endX]
        results.append(((startX, startY, endX, endY), roi))

        # Vẽ bounding box trên ảnh gốc
        cv2.rectangle(original, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return original, results

# Hàm non_max_suppression để loại bỏ các bounding box trùng lặp
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Sử dụng hàm
# image = cv2.imread('path_to_your_image.jpg')
# east_model_path = 'path_to_east_text_detection.pb'
# result_image, text_regions = east_text_detection(image, east_model_path)

# Hiển thị kết quả
# cv2.imshow("Text Detection Result", result_image)
# cv2.waitKey(0)

# In số lượng vùng văn bản được phát hiện
# print(f"Số lượng vùng văn bản được phát hiện: {len(text_regions)}")

# Hiển thị từng vùng văn bản
# for i, (box, region) in enumerate(text_regions):
#     print(f"Vùng văn bản {i + 1}: {box}")
#     cv2.imshow(f"Text Region {i + 1}", region)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
