import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

# Set the path to the Tesseract executable (adjust the path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load pre-trained EAST text detector
net = cv2.dnn.readNet("Text_Detection/frozen_east_text_detection.pb")

def detect_text(image, net, min_confidence=0.5, width=320, height=320):
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Resize the image and grab the new dimensions
    rW = W / float(width)
    rH = H / float(height)
    image = cv2.resize(image, (width, height))
    (H, W) = image.shape[:2]

    # Define the output layer names for the EAST detector model
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Create a blob from the image and perform a forward pass of the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)

    # Get the rows and columns of the scores
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the rows
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # Loop over the columns
        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offsetX, offsetY = (x * 4.0, y * 4.0)
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            endX = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            endY = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scores_data[x])

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        dX = int((endX - startX) * 0.1)
        dY = int((endY - startY) * 0.1)
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(orig.shape[1], endX + (dX * 2))
        endY = min(orig.shape[0], endY + (dY * 2))

        roi = orig[startY:endY, startX:endX]
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        results.append(((startX, startY, endX, endY), text))

    return orig, results