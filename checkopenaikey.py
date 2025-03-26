from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from fastapi.responses import FileResponse
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained EAST text detector model
east_model_path = "C:/Users/Omkar/vedansh_project/models/frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_model_path)

# Function to detect text regions
def detect_text_regions(image):
    orig = image.copy()
    
    # Resize while maintaining aspect ratio
    (H, W) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))

    # Define output layers
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Process detections
    boxes = []
    confidences = []

    for y in range(scores.shape[2]):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(scores.shape[3]):
            if scoresData[x] < 0.5:
                continue

            # Compute coordinates
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offset_x + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offset_y - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Scale bounding boxes to original image size
    for i in indices.flatten():
        (startX, startY, endX, endY) = boxes[i]
        startX, startY = int(startX * rW), int(startY * rH)
        endX, endY = int(endX * rW), int(endY * rH)

        # Draw bounding box
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig, len(indices)

@app.post("/detect-text/")
async def detect_text_api(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image.convert("RGB"))

    # Detect text
    result_image, text_region_count = detect_text_regions(image)

    # Save the result
    output_path = "text_detection_result.jpg"
    cv2.imwrite(output_path, result_image)

    return FileResponse(output_path, media_type="image/jpeg")

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
