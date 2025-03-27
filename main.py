from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import zipfile
import os
from io import BytesIO
import pandas as pd
from paddleocr import PaddleOCR
import pytesseract

# Initialize FastAPI app
app = FastAPI(title="PDF Table Extraction using PaddleOCR")

POPPLER_PATH = r"C:\Users\Omkar\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# from pdf2image import convert_from_bytes
# import cv2
# import numpy as np
# from PIL import Image
# import zipfile
# import os
# import pandas as pd
# from paddleocr import PaddleOCR

# # # Initialize FastAPI app
# app = FastAPI(title="PDF Table Extraction with Alignment using PaddleOCR")

# POPPLER_PATH = r"C:\Users\Omkar\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin"

# Initialize PaddleOCR with table detection
ocr = PaddleOCR(use_angle_cls=True, lang='en')


OUTPUT_ZIP_FILE = "extracted_tables.zip"

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Read PDF file
    pdf_bytes = await file.read()

    # Convert PDF to images
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)

    extracted_tables = []

    for i, img in enumerate(images):
        # Convert PIL image to OpenCV format
        img_cv = np.array(img)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

        # Apply Adaptive Thresholding
        img_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

        # Detect text using PaddleOCR
        results = ocr.ocr(img_threshold, cls=True)

        table_data = []
        text_positions = []

        for result in results:
            for line in result:
                text = line[1][0]  # Extract detected text
                bbox = line[0]  # Bounding box coordinates (x1, y1, x2, y2)

                x_min = min(bbox[0][0], bbox[1][0])  # Get min x for sorting
                y_min = min(bbox[0][1], bbox[2][1])  # Get min y for row alignment

                text_positions.append((y_min, x_min, text))

        # Sort the text first by Y-axis (top to bottom), then by X-axis (left to right)
        text_positions.sort(key=lambda x: (x[0], x[1]))

        # Organize text into rows with proper spacing
        last_y = None
        last_x = None
        row = []
        structured_table = []

        SPACE_THRESHOLD = 4  # Space threshold in pixels to move text to a new column

        for y, x, text in text_positions:
            if last_y is None or abs(y - last_y) < 10:  # Same row
                if last_x is not None and abs(x - last_x) > SPACE_THRESHOLD:
                    # If space is large, move to a new column
                    row.append("")
                row.append(text)
            else:
                structured_table.append(row)  # Store previous row
                row = [text]  # Start new row

            last_y = y
            last_x = x

        if row:
            structured_table.append(row)  # Add last row

        # Save detected table as CSV
        table_csv = f"table_{i+1}.csv"
        df = pd.DataFrame(structured_table)
        df.to_csv(table_csv, index=False, header=False)

        extracted_tables.append(table_csv)

    # Create ZIP file with extracted tables
    with zipfile.ZipFile(OUTPUT_ZIP_FILE, "w") as zip_file:
        for table_csv in extracted_tables:
            zip_file.write(table_csv)
            os.remove(table_csv)  # Cleanup

    return FileResponse(OUTPUT_ZIP_FILE, media_type="application/zip", filename="extracted_tables.zip")


@app.get("/")
def root():
    return {"message": "Upload a PDF to extract tables with alignment using PaddleOCR."}