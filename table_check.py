import io
import cv2
import numpy as np
import requests
import zipfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PyPDF2 import PdfReader, PdfWriter
import pytesseract
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures


app = FastAPI()

# Set Tesseract OCR Path (If on Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Allowed Origins for CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://yourdomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ConvertAPI Token
CONVERT_API_TOKEN = "token_TfNrYD5l"
CONVERT_API_URL = f"https://v2.convertapi.com/convert/pdf/to/jpg?Secret={CONVERT_API_TOKEN}"

# Keywords for OCR Search
KEYWORDS = [
    "EQUITY AND LIABILITIES",
    "Income from Operations",
    "OPERATING ACTIVITIES",
    "operating profit before working capital changes",
    "Cash and cash equivalents at beginning of period",
    "cash flow statement",
    "Balance sheet as at",
    "Note No",
    "Tax Expense"
]

@app.get("/")
def home():
    return RedirectResponse(url="/docs")


def convert_pdf_to_images(pdf_bytes):
    """
    Uses ConvertAPI to convert a PDF to JPG images.
    Returns a list of image URLs.
    """
    files = {"file": ("file.pdf", pdf_bytes, "application/pdf")}
    response = requests.post(CONVERT_API_URL, files=files)

    if response.status_code == 200:
        image_urls = [file["Url"] for file in response.json()["Files"]]
        return image_urls
    else:
        print("ConvertAPI Error:", response.text)
        return []


def download_image(image_url):
    """
    Downloads an image from a given URL.
    """
    response = requests.get(image_url)
    if response.status_code == 200:
        return io.BytesIO(response.content)  # Convert image to in-memory object
    return None


def detect_table(image_data) -> bool:
    """
    Detects table-like structures using OpenCV.
    """
    start_time = time.perf_counter()
    
    image = np.array(cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        ~blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-2
    )

    horizontalsize = max(1, binary.shape[1] // 20)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(binary, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    verticalsize = max(1, binary.shape[0] // 20)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(binary, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    table_mask = cv2.add(horizontal, vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_mask = cv2.dilate(table_mask, kernel, iterations=1)

    intersections = cv2.bitwise_and(horizontal, vertical)
    intersection_count = cv2.countNonZero(intersections)

    end_time = time.perf_counter()
    print(f"Time taken by detect_table: {end_time - start_time:.4f} seconds")

    return intersection_count > 10


def check_for_keywords(image_data) -> bool:
    """
    Extracts text from an image using OCR and checks for any keywords.
    """
    start_time = time.perf_counter()
    
    # Load image for OCR
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use Tesseract OCR
    custom_config = r'--oem 1 --psm 6 -l eng'
    text = pytesseract.image_to_string(binary, config=custom_config).lower()

    for kw in KEYWORDS:
        if kw.lower() in text:
            end_time = time.perf_counter()
            print(f"Time taken by check_for_keywords: {end_time - start_time:.4f} seconds")
            return True

    end_time = time.perf_counter()
    print(f"Time taken by check_for_keywords: {end_time - start_time:.4f} seconds")
    return False


def process_image(image_url):
    """
    Downloads an image, detects tables, and extracts keywords.
    """
    image_data = download_image(image_url)
    if not image_data:
        return None

    if detect_table(image_data):
        image_data.seek(0)  # Reset pointer for OCR processing
        if check_for_keywords(image_data):
            return image_url
    return None


def process_file(file_tuple):
    """
    Handles full PDF processing.
    """
    start_time = time.perf_counter()
    
    fname, file_bytes = file_tuple
    image_urls = convert_pdf_to_images(file_bytes)

    if not image_urls:
        return None  # No images found

    # Process images using multithreading
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_image, image_urls))

    selected_images = [url for url in results if url]

    # Create output
    output = io.BytesIO()
    output.write(b"\n".join(url.encode() for url in selected_images))
    output.seek(0)

    end_time = time.perf_counter()
    print(f"Time taken by process_file {fname}: {end_time - start_time:.4f} seconds")
    
    return fname.replace(".pdf", "_tables.txt"), output.getvalue()


@app.post("/extract_tables_bulk/")
async def extract_tables_bulk(files: List[UploadFile] = File(...)):
    start_time = time.perf_counter()
    
    file_data = [(file.filename, await file.read()) for file in files]
    
    processed_files = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, fd): fd[0] for fd in file_data}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                fname, text_bytes = result
                processed_files[fname] = text_bytes

    # Create a ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, text_bytes in processed_files.items():
            zf.writestr(fname, text_bytes)
    zip_buffer.seek(0)

    end_time = time.perf_counter()
    print(f"Time taken by extract_tables_bulk: {end_time - start_time:.4f} seconds")

    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=processed_tables.zip"}
    )
