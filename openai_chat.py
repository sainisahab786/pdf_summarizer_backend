import io
import cv2
import numpy as np
import concurrent.futures
import zipfile
from pdf2image import convert_from_bytes
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PyPDF2 import PdfReader, PdfWriter
import pytesseract
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

app = FastAPI()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://yourdomain.com",
    "*"
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allows requests from these origins
    allow_credentials=True,       # Allows cookies and authentication headers
    allow_methods=["*"],          # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allows all HTTP headers
)

KEYWORDS = [
    "EQUITY AND LIABILITIES",
    "Income from Operations",
    "OPERATING ACTIVITIES",
    "operating profit before working captial changes",
    "Cash and cash equivalents at beginning of period",
    "cash flow statement",
    "Balance sheet as at",
    "Other Long-Term Liabilities",
    "Note No",
    "Loss before tax",
    "Tax Expense",
    "Trade Payables",
    "Total outstanding dues to creditors other than MSMEs",
    "Property, Plant & Equipment and Intangible Assets",
    "Deferred Tax Assets (Net)",
    "Short-Term Loans and Advances",
    "Cost of Raw Materials consumed ",
    "Cost of materials consumed ",
    "Distributable Profits/ Profits Trfd to Reserves",
    "Profit/(loss) for the period",
    "Adjustment for Non Operative Income/Expenditure",
    "Provisions in respect of Tax earlier year",
    "MAT Credit Entitlement Reversal",
    "Dividend on Shares & Miscellaneous Income",
    "Dividend on shares",
    "(Profit)/Loss on Sale/Disposal of Property, Plant & Equipment's",
    "Dividend on shares",
    "Dividend & Misc. Income"    
]

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

def detect_table(pil_image) -> bool:
    """
    Detects table-like structures using morphological operations.
    """
    start_time = time.perf_counter()  # Start timer
    
    image = np.array(pil_image)
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

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_detected = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w > 50 and h > 50 and area > (image.shape[0] * image.shape[1] * 0.005):
            contour_detected = True
            break

    end_time = time.perf_counter()  # End timer
    print(f"Time taken by detect_table: {end_time - start_time:.4f} seconds")
    
    return contour_detected or (intersection_count > 10)

def check_for_keywords(pil_image, keywords) -> bool:
    """
    Optimized OCR using Tesseract to extract text from the image.
    Checks for ANY of the keywords (case-insensitive).
    """
    start_time = time.perf_counter()  # Start timer
    
    # Convert image to grayscale and enhance contrast for better OCR accuracy
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use Tesseract with LSTM OCR Engine and English language for faster processing
    custom_config = r'--oem 1 --psm 6 -l eng'

    text = pytesseract.image_to_string(binary, config=custom_config).lower()
    
    for kw in keywords:
        if kw.lower() in text:
            end_time = time.perf_counter()  # End timer
            print(f"Time taken by check_for_keywords: {end_time - start_time:.4f} seconds")
            return True
    
    end_time = time.perf_counter()  # End timer
    print(f"Time taken by check_for_keywords: {end_time - start_time:.4f} seconds")
    return False

def process_page(idx_image):
    """
    Processes a single page to check for tables and keywords.
    """
    start_time = time.perf_counter()  # Start timer
    
    i, pil_image = idx_image

    # If no table is detected, skip the page immediately
    if not detect_table(pil_image):
        return None  

    # If table is detected, check for keywords
    if check_for_keywords(pil_image, KEYWORDS):
        end_time = time.perf_counter()  # End timer
        print(f"Time taken by process_page {i}: {end_time - start_time:.4f} seconds")
        return i
    
    end_time = time.perf_counter()  # End timer
    print(f"Time taken by process_page {i}: {end_time - start_time:.4f} seconds")
    return None


def process_file(file_tuple):
    """
    Processes a single PDF file:
    - Reads the file bytes.
    - Converts PDF pages to images with optimized threading.
    - Processes pages using multithreading.
    - Extracts pages that pass the table & keyword checks.
    """
    start_time = time.perf_counter()  # Start timer

    fname, file_bytes = file_tuple
    images = convert_from_bytes(file_bytes, dpi=300, thread_count=4)  # Optimize with 4 threads
    
    # Multiprocessing for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_page, enumerate(images)))
        
    selected_pages = [r for r in results if r is not None]

    reader = PdfReader(io.BytesIO(file_bytes))
    writer = PdfWriter()
    for page_num in selected_pages:
        writer.add_page(reader.pages[page_num])
    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    download_name = fname.replace(".pdf", "_tables.pdf")
    
    end_time = time.perf_counter()  # End timer
    print(f"Time taken by process_file {fname}: {end_time - start_time:.4f} seconds")
    
    return download_name, output.getvalue()


@app.post("/extract_tables_bulk/")
async def extract_tables_bulk(files: List[UploadFile] = File(...)):
    start_time = time.perf_counter()  # Start timer
    
    file_data = []
    for file in files:
        content = await file.read()
        file_data.append((file.filename, content))
    
    # Multithreading for I/O-bound tasks
    processed_files = {}
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = {executor.submit(process_file, fd): fd[0] for fd in file_data}
        for future in concurrent.futures.as_completed(futures):
            fname, pdf_bytes = future.result()
            processed_files[fname] = pdf_bytes

    # Create a ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, pdf_bytes in processed_files.items():
            zf.writestr(fname, pdf_bytes)
    zip_buffer.seek(0)
    
    end_time = time.perf_counter()  # End timer
    print(f"Time taken by extract_tables_bulk: {end_time - start_time:.4f} seconds")
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=processed_tables.zip"}
    )
