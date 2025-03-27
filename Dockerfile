# Use official Python slim image
FROM python:3.10-slim

# Install system packages: tesseract, poppler, OpenCV dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && apt-get clean

# Set working directory inside container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app code
COPY . .

# Command to run your FastAPI app
CMD ["uvicorn", "openai_chat:app", "--host", "0.0.0.0", "--port", "8000"]
