FROM python:3.11-slim

WORKDIR /app

# System dependencies (rarely changes - cached)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Requirements first (only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# YOLO model (cached after first download)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# App code last (changes most frequently)
COPY . .

RUN mkdir -p temp models reference_videos

EXPOSE 8000

# Fixed: Use sh -c to properly expand PORT variable
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
