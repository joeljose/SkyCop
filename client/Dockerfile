FROM python:3.10-slim

# System deps for CARLA client + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install CARLA Python API + common deps
RUN pip install --no-cache-dir \
    carla==0.9.16 \
    numpy \
    opencv-python-headless \
    matplotlib \
    pygame \
    flask

# Create non-root home
RUN mkdir -p /home/carla && chmod 777 /home/carla

WORKDIR /app
