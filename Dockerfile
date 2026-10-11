# =============================================================================
# Dockerfile — containerised Handwritten Digit Recogniser
#
# Build:  docker build -t digit-recognizer .
# Run:    docker run -p 8501:8501 digit-recognizer
# Open:   http://localhost:8501
# =============================================================================
FROM python:3.11-slim

# System libraries TensorFlow / Pillow / Streamlit need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

EXPOSE 8501

# headless = don't try to open a browser; 0.0.0.0 = reachable from outside.
CMD ["streamlit", "run", "webapp/app.py", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
