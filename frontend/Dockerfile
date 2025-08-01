FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY frontend/ ./frontend/
COPY common_utils/ ./common_utils/

# Expose NiceGUI default port
EXPOSE 8080

# Start NiceGUI application
CMD ["python", "app_ui.py"]