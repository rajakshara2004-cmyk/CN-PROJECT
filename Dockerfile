# 1. Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Set working directory to /app
WORKDIR /app

# 2. Install FFmpeg using apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies from existing requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the complete project into /app
COPY . /app

# Expose default port
EXPOSE 10000

# 5-8. Run Streamlit dashboard at dashboard/app.py, bind to 0.0.0.0,
# use Render's PORT variable with 10000 fallback, and enable headless mode
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true"]
