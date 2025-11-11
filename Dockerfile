FROM python:3.10-slim

LABEL maintainer="your-email@example.com"
LABEL description="Malware Detection System with LLM"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data/processed models results logs backups

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=http://ollama:11434

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.maxUploadSize=100"]