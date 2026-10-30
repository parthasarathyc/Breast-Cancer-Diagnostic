# ========================================================
# Stage 1: Build Modern React Frontend
# ========================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend ./
RUN npm run build

# ========================================================
# Stage 2: Python Runtime & FastAPI Multi-Agent Engine
# ========================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV PYTHONPATH=/app/backend

WORKDIR /app

# Install system dependencies for OpenCV and PDF generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch wheels
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (Required for Hugging Face Spaces UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user

# Copy backend code and dataset samples
COPY backend /app/backend

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend_dist

# Create writable directories for reports, models, and cache
RUN mkdir -p /app/backend/reports /app/backend/models /home/user/.cache && \
    chown -R user:user /app /home/user

USER user

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI backend with unified static frontend
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "7860"]
