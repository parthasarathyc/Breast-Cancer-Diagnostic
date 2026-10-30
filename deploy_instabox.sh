#!/bin/bash
set -e

echo "=========================================================="
echo "  Deploying Breast CARE-MA on InstaBox Cloud VM"
echo "=========================================================="

# 1. Update system packages and install system dependencies for OpenCV & PDF
echo "[*] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 nodejs npm

# 2. Build the React Frontend
echo "[*] Building React Frontend..."
cd frontend
npm install
npm run build
cd ..

# 3. Create Python Virtual Environment
echo "[*] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install CPU PyTorch & Backend Dependencies
echo "[*] Installing PyTorch (CPU) and AI libraries..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# 5. Launch the Unified Application (Port 8000)
echo "=========================================================="
echo "[SUCCESS] Breast CARE-MA is ready on InstaBox!"
echo "Starting FastAPI Multi-Agent Engine on Port 8000..."
echo "=========================================================="
export PYTHONPATH=backend
python3 -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
