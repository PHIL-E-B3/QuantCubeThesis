#!/bin/bash
# === FOMC Sentiment Pipeline - Mac/Linux Setup ===
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch..."
# Detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - MPS backend (Apple Silicon) or CPU
    pip install torch torchvision torchaudio
else
    # Linux with CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source venv/bin/activate"
