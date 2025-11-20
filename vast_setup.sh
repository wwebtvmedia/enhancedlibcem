#!/bin/bash
# Vast.ai Auto-Setup Script
# Run this on Vast.ai instance after SSH connection
# Usage: bash /root/vast_setup.sh

set -e

echo "=========================================="
echo "Vast.ai Auto-Setup Script"
echo "=========================================="
echo ""

# Step 1: Update system
echo "1️⃣ Updating system packages..."
apt-get update -qq > /dev/null 2>&1
apt-get upgrade -qq -y > /dev/null 2>&1
echo "✅ System updated"

# Step 2: Install dependencies
echo ""
echo "2️⃣ Installing Python and dependencies..."
apt-get install -qq -y python3 python3-pip git wget curl > /dev/null 2>&1
echo "✅ Python installed: $(python3 --version)"

# Step 3: Install PyTorch
echo ""
echo "3️⃣ Installing PyTorch with CUDA 11.8..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ PyTorch installed"

# Step 4: Install ML libraries
echo ""
echo "4️⃣ Installing ML libraries..."
pip install -q scikit-image scipy numba numpy pillow tensorboard tqdm
echo "✅ ML libraries installed"

# Step 5: Create working directory
echo ""
echo "5️⃣ Creating working directories..."
mkdir -p /root/training/data
mkdir -p /root/training/outputs
mkdir -p /root/training/checkpoints
echo "✅ Directories created"

# Step 6: Verify installation
echo ""
echo "6️⃣ Verifying installation..."
python3 << 'PYTHON_SCRIPT'
import torch
import numpy as np

print("Python version:", __import__('sys').version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("NumPy version:", np.__version__)
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your code (enhancedlibcem.py) to /root/training/"
echo "2. Create train_vast.py (see VAST_AI_SETUP.md)"
echo "3. Run: cd /root/training && python3 train_vast.py"
echo ""
echo "Monitor training in another SSH terminal:"
echo "  tail -f /root/training/outputs/training_*.log"
echo ""
