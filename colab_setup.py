"""
Google Colab Setup Script for Enhanced Fractal Tokenization Model
Run this in the first cell of your Colab notebook
"""

import subprocess
import sys

def install_requirements():
    """Install all required packages for Colab"""
    
    print("🔧 Installing required packages for Google Colab...\n")
    
    # List of packages to install
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",  # PyTorch with CUDA
        "clip-by-openai",  # CLIP model
        "matplotlib",
        "numpy",
        "tqdm",
        "Pillow"
    ]
    
    for package in packages:
        print(f"📦 Installing: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + package.split())
        print(f"✓ {package} installed\n")
    
    print("✅ All packages installed successfully!\n")

def setup_colab_environment():
    """Setup Colab-specific environment"""
    
    print("⚙️  Setting up Colab environment...\n")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted at /content/drive\n")
    except ImportError:
        print("⚠️  Not running in Colab - skipping Google Drive mount\n")
    
    # Create working directory
    import os
    os.makedirs('/content/enhancedlibcem', exist_ok=True)
    os.makedirs('/content/checkpoints', exist_ok=True)
    os.makedirs('/content/data', exist_ok=True)
    
    print("✓ Working directories created\n")
    print("📁 Directories structure:")
    print("  - /content/enhancedlibcem/  (main code)")
    print("  - /content/checkpoints/     (model checkpoints)")
    print("  - /content/data/            (datasets)\n")

def download_code_from_github():
    """Download the code from GitHub (optional)"""
    
    print("\n📥 To download code from GitHub in Colab, use:\n")
    print("!git clone https://github.com/YOUR_USERNAME/enhancedlibcem.git /content/enhancedlibcem")
    print("\nOr upload the enhancedlibcem.py file directly to Colab.\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GOOGLE COLAB SETUP FOR ENHANCED FRACTAL TOKENIZATION MODEL".center(80))
    print("="*80 + "\n")
    
    install_requirements()
    setup_colab_environment()
    download_code_from_github()
    
    print("="*80)
    print("✅ SETUP COMPLETE - Ready to run the model!".center(80))
    print("="*80 + "\n")
