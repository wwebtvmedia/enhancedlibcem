"""
GOOGLE COLAB NOTEBOOK - Enhanced Fractal Tokenization Model
Copy this code into Google Colab cells sequentially

Cell 1: Setup and Dependencies
"""

# ==================== CELL 1: SETUP ====================
# Run this cell first to install all dependencies

!pip install -q torch torchvision torchaudio
!pip install -q open-clip-torch
!pip install -q matplotlib numpy tqdm Pillow
!pip install -q git+https://github.com/openai/CLIP.git

# Mount Google Drive (optional - for saving/loading from Drive)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted")
except:
    print("⚠ Skipping Google Drive mount (not in Colab)")

# Create necessary directories
import os
os.makedirs('/content/checkpoints', exist_ok=True)
os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/results', exist_ok=True)

print("\n✅ Setup complete!\n")


# ==================== CELL 2: IMPORT MAIN CODE ====================
# Upload enhancedlibcem.py or paste the full code here

# Option A: Upload from local file
# Upload the enhancedlibcem.py file using the file manager
# Then load it:

import sys
sys.path.append('/content')

# If you uploaded the file, import it:
# from enhancedlibcem import *

# Option B: Paste the entire enhancedlibcem.py code here
# (Copy all code from enhancedlibcem.py into this cell)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import clip
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import logging
import numpy as np
from torchvision.models import vgg16

# [PASTE THE REST OF enhancedlibcem.py CODE HERE]
# This includes all the classes and functions


# ==================== CELL 3: COLAB-SPECIFIC SETTINGS ====================
# Optimize settings for Colab

import torch
from torch import cuda

# Set up device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Reduce batch size for Colab's GPU memory
BATCH_SIZE = 8  # Reduced from 16
ECM_EPOCHS = 2  # Keep reduced for faster training

# Set number of workers to 0 for Colab (Windows compatibility)
# This is already set in load_dataset()

print("✓ Colab settings configured\n")


# ==================== CELL 4A: TRAINING MODE ====================
# Run this if you want to train the model

if __name__ == '__main__':
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "TRAINING MODE - ENHANCED FRACTAL TOKENIZATION".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    # Train on CIFAR10
    model, loss_history = test_tokenization_and_generation('CIFAR10', '/content/data')
    
    # Save to Google Drive if available
    try:
        import shutil
        shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/improved_lidecm.pt')
        print("✓ Model saved to Google Drive")
    except:
        print("⚠ Could not save to Google Drive (mount it first)")


# ==================== CELL 4B: INFERENCE MODE ====================
# Run this to use a pre-trained model for generation

if __name__ == '__main__':
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "INFERENCE MODE - TEXT-GUIDED IMAGE GENERATION".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    # Custom prompts for generation
    custom_prompts = [
        ("a serene mountain landscape", 1.0),
        ("abstract digital art", 1.3),
        ("vibrant sunset over ocean", 0.9),
        ("futuristic city architecture", 1.1),
        ("mystical forest scene", 1.2),
    ]
    
    # Run inference
    model = run_inference_only('/content/improved_lidecm.pt', custom_prompts)


# ==================== CELL 5: VISUALIZATION ====================
# Display and download results

from IPython.display import Image as IPImage, display
import glob

print("\n📊 Generated Images:\n")

# Display tokenization results
for filepath in glob.glob('/content/tokenization_results_*.png'):
    print(f"\n📷 {os.path.basename(filepath)}")
    display(IPImage(filepath))

# Display generated images
for filepath in glob.glob('/content/generated_images_*.png'):
    print(f"\n🎨 {os.path.basename(filepath)}")
    display(IPImage(filepath))

# Display inference results
if os.path.exists('/content/inference_results.png'):
    print(f"\n✨ Inference Results")
    display(IPImage('/content/inference_results.png'))

# Download results
print("\n⬇️  Download Results:")
print("Files saved in /content/")
print("  - tokenization_results_*.png")
print("  - generated_images_*.png")
print("  - inference_results.png")
print("  - improved_lidecm.pt (model checkpoint)")


# ==================== CELL 6: MONITOR TRAINING ====================
# Real-time monitoring during training

import matplotlib.pyplot as plt

def plot_training_losses(loss_history):
    """Plot training losses during learning phase"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(loss_history['epoch'], loss_history['total'], 'b-o', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Loss statistics
    axes[1].text(0.5, 0.8, 'Training Summary', ha='center', fontsize=14, fontweight='bold')
    axes[1].text(0.5, 0.6, f"Epochs: {len(loss_history['epoch'])}", ha='center', fontsize=12)
    axes[1].text(0.5, 0.5, f"Final Loss: {loss_history['total'][-1]:.6f}", ha='center', fontsize=12)
    axes[1].text(0.5, 0.4, f"Best Loss: {min(loss_history['total']):.6f}", ha='center', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/training_summary.png', dpi=150, bbox_inches='tight')
    display(IPImage('/content/training_summary.png'))


# ==================== CELL 7: ADVANCED OPTIONS ====================
# Custom configurations

# Option 1: Change dataset
# model, loss_history = test_tokenization_and_generation('CIFAR100', '/content/data')

# Option 2: Custom inference parameters
# prompts = [("your custom prompt", temperature), ...]
# model = run_inference_only('/content/improved_lidecm.pt', prompts)

# Option 3: Load checkpoint from Google Drive
# import shutil
# shutil.copy('/content/drive/MyDrive/improved_lidecm.pt', '/content/improved_lidecm.pt')
# model = run_inference_only('/content/improved_lidecm.pt')

# Option 4: GPU memory optimization
# torch.cuda.empty_cache()  # Clear GPU cache
# torch.cuda.reset_peak_memory_stats()  # Reset memory stats


# ==================== TROUBLESHOOTING ====================
"""
Common issues in Colab:

1. Out of Memory (OOM):
   - Reduce BATCH_SIZE (currently 8)
   - Reduce ECM_EPOCHS (currently 2)
   - Reduce NUM_LATENTS in constants
   - Use torch.cuda.empty_cache()

2. CLIP Model Download Issues:
   - Install: !pip install git+https://github.com/openai/CLIP.git
   - Or use: !pip install open-clip-torch

3. Slow Training:
   - Colab's GPU varies; use GPU runtime
   - Check: !nvidia-smi
   - Restart runtime if performance degrades

4. File Save/Load:
   - Use /content/ for temporary storage
   - Use /content/drive/MyDrive/ for permanent storage
   - Mount Google Drive first

5. Dependencies Issues:
   - Clear pip cache: !pip cache purge
   - Reinstall packages with: !pip install --upgrade [package]
"""
