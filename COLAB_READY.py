"""
COPY THIS INTO GOOGLE COLAB - Enhanced Fractal Tokenization Model
Each # ===== CELL N ===== section goes into a separate cell
"""

# ===== CELL 1: INSTALL DEPENDENCIES =====
!pip install -q torch torchvision torchaudio
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q matplotlib numpy tqdm Pillow

import os
os.makedirs('/content/checkpoints', exist_ok=True)
os.makedirs('/content/data', exist_ok=True)

print("✅ Dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")


# ===== CELL 2: MOUNT GOOGLE DRIVE (OPTIONAL) =====
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted")
    HAS_DRIVE = True
except:
    print("⚠ Not in Colab or Drive mount failed")
    HAS_DRIVE = False


# ===== CELL 3: UPLOAD OR LOAD CODE =====
# OPTION A: If you uploaded enhancedlibcem.py file
exec(open('/content/enhancedlibcem.py').read())

# OPTION B: If loading from GitHub
# !git clone https://github.com/YOUR_USERNAME/enhancedlibcem.git /content/repo
# exec(open('/content/repo/enhancedlibcem.py').read())


# ===== CELL 4: CHECK MODEL STATUS =====
import os

checkpoint_exists = os.path.exists('/content/improved_lidecm.pt')

print("\n" + "="*80)
print("MODEL STATUS CHECK".center(80))
print("="*80)
print(f"\nCheckpoint exists: {'✓ YES' if checkpoint_exists else '✗ NO'}")

if checkpoint_exists:
    file_size = os.path.getsize('/content/improved_lidecm.pt') / (1024**2)
    print(f"Checkpoint size: {file_size:.2f} MB")
    print("\n🎯 INFERENCE MODE - Using pre-trained model")
    print("   Run Cell 5 to generate images")
else:
    print("\n📚 TRAINING MODE - Training new model")
    print("   Run Cell 5 to train")


# ===== CELL 5A: TRAINING (if model doesn't exist) =====
# Uncomment the lines below to train

# if not os.path.exists('/content/improved_lidecm.pt'):
#     print("\n" + "█"*80)
#     print("█" + " "*78 + "█")
#     print("█" + "STARTING TRAINING".center(78) + "█")
#     print("█" + " "*78 + "█")
#     print("█"*80 + "\n")
#     
#     model, loss_history = test_tokenization_and_generation(
#         dataset_name='CIFAR10',
#         data_path='/content/data'
#     )
#     
#     # Save to Google Drive if mounted
#     if HAS_DRIVE:
#         import shutil
#         try:
#             shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/improved_lidecm.pt')
#             print("\n✓ Model saved to Google Drive")
#         except:
#             print("\n⚠ Could not save to Drive (check permissions)")


# ===== CELL 5B: INFERENCE (use pre-trained model) =====
# Run this to generate images

print("\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + "INFERENCE - TEXT-GUIDED IMAGE GENERATION".center(78) + "█")
print("█" + " "*78 + "█")
print("█"*80 + "\n")

# Define custom prompts
custom_prompts = [
    ("a serene mountain landscape at sunset", 1.0),
    ("abstract digital art with vibrant colors", 1.3),
    ("a peaceful forest with waterfall", 0.9),
    ("futuristic cyberpunk city skyline", 1.1),
    ("mystical aurora borealis in night sky", 1.2),
]

print(f"Generating {len(custom_prompts)} images...\n")

# Run inference
try:
    model = run_inference_only('/content/improved_lidecm.pt', custom_prompts)
    print("✅ Generation complete!")
except FileNotFoundError:
    print("❌ Model checkpoint not found!")
    print("   Please run Cell 5A to train a model first")


# ===== CELL 6: DISPLAY RESULTS =====
from IPython.display import Image as IPImage, display
from pathlib import Path

print("\n" + "="*80)
print("GENERATED RESULTS".center(80))
print("="*80 + "\n")

result_files = [
    '/content/inference_results.png',
    '/content/generated_images_CIFAR10.png',
    '/content/tokenization_results_CIFAR10.png',
]

for filepath in result_files:
    if Path(filepath).exists():
        print(f"\n📊 {Path(filepath).stem.replace('_', ' ').upper()}\n")
        display(IPImage(filepath))
    else:
        print(f"⚠ {filepath} not found")

# Show checkpoint info
if os.path.exists('/content/improved_lidecm.pt'):
    size_mb = os.path.getsize('/content/improved_lidecm.pt') / (1024**2)
    print(f"\n💾 Model checkpoint: {size_mb:.2f} MB")


# ===== CELL 7: SAVE RESULTS TO DRIVE =====
# Save all results to Google Drive

if HAS_DRIVE:
    import shutil
    from pathlib import Path
    
    print("\n📤 Saving results to Google Drive...\n")
    
    drive_path = '/content/drive/MyDrive/Enhanced_Fractal_Results'
    os.makedirs(drive_path, exist_ok=True)
    
    files_to_save = [
        '/content/improved_lidecm.pt',
        '/content/inference_results.png',
        '/content/generated_images_CIFAR10.png',
        '/content/tokenization_results_CIFAR10.png',
        '/content/best_model.pt',
    ]
    
    for filepath in files_to_save:
        if Path(filepath).exists():
            try:
                shutil.copy(filepath, drive_path)
                print(f"✓ {Path(filepath).name}")
            except:
                print(f"⚠ Failed to save {Path(filepath).name}")
    
    print(f"\n✅ Saved to: {drive_path}")
else:
    print("⚠ Google Drive not mounted - results saved locally")
    print("   Download from /content/ using the file manager")


# ===== CELL 8: MONITOR GPU USAGE =====
!nvidia-smi

print("\n📊 Training Summary (if trained in this session):")
print(f"   Device: {DEVICE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {ECM_EPOCHS}")


# ===== CELL 9: MEMORY MANAGEMENT =====
import torch

print("🧹 Memory Management\n")

# Clear GPU cache
torch.cuda.empty_cache()
print("✓ GPU cache cleared")

# Check memory
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    
    print(f"\n📈 GPU Memory Stats:")
    print(f"   Total: {total_memory:.2f} GB")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")


# ===== CELL 10: CUSTOM GENERATION =====
# Generate more images with custom prompts

custom_prompts = [
    ("your custom prompt here", 1.0),
    ("another interesting prompt", 1.2),
    ("yet another creative prompt", 0.9),
]

print("🎨 Custom Generation\n")
print("Modify the prompts above and run this cell\n")

try:
    model = run_inference_only('/content/improved_lidecm.pt', custom_prompts)
    display(IPImage('/content/inference_results.png'))
except Exception as e:
    print(f"Error: {e}")


# ===== CELL 11: TROUBLESHOOTING =====
"""
Common issues and solutions:

1. ❌ "No module named 'clip'"
   ✓ Solution: !pip install git+https://github.com/openai/CLIP.git

2. ❌ "CUDA out of memory"
   ✓ Solutions:
      - Restart runtime: Runtime → Restart runtime
      - Reduce: BATCH_SIZE = 4, ECM_EPOCHS = 1
      - Clear cache: torch.cuda.empty_cache()

3. ❌ "Google Drive mount failed"
   ✓ Solution: Run Cell 2 again, click the auth link

4. ❌ "File not found: enhancedlibcem.py"
   ✓ Solutions:
      - Upload file via Files panel (left side)
      - Or use: !wget https://github.com/YOUR/enhancedlibcem.py

5. ❌ "The kernel died unexpectedly"
   ✓ Solution: Runtime → Restart runtime, then reduce model size

6. ❌ Slow training
   ✓ Check: !nvidia-smi (should show 100% GPU utilization)
   ✓ If not, restart runtime and enable GPU in Runtime settings
"""

# Run this to check current status
import torch
print("✅ Current Status:")
print(f"   PyTorch: {torch.__version__}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
print(f"   CUDA: {torch.version.cuda}")
