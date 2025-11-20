# 🚀 How to Run in Google Colab - Step by Step

## Overview
This guide walks you through running the Enhanced Fractal Tokenization Model in Google Colab with GPU acceleration.

**Total Time:** ~25 minutes (training) or ~5 minutes (inference only)

---

## ✅ Quick Checklist
- [ ] Google Account (free)
- [ ] Google Drive (optional, for storage)
- [ ] ~5-10 GB free space on Drive (optional)
- [ ] Internet connection

---

## 📋 Step-by-Step Instructions

### STEP 1: Go to Google Colab
1. Open your browser
2. Go to **https://colab.research.google.com**
3. Sign in with your Google account
4. Click **"+ New notebook"**

### STEP 2: Enable GPU (CRITICAL for speed)
1. In the top menu, click **"Runtime"**
2. Click **"Change runtime type"**
3. Under "Hardware accelerator", select **"GPU"**
4. Click **"Save"**
   - ✅ This gives you a free NVIDIA GPU (takes ~30 seconds)

### STEP 3: Install Dependencies
Copy-paste this code into **Cell 1** and press **Ctrl+Enter**:

```
!pip install -q torch torchvision torchaudio
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q matplotlib numpy tqdm Pillow

import os
os.makedirs('/content/checkpoints', exist_ok=True)
os.makedirs('/content/data', exist_ok=True)

print("✅ Installation complete!")
```

**What happens:**
- ⏱ Takes 2-3 minutes
- 📦 Installs PyTorch, CLIP, and other libraries
- 📁 Creates folders for data and checkpoints

### STEP 4: Upload Your Code
**Option A: Direct Upload (Easiest)**
1. Click the 📁 **Files** icon on the left sidebar
2. Click **"Upload"** button (or drag and drop)
3. Select `enhancedlibcem.py` from your computer
4. Wait for upload to complete (shows ✓)

**Option B: From GitHub**
In a new cell:
```
!git clone https://github.com/YOUR_USERNAME/enhancedlibcem.git /content/repo
```

### STEP 5: Load the Code
In a new cell:
```
# Load the uploaded code
exec(open('/content/enhancedlibcem.py').read())

print("✅ Code loaded successfully!")
```

### STEP 6: Mount Google Drive (Optional but Recommended)
In a new cell:
```
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted at /content/drive")
```

**What happens:**
- Asks for permission (click "Connect to Google Drive")
- Allows you to save files permanently
- Access files across sessions

### STEP 7: Train OR Load Model
**Option A: Train a New Model**
In a new cell:
```
# Train the model (takes 15-20 minutes)
model, loss_history = test_tokenization_and_generation(
    dataset_name='CIFAR10',
    data_path='/content/data'
)
```

**What happens:**
- ⏱ Takes ~15-20 minutes with GPU
- 📥 Downloads CIFAR10 dataset (~170 MB)
- 🔄 Trains for 2 epochs
- 💾 Saves to `/content/improved_lidecm.pt`

**Option B: Use Pre-trained Model**
If you already have a trained model:
```
# Load model from Drive
import shutil
shutil.copy('/content/drive/MyDrive/improved_lidecm.pt', '/content/')

# Run inference only (takes 2-3 minutes)
model = run_inference_only('/content/improved_lidecm.pt')
```

### STEP 8: Generate Images
In a new cell:
```
# Define what images you want to generate
prompts = [
    ("a beautiful sunset over mountains", 1.0),
    ("abstract colorful digital art", 1.3),
    ("peaceful forest with waterfall", 0.9),
]

# Generate!
model = run_inference_only('/content/improved_lidecm.pt', prompts)
```

### STEP 9: View Results
In a new cell:
```
from IPython.display import Image, display

# Display the generated images
display(Image('/content/inference_results.png'))

# Display other results if available
display(Image('/content/generated_images_CIFAR10.png'))
```

### STEP 10: Save to Google Drive (Optional)
In a new cell:
```
import shutil

# Create folder in Drive
os.makedirs('/content/drive/MyDrive/Fractal_Results', exist_ok=True)

# Copy files
shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/Fractal_Results/')
shutil.copy('/content/inference_results.png', '/content/drive/MyDrive/Fractal_Results/')

print("✓ Files saved to Google Drive!")
```

---

## 🎯 Common Scenarios

### Scenario A: First Time Training
```
Cell 1: Install dependencies
Cell 2: (Skip - no drive mount needed)
Cell 3: Load code
Cell 4: (Skip - go straight to step 5)
Cell 5: exec(open('/content/enhancedlibcem.py').read())
Cell 6: model, loss = test_tokenization_and_generation('CIFAR10', '/content/data')
Cell 7: model = run_inference_only('/content/improved_lidecm.pt')
Cell 8: display(Image('/content/inference_results.png'))
```
**Total time: ~25 minutes**

### Scenario B: Use Saved Model (Next Day)
```
Cell 1: Install dependencies
Cell 2: Mount Drive
Cell 3: Load code from Drive
Cell 4: Copy model from Drive: shutil.copy('/content/drive/MyDrive/improved_lidecm.pt', '/content/')
Cell 5: Generate: model = run_inference_only('/content/improved_lidecm.pt')
Cell 6: Display results
```
**Total time: ~5 minutes**

### Scenario C: Fine-tune on Custom Data
```
Cell 1-3: Setup (same as Scenario A)
Cell 4: Upload your images to Colab
Cell 5: model, loss = test_tokenization_and_generation('CUSTOM', custom_folder='/content/my_images')
Cell 6-8: View and save results
```
**Total time: Variable (depends on dataset size)**

---

## ⚠️ Troubleshooting

### Problem: "No module named 'clip'"
```python
# Solution: Re-run this
!pip install git+https://github.com/openai/CLIP.git
```

### Problem: "CUDA out of memory"
```python
# Solution 1: Clear cache
import torch
torch.cuda.empty_cache()

# Solution 2: Reduce batch size (edit in code)
BATCH_SIZE = 4  # Change from 8

# Solution 3: Restart runtime
# Runtime → Restart runtime
```

### Problem: "The kernel died"
This means the GPU ran out of memory. 
```python
# Solutions:
1. Runtime → Restart runtime
2. Reduce BATCH_SIZE and ECM_EPOCHS
3. Try inference-only mode instead of training
```

### Problem: GPU not working
```python
# Check if GPU is enabled
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should show GPU name

# If False, go to Runtime → Change runtime type → Select GPU
```

### Problem: File upload is slow
```python
# Alternative: Download from GitHub instead
!git clone https://github.com/YOUR_USERNAME/repo.git /content/repo
```

### Problem: "Google Drive mount failed"
```python
# Try again in a new cell
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## 💡 Tips & Tricks

### Tip 1: Monitor Training Progress
```python
# Run this in a separate cell to monitor GPU
!watch -n 1 nvidia-smi  # Press Ctrl+C to stop
```

### Tip 2: Keep Model for Next Session
```python
# Save to Google Drive after training
import shutil
shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/my_model.pt')
```

### Tip 3: Faster Inference
```python
# Use smaller temperature for faster, more coherent results
prompts = [("your prompt", 0.8)]  # Lower temperature = faster convergence
```

### Tip 4: Check GPU Type
```python
import torch
print(torch.cuda.get_device_name(0))
# May show: Tesla T4, V100, A100 (better GPU = faster)
```

### Tip 5: Download Results
```python
# Click Files → Right-click on file → Download
# Or use this code:
from google.colab import files
files.download('/content/inference_results.png')
```

---

## 🔐 Safety Notes

1. **Don't share notebooks with API keys** - Colab can be shared publicly
2. **Keep Google Drive backups** - Colab sessions reset after ~12 hours of inactivity
3. **Check GPU quota** - You get limited free GPU hours per month (usually enough)
4. **Monitor disk usage** - You have ~50 GB free in `/content/`

---

## 📊 Expected Performance

| Stage | Time | Resources |
|-------|------|-----------|
| Install | 2-3 min | 200 MB |
| Download CIFAR10 | 3-5 min | 500 MB |
| Train (2 epochs) | 12-15 min | 4-5 GB GPU |
| Inference (5 images) | 2-3 min | 3-4 GB GPU |
| **Total (first time)** | **~25 min** | **~5 GB** |
| **Total (inference)** | **~5 min** | **~3 GB** |

---

## ✨ Next Steps

1. ✅ Complete first run (training)
2. ✅ Experiment with different prompts
3. ✅ Try different datasets (CIFAR100, custom images)
4. ✅ Fine-tune on custom data
5. ✅ Share results with friends!

---

## 🆘 Need Help?

- **Check COLAB_GUIDE.md** for detailed documentation
- **Check COLAB_READY.py** for complete working code
- **GitHub Issues** for bug reports
- **Colab Help** (? icon in Colab)

---

**Good luck! 🚀 Enjoy generating amazing images!**
